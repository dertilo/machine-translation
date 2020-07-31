from functools import partial

import shutil

import argparse
import glob
import logging
import os
import time
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from lightning_base import BaseTransformer, add_generic_args, generic_train
from transformers import get_linear_schedule_with_warmup, MBartTokenizer

from seq2seq.utils import (
    assert_all_frozen,
    lmap,
    flatten_list,
    pickle_save,
    save_json,
    freeze_params,
    calculate_rouge,
    get_git_info,
    ROUGE_KEYS,
    calculate_bleu_score,
)
from seq2seq.callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback

from common import (
    calc_loss,
    DataSetType,
    collate_fn,
)
from datasets import TranslationDataset

logger = logging.getLogger(__name__)


class Seq2SeqTransformer(BaseTransformer):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        # use_task_specific_params(self.model, "summarization")#TODO(tilo): what is this good for?
        # save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)

        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None
        if self.model.config.decoder_start_token_id is None and isinstance(
            self.tokenizer, MBartTokenizer
        ):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[
                hparams.tgt_lang
            ]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _calc_losses(self, batch: dict) -> Tuple:
        loss = calc_loss(batch, self.model, self.tokenizer.pad_token_id)
        return (loss,)

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._calc_losses(batch)
        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {
            k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names
        }
        loss = losses["loss"]
        rouges = {
            k: np.array([x[k] for x in outputs]).mean()
            for k in self.metric_names + ["gen_time", "summ_len"]
        }
        rouge_tensor: torch.FloatTensor = torch.tensor(rouges[self.val_metric]).type_as(
            loss
        )
        rouges.update({k: v.item() for k, v in losses.items()})
        losses.update(rouges)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {
            "log": metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": rouge_tensor,
        }

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        save_json(self.metrics, self.metrics_save_path)

    @abstractmethod
    def build_dataset(self, dataset_type: DataSetType):
        raise NotImplementedError

    @abstractmethod
    def calc_generative_metrics(self, preds, target) -> Dict:
        raise NotImplementedError

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["decoder_input_ids"])
        loss_tensors = self._calc_losses(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(
            gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **rouge
        )
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataloader(self, type_path: str, batch_size: int) -> DataLoader:
        dataset = self.build_dataset(type_path)
        if self.hparams.sortish_sampler and type_path == "train":
            assert self.hparams.gpus <= 1  # TODO: assert earlier
            sampler = dataset.make_sortish_sampler(batch_size)
            shuffle = False
        else:
            shuffle = True
            sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=partial(collate_fn, pad_token_id=self.tokenizer.pad_token_id),
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            sampler=sampler,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size)

        t_total = (
            (
                len(dataloader.dataset)
                // (self.hparams.train_batch_size * max(1, self.hparams.gpus))
            )
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total,
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        # fmt: off
        parser.add_argument(
            "--max_source_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=56,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=142,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=142,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            required=True,
            help="The input data dir. Should contain train.source, train.target, val.source, val.target, test.source, test.target",
        )
        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--logger", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        # parser.add_argument("--wandb_project", type=str, default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=500, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        # fmt: on
        return parser


class SummarizationModule(Seq2SeqTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    val_metric = "rouge2"

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)


class TranslationModule(Seq2SeqTransformer):
    mode = "translation"
    loss_names = ["loss"]
    metric_names = ["bleu"]
    val_metric = "bleu"

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

        if self.model.config.decoder_start_token_id is None and isinstance(
            self.tokenizer, MBartTokenizer
        ):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[
                hparams.tgt_lang
            ]

    def calc_generative_metrics(self, preds, target) -> dict:
        return calculate_bleu_score(preds, target)

    def build_dataset(self, dataset_type: DataSetType):
        hparams = self.hparams

        n_observations_per_split = {
            "train": hparams.n_train,
            "val": hparams.n_val,
            "test": hparams.n_test,
        }
        n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        target_lens = {
            "train": hparams.max_target_length,
            "val": hparams.val_max_target_length,
            "test": hparams.test_max_target_length,
        }
        assert target_lens["train"] <= target_lens["val"], f"target_lens: {target_lens}"
        assert (
            target_lens["train"] <= target_lens["test"]
        ), f"target_lens: {target_lens}"

        type_path = dataset_type.name
        max_target_length = target_lens[type_path]

        dataset = TranslationDataset(
            self.tokenizer,
            type_path=type_path,
            max_src_tgt_len=(hparams.max_source_length, max_target_length),
            data_dir=hparams.data_dir,
            prefix=self.model.config.prefix or "",
        )

        return dataset

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser = Seq2SeqTransformer.add_model_specific_args(parser, root_dir)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        return parser


def main(args, model=None) -> Seq2SeqTransformer:
    if args.output_dir == "debug":
        shutil.rmtree(args.output_dir)
    Path(args.output_dir).mkdir(exist_ok=True)
    if len(os.listdir(args.output_dir)) > 3 and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir
            )
        )

    dataset = Path(args.data_dir).name
    if (
        args.logger == "default"
        # or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=args.wandb_project)

    elif args.logger == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")
    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(args.output_dir, model.val_metric),
        logger=logger,
        # TODO: early stopping callback seems messed up
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    checkpoints = list(
        sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True))
    )
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)
    trainer.test(
        model
    )  # this breaks in DDP, known lightning issue. See evaluate_checkpoint to recover metrics.
    return model


if __name__ == "__main__":
    debug_args = """
--data_dir=some_data \
--src_lang=en_XX \
--tgt_lang=ro_RO \
--model_name_or_path=sshleifer/tiny-mbart \
--learning_rate=3e-5 \
--train_batch_size=32 \
--eval_batch_size=32 \
--output_dir=debug \
--num_train_epochs 10 \
--gpus 0 \
--do_train \
--do_predict \
--n_val 1000 \
--val_check_interval 0.1 \
--sortish_sampler \
    """.strip().split()
    parser = argparse.ArgumentParser()
    parser = TranslationModule.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args(debug_args)

    main(args, model=TranslationModule(args))
