from seq2seq.utils import SummarizationDataset
from torch.utils.data import DataLoader


def build_dataset(
    n_obs1, target_lens, tokenizer, dataset_kwargs, type_path
) -> SummarizationDataset:
    n_obs = n_obs1[type_path]
    max_target_length = target_lens[type_path]
    dataset = SummarizationDataset(
        tokenizer,
        type_path=type_path,
        n_obs=n_obs,
        max_target_length=max_target_length,
        **dataset_kwargs,
    )
    return dataset


def get_n_obs(hparams):
    n_observations_per_split = {
        "train": hparams.n_train,
        "val": hparams.n_val,
        "test": hparams.n_test,
    }
    n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}
    return n_obs


def get_target_lens(hparams):
    target_lens = {
        "train": hparams.max_target_length,
        "val": hparams.val_max_target_length,
        "test": hparams.test_max_target_length,
    }
    assert target_lens["train"] <= target_lens["val"], f"target_lens: {target_lens}"
    assert target_lens["train"] <= target_lens["test"], f"target_lens: {target_lens}"
    return target_lens


def get_dataset_kwargs(hparams, model):
    dataset_kwargs = dict(
        data_dir=hparams.data_dir,
        max_source_length=hparams.max_source_length,
        prefix=model.config.prefix or "",
    )
    return dataset_kwargs


def build_dataloader(
    hparams, tokenizer, model, type_path: str, shuffle: bool = False
) -> DataLoader:
    dataset = build_dataset(
        get_n_obs(hparams),
        get_target_lens(hparams),
        tokenizer,
        get_dataset_kwargs(hparams, model),
        type_path,
    )

    batch_size = getattr(
        hparams, "train_batch_size" if type_path == "train" else "eval_batch_size",
    )
    sampler = None
    if hparams.sortish_sampler and type_path == "train":
        assert hparams.gpus <= 1  # TODO: assert earlier
        sampler = dataset.make_sortish_sampler(batch_size)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        num_workers=hparams.num_workers,
        sampler=sampler,
    )
    return dataloader


def calc_loss(batch, model, pad_token_id):
    source_ids, source_mask, y = (
        batch["input_ids"],
        batch["attention_mask"],
        batch["decoder_input_ids"],
    )
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone()
    lm_labels[y[:, 1:] == pad_token_id] = -100
    outputs = model(
        source_ids,
        attention_mask=source_mask,
        decoder_input_ids=y_ids,
        labels=lm_labels,
    )
    loss = outputs[0]
    return loss
