from enum import Enum

from torch import nn
from torch.nn import Module
from typing import Dict

from functools import partial

import torch
from seq2seq.utils import trim_batch

from torch.utils.data import DataLoader, Dataset

DataSetType = Enum("DataSetType", "train val test")


def build_dataloader(
    hparams,
    dataset: Dataset,
    pad_token_id: int,
    batch_size: int,
    shuffle: bool = False,
    sampler=None,
) -> DataLoader:

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
        shuffle=shuffle,
        num_workers=hparams.num_workers,
        sampler=sampler,
    )
    return dataloader


CELOSS_IGNORE_IDX = nn.CrossEntropyLoss().ignore_index


def calc_loss(batch:Dict, model:Module, pad_token_id:int):
    source_ids, source_mask, y = (
        batch["input_ids"],
        batch["attention_mask"],
        batch["decoder_input_ids"],
    )
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone()
    lm_labels[y[:, 1:] == pad_token_id] = CELOSS_IGNORE_IDX
    outputs = model(
        source_ids,
        attention_mask=source_mask,
        decoder_input_ids=y_ids,
        labels=lm_labels,
    )
    loss = outputs[0]
    return loss


def trim_seq2seq_batch(batch, pad_token_id):
    y = trim_batch(batch["decoder_input_ids"], pad_token_id)
    source_ids, source_mask = trim_batch(
        batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"]
    )
    return source_ids, source_mask, y


def collate_fn(batch, pad_token_id) -> dict:
    input_ids = torch.stack([x["input_ids"] for x in batch])
    masks = torch.stack([x["attention_mask"] for x in batch])
    target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
    y = trim_batch(target_ids, pad_token_id)
    source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
    batch = {
        "input_ids": source_ids,
        "attention_mask": source_mask,
        "decoder_input_ids": y,
    }
    return batch
