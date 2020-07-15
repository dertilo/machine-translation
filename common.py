from enum import Enum

from torch.utils.data import DataLoader, Dataset

DataSetType = Enum("DataSetType", "train val test")


def build_dataloader(hparams, dataset: Dataset, shuffle: bool = False) -> DataLoader:

    batch_size = getattr(
        hparams,
        "train_batch_size" if dataset.type_path == "train" else "eval_batch_size",
    )
    sampler = None
    if hparams.sortish_sampler and dataset.type_path == "train":
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
