import os

from seq2seq.utils import encode_file, lmap, SortishSampler, trim_batch

try:
    import git
except:
    print("NO GIT on HPC!")

from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        type_path="train",
        max_source_length=1024,
        max_target_length=56,
        n_obs=None,
        overwrite_cache=False,
        prefix="",
        src_lang=None,
        tgt_lang=None,
    ):
        super().__init__()
        self.type_path = type_path
        # FIXME: the rstrip logic strips all the chars, it seems.
        tok_name = tokenizer.__class__.__name__.lower().rstrip("tokenizer")
        if hasattr(tokenizer, "set_lang") and src_lang is not None:
            tokenizer.set_lang(src_lang)  # HACK: only applies to mbart
        self.source = encode_file(
            tokenizer,
            os.path.join(data_dir, type_path + ".source"),
            max_source_length,
            overwrite_cache=overwrite_cache,
            prefix=prefix,
            tok_name=tok_name,
        )
        tgt_path = os.path.join(data_dir, type_path + ".target")
        if hasattr(tokenizer, "set_lang"):
            assert (
                tgt_lang is not None
            ), "--tgt_lang must be passed to build a translation"
            tokenizer.set_lang(tgt_lang)  # HACK: only applies to mbart
        self.target = encode_file(
            tokenizer,
            tgt_path,
            max_target_length,
            overwrite_cache=overwrite_cache,
            tok_name=tok_name,
        )
        if n_obs is not None:
            self.source = self.source[:n_obs]
            self.target = self.target[:n_obs]
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @property
    def src_lens(self):  # Can delete?
        return lmap(len, self.source)

    @property
    def tgt_lens(self):
        return lmap(len, self.target)

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.source, batch_size)
