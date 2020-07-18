import os
from typing import Tuple

from nlp import load_dataset
from seq2seq.utils import SortishSampler
from torch.utils.data import Dataset
from transformers import MBartTokenizer, AutoTokenizer, BartTokenizer


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer: MBartTokenizer,
        data_dir,
        type_path="train",
        max_src_tgt_len=(1024, 56),
        langs=Tuple[str, str],
        prefix="",
    ):
        super().__init__()
        self.prefix = prefix
        self.type_path = type_path
        self.tokenizer = tokenizer
        self.max_src_tgt_len = max_src_tgt_len
        self.langs = langs

        path = os.path.join(data_dir, type_path)
        dummy_split = "train"  # TODO(tilo): WTF!
        self.src = load_dataset(
            "text",
            name=f"{type_path}-src",
            data_files=[f"{path}.source"],
            cache_dir="huggingface_cache",
            split=dummy_split,
        )
        self.tgt = load_dataset(
            "text",
            name=f"{type_path}-tgt",
            data_files=[f"{path}.target"],
            cache_dir="huggingface_cache",
            split=dummy_split,
        )
        self.pad_token_id = tokenizer.pad_token_id

    def _tokenize(
        self,
        text: str,
        lang: str,
        max_length,
        pad_to_max_length=True,
        return_tensors="pt",
    ):

        extra_kw = (
            {"add_prefix_space": True}
            if isinstance(self.tokenizer, BartTokenizer)
            else {}
        )

        self.tokenizer.set_lang(lang)
        tokenized = self.tokenizer(
            [text],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            **extra_kw,
        )
        assert tokenized.input_ids.shape[1] == max_length
        return tokenized

    def _preprocess(self, text: str):
        return self.prefix + text.strip()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        index = int(index)
        srcl, tgtl = self.langs
        msrcl, mtgtl = self.max_src_tgt_len
        src = self._tokenize(self._preprocess(self.src[index]["text"]), srcl, msrcl)
        tgt = self._tokenize(self._preprocess(self.tgt[index]["text"]), tgtl, mtgtl)

        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "decoder_input_ids": tgt["input_ids"],
        }


    def make_sortish_sampler(self, batch_size):
        num_chars = [list(range(len(d["text"]))) for d in self.src]
        return SortishSampler(num_chars, batch_size)


if __name__ == "__main__":

    dataset = Seq2SeqDataset(
        tokenizer=AutoTokenizer.from_pretrained(
            "sshleifer/tiny-mbart", cache_dir="cache_dir",
        ),
        type_path="train",
        data_dir=os.environ["HOME"] + "/code/NLP/MT/machine-translation/some_data",
        langs=("en_XX", "ro_RO"),
    )

    print(dataset.src[0])
