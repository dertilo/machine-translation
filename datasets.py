import os
from typing import Tuple, Union, List

from nlp import load_dataset
from seq2seq.utils import SortishSampler
from torch.utils.data import Dataset
from transformers import (
    MBartTokenizer,
    AutoTokenizer,
    BartTokenizer,
    MarianTokenizer,
    BatchEncoding,
)


def marian_tokenize(
    src_texts: List[str],
    tgt_texts: List[str],
    tok: MarianTokenizer,
    max_src_tgt_len: Tuple[int, int],
):
    # based on: https://github.com/dertilo/transformers/blob/281e394889b33d0650bcd13f120c3f75a799679a/src/transformers/tokenization_marian.py#L125
    msrcl, mtgtl = max_src_tgt_len
    tok.current_spm = tok.spm_source
    # src_texts = [tok.normalize(t) for t in src_texts]  # this does not appear to do much -> TODO(tilo): WTF!?
    tokenizer_kwargs = dict(
        add_special_tokens=True,
        return_tensors="pt",
        max_length=msrcl,
        pad_to_max_length=True,
        truncation_strategy="only_first",
        padding="max_length",  # TODO(tilo): redundant!!
    )
    model_inputs: BatchEncoding = tok(src_texts, **tokenizer_kwargs)
    tok.current_spm = tok.spm_target
    tokenizer_kwargs["max_length"] = mtgtl
    decoder_inputs: BatchEncoding = tok(tgt_texts, **tokenizer_kwargs)
    tok.current_spm = tok.spm_source
    return decoder_inputs, model_inputs


import nlp


class Text(nlp.GeneratorBasedBuilder):
    def _info(self):
        return nlp.DatasetInfo(features=nlp.Features({"text": nlp.Value("string"),}))

    def _split_generators(self, dl_manager):
        """ The `datafiles` kwarg in load_dataset() can be a str, List[str], Dict[str,str], or Dict[str,List[str]].

            If str or List[str], then the dataset returns only the 'train' split.
            If dict, then keys should be from the `nlp.Split` enum.
        """
        if isinstance(self.config.data_files, (str, list, tuple)):
            # Handle case with only one split
            files = self.config.data_files
            if isinstance(files, str):
                files = [files]
            return [
                nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"files": files})
            ]
        else:
            # Handle case with several splits and a dict mapping
            splits = []
            for split_name in [nlp.Split.TRAIN, nlp.Split.VALIDATION, nlp.Split.TEST]:
                if split_name in self.config.data_files:
                    files = self.config.data_files[split_name]
                    if isinstance(files, str):
                        files = [files]
                    splits.append(
                        nlp.SplitGenerator(name=split_name, gen_kwargs={"files": files})
                    )
            return splits

    def _generate_examples(self, files):
        """ Read files sequentially, then lines sequentially. """
        idx = 0
        for filename in files:
            with open(filename) as file:
                for line in file:
                    yield idx, {"text": line}
                    idx += 1


def load_dataset_offline(name, data_files, cache_dir):
    builder_instance = Text(
        cache_dir=cache_dir,
        name=name,
        version=None,
        data_dir=None,
        data_files=data_files,
        hash=None,
        features=None,
    )
    builder_instance.download_and_prepare(
        download_config=None, download_mode=None, ignore_verifications=True,
    )
    ds = builder_instance.as_dataset(split="train", ignore_verifications=True)
    return ds


class TranslationDataset(Dataset):
    def __init__(
        self,
        tokenizer: MarianTokenizer,
        data_dir: str,
        type_path="train",
        max_src_tgt_len=(1024, 56),
        prefix="",
    ):
        super().__init__()
        self.prefix = prefix
        self.type_path = type_path
        self.tokenizer = tokenizer
        self.max_src_tgt_len = max_src_tgt_len

        path = os.path.join(data_dir, type_path)
        cache_dir = "huggingface_cache"

        self.src = load_dataset_offline(
            f"{type_path}-src", [f"{path}.source"], cache_dir
        )
        self.tgt = load_dataset_offline(
            f"{type_path}-tgt", [f"{path}.target"], cache_dir
        )

        self.pad_token_id = tokenizer.pad_token_id

    def _preprocess(self, text: str):
        return self.prefix + text.strip()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        index = int(index)
        src_text = self._preprocess(self.src[index]["text"])
        tgt_text = self._preprocess(self.tgt[index]["text"])

        decoder_inputs, model_inputs = marian_tokenize(
            [src_text], [tgt_text], self.tokenizer, self.max_src_tgt_len
        )
        assert model_inputs.input_ids.shape[1] == self.max_src_tgt_len[0]
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
        }

    def make_sortish_sampler(self, batch_size):
        num_chars = [list(range(len(d["text"]))) for d in self.src]
        return SortishSampler(num_chars, batch_size)


if __name__ == "__main__":

    dataset = TranslationDataset(
        tokenizer=AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-ro", cache_dir="cache_dir",
        ),
        type_path="train",
        data_dir=os.environ["HOME"] + "/code/NLP/MT/machine-translation/some_data",
    )

    print(dataset[0])
