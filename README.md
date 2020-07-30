# machine-translation
* [quantization](https://pytorch.org/docs/stable/quantization.html)
* [salience](https://arxiv.org/pdf/1906.10282.pdf)
* [huggingface-marian-mt](https://huggingface.co/transformers/model_doc/marian.html)

## huggingface transformers
### setup
* (optional) if on compute node without internet connection → due to _wandb on compute-node socket timeout issue_ pip install: `https://github.com/dertilo/client` which contains hacky patch
* install transformers
```shell script
git clone https://github.com/dertilo/transformers.git
cd transformers && git checkout lightning_examples
pip install -e .
cd examples
pip install -r requirements.txt
```
* install apex
```shell script
export CUDA_HOME=/usr/local/cuda-10.1
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

#### WMT16 English-Romanian 
* get data
```shell script
wget https://s3.amazonaws.com/datasets.huggingface.co/translation/wmt_en_ro.tar.gz
tar -xzvf wmt_en_ro.tar.gz
cat wmt_en_ro/train.source | wc -l
610319
```
* train
```shell script
OMP_NUM_THREADS=2 wandb init # on frontend

export PYTHONPATH=~/transformers/examples

CUDA_VISIBLE_DEVICES=0 WANDB_MODE=dryrun python ../transformers/examples/seq2seq/finetune.py \
--data_dir=$HOME/data/parallel_text_corpora/wmt_en_ro \
--model_name_or_path=Helsinki-NLP/opus-mt-en-ro \
--learning_rate=3e-5 \
--max_source_length=128 \
--max_target_length=128 \
--train_batch_size=32 \
--eval_batch_size=32 \
--output_dir=en-ro-helsinki \
--num_train_epochs 10 \
--fp16 \
--gpus 1 \
--do_train \
--do_predict \
--n_val 1000 \
--val_check_interval 0.1 \
--sortish_sampler \
--logger wandb \
--wandb_project machine-translation
```
* [en-ro training progress](https://app.wandb.ai/dertilo/machine-translation/runs/20inpc06/overview?workspace=user-)
* evaluate # TODO(tilo)
```shell script
python transformers/examples/seq2seq/run_eval.py Helsinki-NLP/opus-mt-en-ro wmt_en_ro/test.source output.txt  --reference_path wmt_en_ro/test.target --score_path scores.json --task translation --bs 32 --fp16
python ../transformers/examples/seq2seq/run_eval.py ~/data/parallel_text_corpora/wmt_en_ro/test.source output.txt Helsinki-NLP/opus-mt-en-ro --reference_path ~/data/parallel_text_corpora/wmt_en_ro/test.target --score_path scores.json --metric bleu --bs 32 --fp16
cat scores.json
{"bleu": 27.651824005955024}
CUDA_VISIBLE_DEVICES=1 python ../transformers/examples/seq2seq/run_eval.py ~/data/parallel_text_corpora/wmt_en_ro/test.source output.txt en-ro-helsinki/best_tfmr --reference_path ~/data/parallel_text_corpora/wmt_en_ro/test.target --score_path scores.json --metric bleu --bs 32 --fp16
```
##### distillation
```shell script
python machine-translation/distillation.py \
--data_dir=some_data \
--src_lang=en_XX \
--tgt_lang=ro_RO \
--model_name_or_path IGNORED \
--learning_rate=3e-4 \
--train_batch_size=4 \
--eval_batch_size=4 \
--teacher Helsinki-NLP/opus-mt-en-ro \
--tokenizer_name  Helsinki-NLP/opus-mt-en-ro \
--warmup_steps 500 \
--student_decoder_layers 2 --student_encoder_layers 2 \
--freeze_embeds \
--alpha_hid=3. --length_penalty=0.5 \
--gradient_accumulation_steps=2 \
--max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
--output_dir=debug \
--num_train_epochs 3 \
--gpus 1 \
--fp16 \
--do_train \
--do_predict \
--val_check_interval 0.2 \
--sortish_sampler \
--logger wandb \
--wandb_project machine-translation
```

# datasets
* some en-de data #TODO(tilo): unused
```shell script
wget https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz
TEXT=wmt_ende_sp
mkdir -p $TEXT
tar -xzvf $TEXT.tar.gz -C $TEXT
```
* [movie-subtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php)
* [europal](https://www.statmt.org/europarl/)
`opus_read  -d ParaCrawl   --source en     --target es     --preprocess raw    --leave_non_alignments_out -w en_es.txt`

## libraries
* [opus train](https://github.com/Helsinki-NLP/OPUS-MT-train)
* [marian](https://github.com/marian-nmt/marian)
* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* [masakhane-mt](https://github.com/masakhane-io/masakhane-mt)
* https://github.com/facebookresearch/UnsupervisedMT
* [fairseq+wandb](https://github.com/Guitaricet/fairseq)