# machine-translation

# datasets
* [movie-subtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php)
* [europal](https://www.statmt.org/europarl/)

# [fairseq-mt](https://github.com/pytorch/fairseq/tree/master/examples/translation)
* [models](https://modelzoo.co/model/fairseq-py)
## setup
```shell script
conda create -n fairseq python=3.7 -y
conda activate fairseq
cd fairseq && OMP_NUM_THREADS=4 pip install -e .
pip install sacremoses
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch # if using CUDA Version: 10.1
```
## IWSLT'14 German to English [see](https://github.com/pytorch/fairseq/tree/master/examples/translation)
```shell script
[de] Dictionary: 7432 types
[de] examples/translation/iwslt14.tokenized.de-en/train.de: 12928 sents, 320235 tokens, 0.0% replaced by <unk>
[de] Dictionary: 7432 types
[de] examples/translation/iwslt14.tokenized.de-en/valid.de: 587 sents, 14527 tokens, 0.145% replaced by <unk>
[de] Dictionary: 7432 types
[de] examples/translation/iwslt14.tokenized.de-en/test.de: 6750 sents, 163940 tokens, 0.28% replaced by <unk>
[en] Dictionary: 5912 types
[en] examples/translation/iwslt14.tokenized.de-en/train.en: 12928 sents, 311593 tokens, 0.0% replaced by <unk>
[en] Dictionary: 5912 types
[en] examples/translation/iwslt14.tokenized.de-en/valid.en: 587 sents, 14324 tokens, 0.0977% replaced by <unk>
[en] Dictionary: 5912 types
[en] examples/translation/iwslt14.tokenized.de-en/test.en: 6750 sents, 159240 tokens, 0.139% replaced by <unk>
```
* on __frontend__ Download and prepare the data*
```shell script
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..
```
* on __node__ Preprocess/binarize the data
```shell script
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```
* on __node__ train it
* check that GPUs are recognized: `CUDA_VISIBLE_DEVICES=0 python -c 'import torch; print(torch.cuda.device_count())'`
* [error](https://github.com/pytorch/pytorch/issues/37377): `Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library` -> `MKL_THREADING_LAYER=GNU`
```shell script
MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    -s de -t en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```
* evaluate
```shell script
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe --gen-subset test
```

## WMT16
### huggingface transformers
* 
```shell script
wget https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz
TEXT=wmt_ende_sp
mkdir -p $TEXT
tar -xzvf $TEXT.tar.gz -C $TEXT
```
### [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/scaling_nmt/README.md)
* [scaling-nmt](https://arxiv.org/pdf/1806.00187.pdf)

```shell script
2020-07-02 16:40:54 | INFO | fairseq_cli.preprocess | [en] Dictionary: 32768 types
2020-07-02 16:42:25 | INFO | fairseq_cli.preprocess | [en] /home/users/t/tilo-himmelsbach/data/wmt16_en_de_bpe32k/train.tok.clean.bpe.32000.en: 4500966 sents, 132886171 tokens, 0.00786% replaced by <unk>
2020-07-02 16:42:25 | INFO | fairseq_cli.preprocess | [en] Dictionary: 32768 types
2020-07-02 16:42:26 | INFO | fairseq_cli.preprocess | [en] /home/users/t/tilo-himmelsbach/data/wmt16_en_de_bpe32k/newstest2013.tok.bpe.32000.en: 3000 sents, 78126 tokens, 0.00128% replaced by <unk>
2020-07-02 16:42:26 | INFO | fairseq_cli.preprocess | [en] Dictionary: 32768 types
2020-07-02 16:42:26 | INFO | fairseq_cli.preprocess | [en] /home/users/t/tilo-himmelsbach/data/wmt16_en_de_bpe32k/newstest2014.tok.bpe.32000.en: 3003 sents, 82800 tokens, 0.0% replaced by <unk>
2020-07-02 16:42:26 | INFO | fairseq_cli.preprocess | [de] Dictionary: 32768 types
2020-07-02 16:43:59 | INFO | fairseq_cli.preprocess | [de] /home/users/t/tilo-himmelsbach/data/wmt16_en_de_bpe32k/train.tok.clean.bpe.32000.de: 4500966 sents, 137628246 tokens, 0.00719% replaced by <unk>
2020-07-02 16:43:59 | INFO | fairseq_cli.preprocess | [de] Dictionary: 32768 types
2020-07-02 16:44:00 | INFO | fairseq_cli.preprocess | [de] /home/users/t/tilo-himmelsbach/data/wmt16_en_de_bpe32k/newstest2013.tok.bpe.32000.de: 3000 sents, 83913 tokens, 0.00596% replaced by <unk>
2020-07-02 16:44:00 | INFO | fairseq_cli.preprocess | [de] Dictionary: 32768 types
2020-07-02 16:44:01 | INFO | fairseq_cli.preprocess | [de] /home/users/t/tilo-himmelsbach/data/wmt16_en_de_bpe32k/newstest2014.tok.bpe.32000.de: 3003 sents, 87313 tokens, 0.0% replaced by <unk>

```
* preprocess
```shell script
export TEXT=~/data/wmt16_en_de_bpe32k
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20
```
* train
```shell script
MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
    data-bin/wmt16_en_de_bpe32k \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16 --tensorboard-logdir tensorboard_logdir
```
* evaluate
```shell script
fairseq-generate data-bin/wmt16_en_de_bpe32k \
    --path checkpoints/checkpoint4.pt \
    --beam 4 --lenpen 0.6 --remove-bpe
...
2020-07-04 11:03:03 | INFO | fairseq_cli.generate | Translated 3003 sentences (86379 tokens) in 76.2s (39.39 sentences/s, 1132.93 tokens/s)
2020-07-04 11:03:03 | INFO | fairseq_cli.generate | Generate test with beam=4: BLEU4 = 1.25, 18.2/2.2/0.5/0.1 (BP=1.000, ratio=1.107, syslen=69852, reflen=63078)
```
* sync [results to wandb](https://app.wandb.ai/dertilo/fairseq-nmt/runs/3fm3yx2f?workspace=user-)
```shell script
OMP_NUM_THREADS=4 wandb sync tensorboard_logdir/
```
## others
    
    # Download and prepare the data
    cd examples/translation/
    bash prepare-wmt14en2de.sh
    cd ../..

    fairseq-train \
    data-bin/wmt17_en_de \
    --arch fconv_wmt_en_de \
    -s en -t de \
    --tensorboard-logdir tensorboard_logdir/wmt_en_de \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/fconv_wmt_en_de
    
    fairseq-generate data-bin/wmt17_en_de \
    --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt \
    --beam 5 --remove-bpe
    
    2020-03-09 23:29:55 | INFO | fairseq_cli.generate | Generate test with beam=5: BLEU4 = 18.06, 50.4/23.4/12.6/7.1 (BP=1.000, ratio=1.031, syslen=66478, reflen=64506)


### spanish-english

    TEXT=examples/translation/es_en
    fairseq-preprocess \
    --source-lang en --target-lang es \
    --trainpref $TEXT/train --validpref $TEXT/valid  \
    --destdir data-bin/en_es --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

es-en training

    fairseq-train \
    data-bin/en_es \
    --arch fconv_wmt_en_de \
    -s en -t es \
    --tensorboard-logdir $HOME/data/tensorboard_logdir/en_es \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/en_es

es-en training transformer
    
    fairseq-train \
        data-bin/en_es \
        --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        -s en -t es \
        --tensorboard-logdir $HOME/data/tensorboard_logdir/en_es_transformer \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
        --dropout 0.3 --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 3584 \
        --fp16 \
        --save-dir checkpoints/en_es_transformer

    fairseq-train \
        data-bin/en_es \
        --arch transformer_vaswani_wmt_en_de_big \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        -s en -t es \
        --tensorboard-logdir $HOME/data/tensorboard_logdir/en_es_transformer \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
        --dropout 0.3 --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --max-tokens 3584 \
        --fp16 \
        --save-dir checkpoints/en_es_transformer

### portuguese
    
    bash prepare-ptbr2en.sh

    TEXT=examples/translation/pt_en
    fairseq-preprocess \
    --source-lang pt --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid  \
    --destdir data-bin/pt_en --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

pt-en training

    fairseq-train \
    data-bin/pt_en \
    --arch fconv_wmt_en_de \
    -s pt -t en \
    --tensorboard-logdir tensorboard_logdir/pt_en \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/pt_en

en-pt training

    fairseq-train \
    data-bin/pt_en \
    --arch fconv_wmt_en_de \
    -s en -t pt \
    --tensorboard-logdir tensorboard_logdir/en_pt \
    --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --save-dir checkpoints/en_pt
#### results
![pt-en](images/pt_en.png)
### deployment
    rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" --progress /home/tilo/data/models/MT gunther@gunther:/home/gunther/tilo_data/

used standard ml-container
pip install fairseq==0.9.0

## libraries
* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* [masakhane-mt](https://github.com/masakhane-io/masakhane-mt)
* https://github.com/facebookresearch/UnsupervisedMT
* [fairseq+wandb](https://github.com/Guitaricet/fairseq)