# machine-translation

# datasets
* [movie-subtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php)
* [europal](https://www.statmt.org/europarl/)

# [fairseq-mt](https://github.com/pytorch/fairseq/tree/master/examples/translation)
[also see](https://modelzoo.co/model/fairseq-py)
## setup
```shell script
conda create -n fairseq python=3.7 -y
conda activate fairseq
cd fairseq && OMP_NUM_THREADS=4 pip install -e .
pip install sacremoses
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
```
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```
* on __node__ train it

```shell script
fairseq-train \
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
