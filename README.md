# machine-translation

# datasets
* [movie-subtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php)
* [europal](https://www.statmt.org/europarl/)

# [fairseq-mt](https://github.com/pytorch/fairseq/tree/master/examples/translation)
[also see](https://modelzoo.co/model/fairseq-py)

    
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
