import argparse
import os

from distillation import BartTranslationDistiller, distill_main

if __name__ == "__main__":
    debug_args = """
--data_dir=some_data \
--src_lang=en_XX \
--tgt_lang=ro_RO \
--model_name_or_path IGNORED \
--learning_rate=3e-4 \
--train_batch_size=4 \
--eval_batch_size=4 \
--teacher Helsinki-NLP/opus-mt-en-ro \
--tokenizer_name Helsinki-NLP/opus-mt-en-ro \
--warmup_steps 500 \
--student_decoder_layers 2 --student_encoder_layers 2 \
--freeze_embeds \
--alpha_hid=3. --length_penalty=0.5 \
--gradient_accumulation_steps=2 \
--max_target_length=60 --val_max_target_length=60 --test_max_target_length=100 \
--output_dir=debug \
--num_train_epochs 3 \
--gpus 0 \
--do_train \
--do_predict \
--val_check_interval 0.2 \
--sortish_sampler \
    """.strip().split()
    # --fp16 \
    parser = argparse.ArgumentParser()
    parser = BartTranslationDistiller.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args(debug_args)

    distill_main(args)