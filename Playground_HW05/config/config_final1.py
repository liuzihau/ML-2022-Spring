from argparse import Namespace

result = "validation loss:	3.1309 BLEU = 27.33 58.8/34.3/21.0/13.4"
'''
This is final translation training
use back translate model zh_en4:
"validation loss:	2.6251 BLEU = 20.22 57.0/28.4/16.1/9.5 "
use config4:
"validation loss:	3.2772 BLEU = 25.24 59.6/34.1/20.3/12.7"
'''


class Config:
    def __init__(self):
        self.args = Namespace(
            seed=73,
            datadir="./DATA/data-bin/ted2020_final",
            savedir="./checkpoints/transformer",
            source_lang="en",
            target_lang="zh",

            # cpu threads when fetching & processing data.
            num_workers=2,
            # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
            max_tokens=8192,
            accum_steps=4,

            # choose model architecture
            model='transformer',
            # the lr s calculated from Noam lr scheduler. you can tune the maximum lr by this factor.
            lr_factor=2.,
            lr_warmup=4000,

            # clipping gradient norm helps alleviate gradient exploding
            clip_norm=1.0,

            # maximum epochs for training
            max_epoch=30,
            start_epoch=1,

            # beam size for beam search
            beam=10,
            # generate sequences of maximum length ax + b, where x is the source length
            max_len_a=1.2,
            max_len_b=10,
            # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
            post_process="sentencepiece",

            # checkpoints
            keep_last_epochs=5,
            resume=None,  # if resume from checkpoint name (under config.savedir)
            average_checkpoints_number=5,

            # logging
            use_wandb=False,
        )
        self.architecture = Namespace(
            encoder_embed_dim=512,
            encoder_ffn_embed_dim=512,
            encoder_layers=3,
            decoder_embed_dim=512,
            decoder_ffn_embed_dim=1024,
            decoder_layers=3,
            # RNN = True , Transformer = False
            share_decoder_input_output_embed=False,
            dropout=0.3,
            # HINT: these patches on parameters for Transformer
            encoder_attention_heads=16,
            encoder_normalize_before=True,

            decoder_attention_heads=16,
            decoder_normalize_before=True,

            activation_fn="relu",
            max_source_positions=1024,
            max_target_positions=1024
        )
