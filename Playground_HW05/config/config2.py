from argparse import Namespace

result = "validation loss:	3.4671 BLEU = 23.24 58.4/32.5/19.0/11.6"
'''
accum_steps=2-->4
beam=5-->10
encoder_embed_dim=256-->512
encoder_layers=1-->2
decoder_embed_dim=256-->512
decoder_layers=1-->2
'''


class Config:
    def __init__(self):
        self.args = Namespace(
            seed=73,
            datadir="./DATA/data-bin/ted2020",
            savedir="./checkpoints/transformer",
            source_lang="en",
            target_lang="zh",

            # cpu threads when fetching & processing data.
            num_workers=2,
            # batch size in terms of tokens. gradient accumulation increases the effective batchsize.
            max_tokens=8192,
            accum_steps=4,

            # choose modelarchitecture
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
            encoder_layers=2,
            decoder_embed_dim=512,
            decoder_ffn_embed_dim=1024,
            decoder_layers=2,
            # RNN = True , Transformer = False
            share_decoder_input_output_embed=False,
            dropout=0.3,

            # HINT: these patches on parameters for Transformer
            encoder_attention_heads=4,
            encoder_normalize_before=True,

            decoder_attention_heads=4,
            decoder_normalize_before=True,

            activation_fn="relu",
            max_source_positions=1024,
            max_target_positions=1024
        )
