"""
Concrete definitions of the hyperparameters. If you want to modify something, this is the best place to start.
"""

from .map import Map


hparams = Map(
    mel_window_step=10,
    mel_n_channels=80,
    mel_window_length=25,
    source_sr=None,
    sr=22050,
    min_level_db=-35,
    max_level_db=35,

    model_block_params=("n_sub", "out_channels", "kernel_size", "dropout_prob", "dilation", "activation"),
    block_params_default=(1, 256, 22, 0.2, 1, 1, "relu"),
    learning_rate=0.0001,
    batch_size=32,
    n_epochs=400000
)

timit = Map(
    hparams,
    phonemes=['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
              'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
              'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
              'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
              'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'],
    mel_n_channels=40,
    source_sr=16000,
    sr=16000
)
timit = Map(
    timit,
    model_blocks=[
        (1, 256, 11, 0.2, 2),
        (5, 256, 11, 0.2),
        (5, 384, 13, 0.2),
        (5, 512, 17, 0.2),
        (5, 640, 21, 0.3),
        (5, 768, 25, 0.3),
        (1, 896, 29, 0.4, 2),
        (1, 1024, 1, 0.4),
        (1, len(timit.phonemes), 1, 0, 1, "softmax")
    ],
    downscale_factor=2,
    in_channels=timit.mel_n_channels
)

librispeech = Map(
    hparams,
    n_samples=10
)
