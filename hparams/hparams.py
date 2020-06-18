"""
Concrete definitions of the hyperparameters. If you want to modify something, this is the best place to start.
"""

from .map import Map


hparams = Map(
    mel_window_step=10,
    mel_n_channels=80,
    mel_window_length=20,
    source_sr=None,
    sr=22050,
    min_level_db=-35,
    max_level_db=35,

    model_block_params=("tsconv", "n_sub", "out_channels", "kernel_size",
                        "dropout_prob", "stride", "activation"),
    block_params_default=(False, 1, 256, 22, 0.2, 1, "relu"),
    dilation=1,
    stride=1,
    device="cuda:0",
    seed=97
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
    sr=16000,

    downscale_ratio=3,
    test_size=0.1,
    learning_rate=0.0001,
    batch_size=32,
    test_batch_size=60,
    n_batches=100_000
)
timit = Map(
    timit,
    in_channels=timit.mel_n_channels,
    model_blocks=[
        # first prep-processing layer
        (False, 1, 256, 11, 0.2, 3),
        # five groups of blocks
        (True, 5, 256, 11, 0.2),
        (True, 5, 256, 13, 0.2),
        (True, 5, 512, 17, 0.3),
        (True, 5, 512, 21, 0.3),
        (True, 5, 512, 25, 0.3),
        # post-processing layers
        (False, 1, 512, 29, 0.3),
        (False, 1, 1024, 1, 0.4),
        (False, 1, len(timit.phonemes), 1, 0.0, 1, "softmax")
    ]
)

librispeech = Map(
    hparams,
    n_samples=10
)
