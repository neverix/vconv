"""
Concrete definitions of the hyperparameters. If you want to modify something, this is the best place to start.
"""

from .map import Map


hparams = Map(
    resemblyzer_n_samples=10,
    mel_window_step=10,
    mel_n_channels=80,
    mel_window_length=25,
    source_sr=None,
    sr=22050,
    min_level_db=-35,
    max_level_db=35
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

librispeech = Map(
    hparams,
    n_samples=10
)
