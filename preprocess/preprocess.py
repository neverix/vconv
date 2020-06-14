from hparams import hparams
from pathlib import Path
import librosa
import numpy as np


def preprocess_wav(path: Path, hp=hparams):
    """
    Reads audio file and converts it into a spectrogram.
    :param path: path to the audio file
    :param hp: hyperparameters object
    :return: resulting mel spectrogram
    """

    x, sr = librosa.load(path, sr=hp.source_sr)
    if (hp.sr is not None) and sr != hp.sr:
        x, sr = librosa.core.resample(x, sr, hp.sr), hp.sr
    mel = librosa.feature.melspectrogram(
        x, hp.sr,
        n_mels=hp.mel_n_channels,
        n_fft=int(hp.mel_window_length * hp.sr / 1000),
        hop_length=int(hp.mel_window_step * hp.sr / 1000),
    ).T
    mel = np.clip((librosa.power_to_db(mel) + hp.min_level_db) / (hp.max_level_db + hp.min_level_db), 0, 1)
    return mel.astype(np.float32)
