import resemblyzer
from hparams import *
import random
from pathlib import Path
import librosa
import numpy as np


def preprocess_wav(path, sr=None):
    x, sr = librosa.load(path, sr=sr)
    if sr != sampling_rate:
        x, sr = librosa.core.resample(x, sr, sampling_rate), sampling_rate
    mel = librosa.feature.melspectrogram(
        x,
        sampling_rate,
        n_mels=mel_n_channels,
        n_fft=int(mel_window_length * sampling_rate / 1000),
        hop_length=int(mel_window_step * sampling_rate / 1000),
    ).T
    mel = np.clip((librosa.power_to_db(mel) + min_level_db) / (ref_level_db + min_level_db), 0, 1)
    return mel.astype(np.float32)


def preprocess_phn_timit(path: Path, mel):
    phones = np.zeros((mel.shape[0], len(phonemes)), dtype=np.float32)
    for line in path.open():
        start, end, phn = line.split()
        start = int(start) // mel_window_step // (sampling_rate // 1000)
        end = int(end) // mel_window_step // (sampling_rate // 1000)
        idx = phonemes.index(phn)
        phones[start:end, idx] = 1
    return phones


def embed_speaker(speaker_path: Path, voice_encoder):
    wavs = speaker_path.rglob("*.flac")
    wavs = random.sample(list(wavs), resemblyzer_n_samples)
    wavs = [resemblyzer.preprocess_wav(wav) for wav in wavs]
    return voice_encoder.embed_speaker(wavs)
