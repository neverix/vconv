"""
TIMIT preprocessing. This is a runnable file.

If `data/timit/train` is the top-level TIMIT directory (containing DR1, DR2 etc),
and `data/recognition/train.pkl` is the path to store the preprocessed data in:

 $ python -m preprocess.preprocess_timit data/timit/train/ data/recognition/train.pkl
"""


from hparams import timit
from . import preprocess
from pathlib import Path
import pickle
import argparse
import numpy as np


def preprocess_wav_timit(wav_path: Path, hp=timit):
    """
    Open and preprocess a wav/phn pair.
    :param wav_path: path to the wav file
    :param hp: hyperparameters object
    :return: preprocessed mel spectrogram/phoneme tuple
    """
    mel = preprocess.preprocess_wav(wav_path, hp)
    phn_path = wav_path.with_suffix(".PHN")
    phn = np.zeros((mel.shape[0], len(hp.phonemes)), dtype=np.float32)
    for line in phn_path.open():
        start, end, phn_path = line.split()
        start = int(start) // hp.mel_window_step // (hp.sr // 1000)
        end = int(end) // hp.mel_window_step // (hp.sr // 1000)
        idx = hp.phonemes.index(phn_path)
        phn[start:end, idx] = 1
    return mel, phn


def main():
    """
    The main function.
    """
    parser = argparse.ArgumentParser(description='Preprocess TIMIT.')
    parser.add_argument("source", help="Path to TIMIT dataset at root level (containing folders like DR1, DR2 etc).")
    parser.add_argument("target", help="Path to the pickle file to store the preprocessed data in.")
    args = parser.parse_args()

    data = []
    wavs = list(Path(args.source).rglob("*.WAV"))
    for i, path in enumerate(wavs):
        print(f"\rPreprocessing file {i + 1}/{len(wavs)}...", end='')
        data.append(preprocess_wav_timit(path))

    print("\nFinished preprocessing, saving...")
    save_file = Path(args.target)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, save_file.open('wb'))

    print("Done.")


if __name__ == '__main__':
    main()