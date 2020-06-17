"""
TIMIT preprocessing. This is a runnable file.

If `data/timit/train` is the top-level TIMIT directory (containing DR1, DR2 etc),
and `data/recognition/train.pkl` is the path to store the preprocessed data in:

 $ python -m preprocess.preprocess_timit data/timit/train/ data/recognition/train.pkl
"""

from hparams import timit, Map
from . import preprocess
from pathlib import Path
import pickle
import argparse
import numpy as np


def preprocess_wav_timit(wav_path: Path, hp: Map):
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


def walk_dataset_timit(dataset_path: Path, hp: Map):
    """
    Walk through all .wav files in the TIMIT dataset.
    :param dataset_path: path to TIMIT dataset
    :param hp: hyperparameters object
    :return: a generator of data samples
    """
    for path in dataset_path.rglob("*.WAV"):
        yield preprocess_wav_timit(path, hp)


def main():
    """
    The main function.
    """
    parser = argparse.ArgumentParser(description='Preprocess TIMIT.')
    parser.add_argument("source", help="Path to TIMIT dataset at root level (containing folders like DR1, DR2 etc).")
    parser.add_argument("target", nargs="?", help="Path to the pickle file to store the preprocessed data in.",
                        default="data/recognition/train.pkl")
    args = parser.parse_args()

    data = []
    for i, x in enumerate(walk_dataset_timit(Path(args.source), hp=timit)):
        data.append(x)
        print(f"\rPreprocessed file #{i + 1}.", end='')

    print("\nFinished preprocessing, saving...")
    save_file = Path(args.target)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, save_file.open('wb'))

    print("Done.")


if __name__ == '__main__':
    main()
