"""
LibriSpeech preprocessing. This is a runnable file.

If `data/librispeech/train` is the top-level LibriSpeech directory (containing 19, 26 etc),
and `data/synthesis/train.pkl` is the path to store the preprocessed data in:

 $ python -m preprocess.preprocess_librispeech data/librispeech/train/ data/synthesis/train.pkl
"""


from hparams import librispeech, Map
from . import preprocess
from pathlib import Path
import pickle
import argparse
import random
import resemblyzer
# import numpy as np


voice_encoder = resemblyzer.VoiceEncoder("cuda")


def preprocess_flac_librispeech(flac_path: Path, embedding, hp: Map):
    """
    Open and preprocess a flac file.
    :param flac_path: path to the flac file
    :param embedding: speaker embedding
    :param hp: hyperparameters object
    :return: preprocessed mel spectrogram with prosodic information baked in
    """
    mel = preprocess.preprocess_wav(flac_path, hp)
    # TODO add prosodic and phonetic info
    return None, mel


def walk_dataset_librispeech(dataset_path: Path, hp: Map):
    """
    Walk through all .flac files in the LibriSpeech dataset.
    :param hp: hyperparameters object
    :param dataset_path: path to LibriSpeech dataset
    :return: a generator of data samples
    """
    for speaker in dataset_path.iterdir():
        embedding = None  # embedding = embed_speaker_librispeech(speaker, hp)
        for flac in speaker.rglob("*.flac"):
            yield preprocess_flac_librispeech(flac, embedding, hp)


def embed_speaker_librispeech(speaker_path: Path, hp: Map):
    """
    Create an embedding of a speaker directory using `resemblyzer`.
    :param speaker_path: path to speaker directory
    :param hp: hyperparameters object
    :return: speaker embedding
    """
    flacs = list(speaker_path.rglob("*.flac"))
    flacs = random.sample(flacs, hp.n_samples)
    flacs = [resemblyzer.preprocess_wav(flac) for flac in flacs]
    return voice_encoder.embed_speaker(flacs)


def main():
    """
    The main function.
    """
    parser = argparse.ArgumentParser(description='Preprocess LibriSpeech.')
    parser.add_argument("source", help="Path to LibriSpeech dataset at root level "
                                       "(containing folders like 19, 26 etc).")
    parser.add_argument("target", nargs="?", help="Path to the pickle file to store the preprocessed data in.",
                        default="data/synthesis/train.pkl")
    args = parser.parse_args()

    data = []
    for i, x in enumerate(walk_dataset_librispeech(Path(args.source), hp=librispeech)):
        data.append(x)
        print(f"\rPreprocessed file {i}.", end='')

    print("\nFinished preprocessing, saving...")
    save_file = Path(args.target)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, save_file.open('wb'))

    print("Done.")


if __name__ == '__main__':
    main()
