import preprocess
from pathlib import Path
import sys
import pickle


def preprocess_recognition_timit(dataset_path: Path):
    xs = []
    ys = []
    wavs = list(dataset_path.rglob("*.WAV"))
    for i, wav in enumerate(wavs):
        print(f"\rPreprocessing file {i+1}/{len(wavs)}...", end='')
        phn = wav.with_suffix(".PHN")
        mel = preprocess.preprocess_wav(wav, sr=16000)
        xs.append(mel)
        phn = preprocess.preprocess_phn_timit(phn, mel)
        ys.append(phn)
    return xs, ys


def main():
    data = preprocess_recognition_timit(Path(sys.argv[1]))
    print("\nFinished preprocessing, saving...")
    save_file = Path(sys.argv[2])
    save_file.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, save_file.open('wb'))
    print("Done.")


if __name__ == '__main__':
    main()
