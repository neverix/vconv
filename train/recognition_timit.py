"""
TIMIT phoneme recognition model training script.

If `data/recognition/train.pkl` is the path of the preprocessed dataset and
`data/recognition/model.pt` is the path to save the model to:

 $ python -m train.recognition_timit data/recognition/train.pkl data/recognition/model.pt
"""

import argparse


def main():
    """
    The main function.
    """
    parser = argparse.ArgumentParser(description='Train recognition model.')
    parser.add_argument("source", help="Path to preprocessed TIMIT dataset.")
    parser.add_argument("target", nargs="?", help="Path to the torch file to store the model to.",
                        default="data/recognition/model.pt")
    args = parser.parse_args()

    print(args)


if __name__ == '__main__':
    main()
