"""
TIMIT phoneme recognition model training script.

If `data/recognition/train.pkl` is the path of the preprocessed dataset and
`data/recognition/model.pt` is the path to save the model to:

 $ python -m train.recognition_timit data/recognition/train.pkl data/recognition/model.pt
"""

from hparams import timit, Map
from models.quartznet import QuartzNet
import argparse
import pickle
import math
from pathlib import Path
import sklearn.model_selection
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def prep(train_data, test_data, hp: Map):
    """
    Prepare objects for training.
    :param train_data: preprocessed dataset path
    :param test_data: test dataset path (can be replaced with None)
    :param hp: hyperparameters object
    :return: model, loss, optimizer and training data loader
    """
    torch.manual_seed(hp.seed)
    model = QuartzNet(timit).to(hp.device)
    parameters = sum([p.numel() for p in model.parameters()]) / 1_000_000
    print(f"Model loaded. Parameters: {parameters:.3f}M.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

    train_data = pickle.load(open(train_data, 'rb'))
    if test_data is None:
        train_data, test_data = sklearn.model_selection.train_test_split(train_data, test_size=hp.test_size,
                                                                         random_state=hp.seed)
    else:
        test_data = pickle.load(open(test_data, 'rb'))

    train_dl = DataLoader(train_data,
                          batch_size=hp.batch_size,
                          collate_fn=lambda b: collate_fn(b, hp),
                          shuffle=True)
    test_dl = DataLoader(test_data,
                         batch_size=hp.test_batch_size,
                         collate_fn=lambda b: collate_fn(b, hp),
                         shuffle=True)
    return model, criterion, optimizer, train_dl, test_dl


def collate_fn(batch, hp: Map):
    """
    Collation function for the data loader
    :param batch: batch to collate
    :param hp: hyperparameters object
    :return: input and output tensors
    """
    batch_x, batch_y = [torch.nn.utils.rnn.pad_sequence(
        [torch.from_numpy(x) for x in seq],
        batch_first=True
    ) for seq in zip(*batch)]  # unzip, convert to tensor and pad
    # do the stride downscaling thing
    indices = torch.arange(0, batch_x.shape[1], hp.downscale_ratio)
    assert indices.shape[0] == math.ceil(batch_y.shape[1] / hp.downscale_ratio)
    batch_y = batch_y[:, indices]
    # permute dimensions into channel format
    batch_x = batch_x.permute(0, 2, 1)
    return batch_x, batch_y


def fit(model, criterion, optimizer, train_dl, test_dl, target, hp):
    """
    Fit model.
    :param model: model to train
    :param criterion: objective function
    :param optimizer: model optimizer
    :param train_dl: training data loader
    :param test_dl: test data loader
    :param target: path to save the model to
    :param hp: hyperparameters object
    """
    batch = 0
    epoch = 0

    while batch < hp.n_batches:
        print(f"Epoch: {epoch + 1}; Batch: {batch + 1}/{hp.n_batches}.")

        if test_dl is not None:
            model.eval()
            correct, total = 0, 0
            for x, y in test_dl:
                x = x.to(hp.device)
                y = y.to(hp.device)
                y_ = model(x)
                counts = y_.argmax(dim=1) == y
                correct += counts.float().mean().item()
                total += 1
            if total > 0:
                acc = correct / total
                print(f" Validation accuracy: { acc*100:.5}%")

        for x, y in train_dl:
            model.train()
            x = x.to(hp.device)
            y = y.to(hp.device)
            optimizer.zero_grad()
            y_ = model(x)

            loss = criterion(y_, y)
            loss.backward()
            optimizer.step()
            batch += 1

            acc = (y_.argmax(dim=1) == y).float().mean().item()
            print(f"\r Batch {batch}: Loss: {loss.item():.7}; Acc: { acc*100:.5}%", end='')

        torch.save(model.state_dict(), target)
        print()
        epoch += 1


def main():
    """
    The main function.
    """
    parser = argparse.ArgumentParser(description='Train recognition model.')
    parser.add_argument("source", nargs="?", help="Path to preprocessed TIMIT training dataset.")
    parser.add_argument("target", nargs="?", help="Path to the torch file to store the model to.")
    parser.add_argument("test", nargs="?", help="Path to preprocessed TIMIT testing dataset.", default="")
    args = parser.parse_args()

    model, criterion, optimizer, train_dl, test_dl = prep(args.source, args.test if args.test != "" else None, timit)
    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fit(model, criterion, optimizer, train_dl, test_dl, target, timit)


if __name__ == '__main__':
    main()
