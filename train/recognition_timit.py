"""
TIMIT phoneme recognition model training script.

If `data/recognition/train.pkl` is the path of the preprocessed dataset and
`data/recognition/model.pt` is the path to save the model to:

 $ python -m train.recognition_timit data/recognition/train.pkl data/recognition/model.pt
"""

from hparams import timit, Map
from models.jasper import JasperModel
import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


torch.manual_seed(99)
device = "cuda"


def prep(train_data, hp: Map):
    """
    Prepare objects for training.
    :param train_data: preprocessed dataset path
    :param hp: hyperparameters object
    :return: model, loss, optimizer and training data loader
    """
    model = JasperModel(timit).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    train_data = pickle.load(open(train_data, 'rb'))
    train_dl = DataLoader(train_data,
                          batch_size=hp.batch_size,
                          collate_fn=collate_fn,
                          shuffle=True)
    return model, criterion, optimizer, train_dl


def fit(model, criterion, optimizer, train_dl, hp):
    """
    Fit model.
    :param model: model to train
    :param criterion: objective function
    :param optimizer: model optimizer
    :param train_dl: training data loader
    :param hp: hyperparameters object
    """
    for epoch in range(hp.n_epochs):
        print(f"Epoch {epoch + 1}/{hp.n_epochs}.")
        model.train()
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_ = model(x)

            loss = criterion(y, y_)
            loss.backward()
            optimizer.step()


def collate_fn(batch):
    """
    Collation function for the data loader
    :param batch: batch to collate
    :return: input and output tensors
    """
    batch_x, batch_y = [[torch.from_numpy(x) for x in seq]
                        for seq in zip(*batch)]  # unzip and convert to tensor
    batch_x = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first=True)

    max_len = max([y.shape[0] for y in batch_y])
    for i, y in enumerate(batch_y):
        pad_length = max_len - y.shape[0]
        # make the padded sequence activate the h# phoneme that represents silence and is coincidentally the first one
        first_onehot = torch.zeros((pad_length, y.shape[1]))
        first_onehot[:, 0] = 1
        y = torch.cat((y, first_onehot))
        batch_y[i] = y
    batch_y = torch.stack(batch_y)

    batch_x = batch_x.permute(0, 2, 1)
    batch_y = batch_y.permute(0, 2, 1)

    return batch_x, batch_y


def main():
    """
    The main function.
    """
    parser = argparse.ArgumentParser(description='Train recognition model.')
    parser.add_argument("source", help="Path to preprocessed TIMIT dataset.")
    parser.add_argument("target", nargs="?", help="Path to the torch file to store the model to.",
                        default="data/recognition/model.pt")
    args = parser.parse_args()

    model, criterion, optimizer, train_dl = prep(args.source, timit)
    fit(model, criterion, optimizer, train_dl, timit)


if __name__ == '__main__':
    main()
