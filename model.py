import argparse
import numpy as np
import os
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

DATA_POINTS = 10000  # 100k ~ 2.5 secs


class MusicNet(nn.Module):
    def __init__(self):
        super(MusicNet, self).__init__()
        self.linn1 = nn.Linear(DATA_POINTS, 5000)
        self.linn11 = nn.Linear(5000, 5000)
        self.linn2 = nn.Linear(5000, 1000)
        self.linn22 = nn.Linear(1000, 1000)
        self.linn3 = nn.Linear(1000, 100)
        self.linn4 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.linn1(x)
        x = F.relu(x)
        x = self.linn11(x)
        x = F.relu(x)
        x = self.linn2(x)
        x = F.relu(x)
        x = self.linn22(x)
        x = F.relu(x)
        x = self.linn3(x)
        x = F.relu(x)
        x = self.linn4(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class Dataset(torch.utils.data.Dataset):

    def __init__(self, list_IDs, labels, subfolder):
        self.labels = labels
        self.list_IDs = list_IDs
        self.subfolder = subfolder

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        samplerate, X = wavfile.read(os.getcwd() + '\\data\\' + self.subfolder + str(ID) + '.wav')
        y = self.labels[ID]
        X = np.array(X[100000:])
        if len(X.shape) > 1:
            X = [sum(i) for i in X]
        X = np.trim_zeros(X, 'f')

        return torch.as_tensor(torch.from_numpy(np.array(X[:DATA_POINTS])), dtype=torch.float), y


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("Running on ", device)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    labels_train = {1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 1,
                    10: 0,
                    11: 0,
                    12: 0,
                    13: 0,
                    14: 0,
                    15: 0,
                    16: 0,
                    17: 0}
    labels_test = {1: 1,
                   2: 1,
                   3: 0,
                   4: 0
                   }
    ids_train = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
    ids_test = [1, 2, 3, 4]
    params = {'batch_size': 2}

    dataset_train = Dataset(ids_train, labels_train, 'TRAIN\\')
    dataset_test = Dataset(ids_test, labels_test, 'TEST\\')

    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = MusicNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "music_ai_bot.pt")
