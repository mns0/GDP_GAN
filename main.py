import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision
import datetime
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

from torch.nn import init

import torch.nn as nn
import torch.nn.functional as F


# Create a dataloader
class GDPData(Dataset):
    """GDP time series """

    def __init__(self, csv_file, series_length=15, normalize=False):
        df = pd.read_csv(csv_file)
        df = df.drop(['Unnamed: 0'], axis=1)
        if normalize:
            self.normalize_table = np.zeros((df.shape[1], 2))  # (min, max)
            self.normalize(df)
            #save_table
            np.savetxt(csv_file+"normalization_table.txt",self.normalize_table)

        df.drop(df[(df.shape[0] // series_length *
                    series_length):].index, inplace=True)
        x = df.loc[:, (df.columns != 'GDP')]

        full = np.array(
            np.array_split(
                df.to_numpy(),
                df.shape[0] //
                series_length))
        x = np.array(np.array_split(x.to_numpy(), x.shape[0] // series_length))
        y = df["GDP"]
        y = np.array(np.array_split(y.to_numpy(), y.shape[0] // series_length))
        self.t = x.shape[0]
        #print("d", type(x),x.shape.shape)
        x = np.expand_dims(x, -1)
        full = np.expand_dims(full, -1)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.full = torch.from_numpy(full).float()

    def normalize(self, x):
        """
          Normalize data to [-1,1] range
          for all predictors in dataframe
        """
        if not hasattr(self, 'normalize_table'):
            raise Exception("Incorrectly calling normalize")
        for idx, c in enumerate(x.columns):
            l, m = x[c].min(), x[c].max()
            self.normalize_table[idx][0] = l
            self.normalize_table[idx][1] = m
            x[c] = 2 * ((x[c] - l) / (m - l)) - 1

    def unnormalize(self, x):
        """
          Unnormalize data from [-1,1] range
          for all predictors in dataframe
        """
        if not hasattr(self, 'normalize_table'):
            raise Exception("Incorrectly calling normalize")
        for idx, c in enumerate(x.columns):
            _min = self.normalize_table[idx][0]
            _max = self.normalize_table[idx][1]
            x[c] = 0.5* (x*_max - x*_min + _max + _min)
        return x

    def __len__(self):
        return self.t

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        xr = self.x[idx]
        yr = self.y[idx]
        # the conditional
        full = self.full[idx]
        return xr, yr, full


class StatefulLSTM(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM, self).__init__()
        self.lstm = nn.LSTMCell(in_size, out_size)
        self.out_size = out_size
        self.h, self.c = None, None

    def reset_state(self):
        self.h, self.c = None, None

    def forward(self, x):
        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = torch.zeros(state_size)
            self.h = torch.zeros(state_size)
        self.h, self.c = self.lstm(x, (self.h, self.c))
        return self.h


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if not train:
            return x

        if (self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)
        return mask * x


class Generator_RNN(nn.Module):
    def __init__(
            self,
            predictor_dim,
            seq_len=15,
            target_size=1,
            hidden_dim=1,
            num_layers=1,
            dropout_prob=0,
            train=True):
        super(Generator_RNN, self).__init__()
        self.train = train
        self.dropout_prob = dropout_prob

        self.lstm1 = StatefulLSTM(predictor_dim, hidden_dim)
        self.bn_lstm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = LockedDropout()

        self.lstm2 = StatefulLSTM(hidden_dim, hidden_dim)
        self.bn_lstm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = LockedDropout()
        self.fc_output = nn.Sequential(
            nn.Linear(hidden_dim, target_size), nn.Tanh())

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()

    def forward(self, inputx, inputc):
        self.reset_state()
        # input - batch_size x time_steps x features
        input = torch.cat((inputx, inputc), 2)
        batch_size, no_of_timesteps, features = input.size(
            0), input.size(1), input.size(2)
        outputs = []
        #print("In generator input size", input.shape)

        # lstm on each sequence
        for i in range(batch_size):
            h = self.lstm1(input[i, :, :])
            h = self.bn_lstm1(h)
            h = self.dropout1(h)

            h = self.lstm2(h)
            h = self.bn_lstm2(h)
            h = self.dropout2(h)
            h = self.fc_output(h)
            h = torch.squeeze(h)
            outputs.append(h)

        outputs = torch.stack(outputs)  # batch_size, timesteps
        return outputs


class Discriminator_RNN(nn.Module):
    def __init__(
            self,
            predictor_dim,
            target_size=1,
            hidden_dim=200,
            num_layers=1,
            dropout_prob=0,
            train=True):
        super(Discriminator_RNN, self).__init__()
        self.train = train
        self.dropout_prob = dropout_prob
        self.lstm1 = StatefulLSTM(predictor_dim, hidden_dim)
        self.bn_lstm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = LockedDropout()
        self.fc_output = nn.Sequential(
            nn.Linear(
                hidden_dim,
                target_size),
            nn.Sigmoid())

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()

    def forward(self, _inputx, _inputy):
        self.reset_state()
        # input - batch_size x time_steps x features
        _input = torch.cat((_inputx, _inputy), 2)
        no_of_timesteps = _input.shape[1]
        outputs = []
        for i in range(no_of_timesteps):
            h = self.lstm1(_input[:, i, :])
            h = self.bn_lstm1(h)
            h = self.dropout1(h, dropout=self.dropout_prob, train=self.train)
            outputs.append(h)

        outputs = torch.stack(outputs)  # time_steps, batch_size, features
        outputs = outputs.permute(1, 2, 0)  # time_steps, features, batch_size

        pool = nn.MaxPool1d(no_of_timesteps)

        h = pool(outputs)
        h = h.view(h.size(0), -1)
        h = self.fc_output(h)
        return h


def d_train_step(
        batch_size,
        seq_len,
        x,
        conditional,
        real_label,
        fake_label,
        netD,
        netG,
        optimizerD,
        criterion,
        device):

    netD.zero_grad()
    # train with real batch
    real_D = netD(x, conditional).view(-1)
    rlabel = torch.full((batch_size, ), real_label, device=device)
    errD_real = criterion(real_D, rlabel)

    # training with the fake batch
    # G(Z|Y) normal noise + condition
    noise = 2*torch.randn(batch_size, seq_len, 1, device=device)-1
    fake_data = netG(noise, conditional)
    flabel = torch.full((batch_size, ), fake_label, device=device)
    fake_data = torch.unsqueeze(fake_data, 2)
    output = netD(fake_data.detach(), conditional.detach()).view(-1)
    errD_fake = criterion(output, flabel)

    errD = errD_real + errD_fake
    errD.backward()
    optimizerD.step()
    return errD_real.mean().item(), errD_fake.mean().item(), errD.mean().item()


def g_train_step(
        batch_size,
        seq_len,
        netD,
        netG,
        criterion,
        real_label,
        conditional,
        optimizerG,
        device):

    netG.zero_grad()
    label = torch.full((batch_size, ), real_label, device=device)
    noise = 2*torch.randn(batch_size, seq_len, 1, device=device)-1
    fake_data = netG(noise, conditional).unsqueeze(2)
    output = netD(fake_data.detach(), conditional.detach()).view(-1)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()
    return errG.mean().item(), D_G_z2


def main():
    epoches, batch_size = 100, 20
    real_label, fake_label = 1, 0
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get the data
    dir = './'
    train_data = 'Preprocessed_data_gdp_pc.dat'

    gdp_dataset = GDPData(dir + train_data, normalize=True)
    dataloader = torch.utils.data.DataLoader(
        gdp_dataset, batch_size=batch_size, shuffle=False)

    kwagsD = {"target_size": 1, "predictor_dim": 10, "hidden_dim": 200,
              "num_layers": 1, "dropout_prob": 0, 'train': True}

    kwagsG = {"target_size": 1, "predictor_dim": 10, "hidden_dim": 200,
              "num_layers": 1, "dropout_prob": 0, 'train': True}

    netD = Discriminator_RNN(**kwagsD).to(device)
    netG = Generator_RNN(**kwagsG).to(device)

    criterion = nn.BCELoss().to(device)
    optimizerD = optim.Adam(netD.parameters(), lr=0.001,
                            betas=(0.9, 0.999), eps=1e-08,
                            weight_decay=0, amsgrad=True)

    optimizerG = optim.Adam(netG.parameters(), lr=0.001,
                            betas=(0.9, 0.999), eps=1e-08,
                            weight_decay=0, amsgrad=True)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    #noise = torch.randn(batch_size, seq_len, 1, device=device)
    #fake_data = netG(noise, conditional).unsqueeze(2)
    # Establish convention for real and fake labels during training
    arrloss = np.zeros((num_epochs,2))
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        dloss, gloss = 0,0
        for i, data in enumerate(dataloader):
            conditional, x, _ = data
            if conditional.size(0) < 20:
                continue

            conditional, x = conditional.to(device), x.to(device)
            batch_size, seq_len = x.size(0), x.size(1)
            x = x.unsqueeze(2)
            conditional = conditional.squeeze(3)

            # Update D network: maximize log(D(x|y)) + log(1 - D(G(z|y)))
            D_x, D_G_z1, errD = d_train_step(
                batch_size,
                seq_len,
                x,
                conditional,
                real_label,
                fake_label,
                netD,
                netG,
                optimizerD,
                criterion,
                device)

            # Update G network: maximize log(D(G(z|y)))
            errG, D_G_z2 = g_train_step(
                batch_size,
                seq_len,
                netD,
                netG,
                criterion,
                real_label,
                conditional,
                optimizerG,
                device)
            dloss, gloss = errD + dloss, errG + gloss
            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' %
                    (epoch, num_epochs, i, len(dataloader), errD, errG, D_x, D_G_z1, D_G_z2))

        
        arrloss[i] = [dloss,gloss]
        if epoch % 5 == 0:
            torch.save(netD, f'./models/netD_e{epoch}.pth')
            torch.save(netG, f'./models/netG_e{epoch}.pth')



    np.savetxt("loss.txt",arrloss)

if __name__ == '__main__':
    main()
