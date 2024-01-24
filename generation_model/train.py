from email.policy import default
from random import shuffle
from symbol import parameters
from matplotlib.style import available
import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib, os
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from csv import writer
from model import *
import itertools


feat_dim = 150
number_constraints = 2

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int, help='number of epochs to train the VAE for')
args = vars(parser.parse_args())

epochs = args['epochs']
lr = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder().to(device)
decoder = Decoder().to(device)
discriminator = Discriminator().to(device)

optimizer_g = optim.Adam(itertools.chain(encoder.parameters(),decoder.parameters()), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)
gan_loss_fn = torch.nn.BCEWithLogitsLoss()
info_loss_fn = torch.nn.CrossEntropyLoss()

target = "6"

def adversarial_loss(dis_results, validity, class_label=None):
    pred_valid, cls = dis_results
    adv_loss = gan_loss_fn(pred_valid,validity)
    if class_label is not None:
        class_loss =  info_loss_fn(cls, class_label)
        return adv_loss, class_loss
    return adv_loss

def fit(dataloader, epoch):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        data = data.to(device)
        if data.size()[0] != 16:
            continue
        # print(f"here65: {data.shape}")
        data_source = data[:, :300+number_constraints*2]
        data_source = torch.nn.functional.normalize(data_source)
        data_target = data[:, 300+number_constraints*2: 300+number_constraints*2+152]
        data_target = torch.nn.functional.normalize(data_target)
        data_label = data[:, 300+number_constraints*2+152:]

        data_label = torch.from_numpy(data_label.squeeze(-1).cpu().detach().numpy().astype(np.int64)).to(device)

        data_source = torch.cat([F.one_hot(data_label, 9), data_source], dim=-1)

        data_source = data_source.reshape(16, 300 + number_constraints*2 + 9)
        data_source = data_source.to(device)
        data_target = data_target.to(device)

        # soft label
        valid = Variable(torch.Tensor(data_target.size()[0], 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(data_target.size()[0], 1).fill_(0.0), requires_grad=False).to(device)

        torch.autograd.set_detect_anomaly(True)
        optimizer_g.zero_grad()
        encoded_data = encoder(data_source)
        decoded_data = decoder(encoded_data)
        # class loss
        _, g_class = adversarial_loss(discriminator(decoded_data), valid, data_label)
        g_class.backward()

        # generator loss
        g_loss = 0.2*adversarial_loss(discriminator(decoded_data), valid) \
            + 0.8*torch.nn.functional.mse_loss(decoded_data, data_target)
        g_loss.backward()
        optimizer_g.step()

        optimizer_d.zero_grad()
        # real-fake loss
        fake_loss = adversarial_loss(discriminator(decoded_data.detach()), fake)
        real_loss, cls_loss = adversarial_loss(discriminator(data_target), valid, data_label)
        d_loss = 0.5*(real_loss + fake_loss) + cls_loss

        running_loss += d_loss.item()
        d_loss.backward(retain_graph=True)
        optimizer_d.step()

        if epoch == 49:
            torch.save(encoder, f"../pixel4-model/VAEGAN-encoder-g{target}")
            torch.save(decoder, f"../pixel4-model/VAEGAN-decoder-g{target}")
    
    train_loss = running_loss/len(dataloader.dataset)
    
    return train_loss


train_loss = []
val_loss = []


gesture_source_1 = ""
gesture_source_2 = ""
gesture_traget = f"/gesture{target}.csv"

dataset_gesture_source_1 = pd.read_csv(gesture_source_1, header=None)
dataset_gesture_source_2 = pd.read_csv(gesture_source_2, header=None)
dataset_gesture_target = pd.read_csv(gesture_traget, header=None)

dataset_gesture_source = pd.concat(
    [dataset_gesture_source_1.iloc[:, :feat_dim+number_constraints], 
    dataset_gesture_source_2.iloc[:, :feat_dim+number_constraints]], 
    axis=1
)

dataset_gesture_source = dataset_gesture_source.iloc[:, :feat_dim*2 + number_constraints * 2]
dataset_gesture_target = dataset_gesture_target.iloc[:, :feat_dim + number_constraints + 1]

dataset_gesture_source.dropna(axis=0, inplace=True)
dataset_gesture_target.dropna(axis=0, inplace=True)

print(f"shape of input: {dataset_gesture_source.shape}")
print(f"shape of output: {dataset_gesture_target.shape}")

dataset = pd.concat([dataset_gesture_source, dataset_gesture_target], axis=1)
print(f"shape of final dataset input: {dataset.shape}")

dataset = dataset.to_numpy()
dataset = torch.from_numpy(dataset)
# print(torch.argwhere(dataset.isnan()))
dataset = dataset.float()

# dataset = torch.nn.functional.normalize(dataset)

train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16)

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(train_dataloader, epoch)
    # val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    # val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    # print(f"Val Loss: {val_epoch_loss:.4f}")    
