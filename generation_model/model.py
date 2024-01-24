import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 


LATENT_FEATURE = 16
number_constraints = 2
VACTOR_LENGTH = 150
N_CLASS = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)    # standard deviation
    sampled_z = Variable(torch.Tensor(np.random.normal(0, 1, (mu.size(0), LATENT_FEATURE)))).to(device)
    sample = mu + (sampled_z * std)       # sampling as if coming from the input space

    return sample

class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=300 + number_constraints * 2 + N_CLASS, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(64, LATENT_FEATURE)
        self.logvar = nn.Linear(64, LATENT_FEATURE)

    def forward(self, x):
        # print(f"==========> {x.shape}")
        x = self.model(x)
        # print(f"----------> {x.shape}")
        # # exit(0)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterize(mu, logvar)
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(LATENT_FEATURE, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 150 + number_constraints),
        )

    def forward(self, z):
        print(z.shape)
        x = self.model(z)

        return x


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(VACTOR_LENGTH + number_constraints, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.real_fake_fc = nn.Linear(128, 1)
        self.class_fc = nn.Linear(128, N_CLASS)
    
    def forward(self, z):
        feat = self.model(z)
        validity = self.real_fake_fc(feat)
        cls = self.class_fc(feat)

        return validity, cls

