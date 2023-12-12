import torch
import torch.nn as nn

from torch.utils.data import DataLoader

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(52 * 768, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 256),  # Starting from a 100-dim latent space
            nn.ReLU(),
            nn.Linear(256, 52 * 768),  # Adjust the output size to match the flattened shape
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        return x.view(-1, 52, 768)  # Reshape to the required output shape

