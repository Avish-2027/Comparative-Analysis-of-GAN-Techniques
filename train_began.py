import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


# Settings

image_size = 64
batch_size = 64
nz = 64
num_epochs = 5
lr = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_root = r"./data/UTKFace-master/faces/all"
os.makedirs("samples_began", exist_ok=True)


# Dataset

class FaceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = glob.glob(os.path.join(root, "*.jpg"))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.files)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])

dataset = FaceDataset(data_root, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nz, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.net(z)
        return x.view(-1, 3, image_size, image_size)


# Autoencoder Discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size * image_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out.view(-1, 3, image_size, image_size)

# Init

netG = Generator().to(device)
netD = Discriminator().to(device)

opt_G = optim.Adam(netG.parameters(), lr=lr)
opt_D = optim.Adam(netD.parameters(), lr=lr)

# BEGAN parameters
k = 0.0
lambda_k = 0.001
gamma = 0.5

# Training

for epoch in range(num_epochs):
    for i, real in enumerate(dataloader):

        real = real.to(device)
        batch_size = real.size(0)

        # Train Generator
        
        z = torch.randn(batch_size, nz).to(device)
        fake = netG(z)

        fake_recon = netD(fake)
        lossG = torch.mean(torch.abs(fake - fake_recon))

        opt_G.zero_grad()
        lossG.backward()
        opt_G.step()

        # Train Discriminator
        
        real_recon = netD(real)
        fake_recon = netD(fake.detach())

        loss_real = torch.mean(torch.abs(real - real_recon))
        loss_fake = torch.mean(torch.abs(fake.detach() - fake_recon))

        lossD = loss_real - k * loss_fake

        opt_D.zero_grad()
        lossD.backward()
        opt_D.step()

     
        # Update k
      
        k = k + lambda_k * (gamma * loss_real.item() - loss_fake.item())
        k = min(max(k, 0), 1)

        if i % 50 == 0:
            print(f"[{epoch+1}/{num_epochs}] LossD: {lossD.item():.4f} LossG: {lossG.item():.4f}")

    save_image(fake[:25], f"samples_began/epoch_{epoch+1}.png", nrow=5, normalize=True)

print("BEGAN Training Done!")