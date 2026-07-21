import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image


# Settings

image_size = 64
batch_size = 64
nz = 100      # latent vector size
ngf = 64
ndf = 64
nc = 3        # RGB
num_epochs = 5
lr = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_root = r"./data/UTKFace-master/faces"
os.makedirs("samples", exist_ok=True)

# Dataset

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(
    root=data_root,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=0)


# Generator

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# Discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# Init

netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Training loop

for epoch in range(num_epochs):
    for i, (real, _) in enumerate(dataloader):

        real = real.to(device)
        b_size = real.size(0)

     
        # Train Discriminator
        
        netD.zero_grad()

        label_real = torch.ones(b_size, device=device)
        output_real = netD(real)
        lossD_real = criterion(output_real, label_real)

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)

        label_fake = torch.zeros(b_size, device=device)
        output_fake = netD(fake.detach())
        lossD_fake = criterion(output_fake, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

       
        # Train Generator
       
        netG.zero_grad()

        label_gen = torch.ones(b_size, device=device)
        output = netD(fake)
        lossG = criterion(output, label_gen)

        lossG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"[{epoch+1}/{num_epochs}] [{i}/{len(dataloader)}] "
                  f"LossD: {lossD.item():.4f}  LossG: {lossG.item():.4f}")

    with torch.no_grad():
        fake_samples = netG(fixed_noise).detach().cpu()
        save_image(fake_samples, f"samples/epoch_{epoch+1}.png", normalize=True)

print("Training finished.")