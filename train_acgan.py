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
nz = 100
num_classes = 2   # gender: 0,1
num_epochs = 5
lr = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_root = r"./data/UTKFace-master/faces/all"
os.makedirs("samples_acgan", exist_ok=True)

# Dataset (UTKFace with gender)

class UTKDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = glob.glob(os.path.join(root, "*.jpg"))
        self.transform = transform

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")

        name = os.path.basename(path)
        gender = int(name.split("_")[1])  # 0 or 1

        if self.transform:
            img = self.transform(img)

        return img, gender

    def __len__(self):
        return len(self.files)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = UTKDataset(data_root, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, nz)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = noise + label_input
        x = x.view(x.size(0), nz, 1, 1)
        return self.model(x)


# Discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.adv_layer = nn.Linear(512*4*4, 1)
        self.aux_layer = nn.Linear(512*4*4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        validity = torch.sigmoid(self.adv_layer(x))
        label = self.aux_layer(x)

        return validity, label

# Init

netG = Generator().to(device)
netD = Discriminator().to(device)

adv_loss = nn.BCELoss()
aux_loss = nn.CrossEntropyLoss()

opt_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# Training

for epoch in range(num_epochs):
    for i, (real_imgs, labels) in enumerate(dataloader):

        real_imgs = real_imgs.to(device)
        labels = labels.to(device)

        batch_size = real_imgs.size(0)

        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

       
        # Train Discriminator
        
        opt_D.zero_grad()

        real_pred, real_aux = netD(real_imgs)
        d_real_loss = adv_loss(real_pred, valid) + aux_loss(real_aux, labels)

        z = torch.randn(batch_size, nz).to(device)
        gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        gen_imgs = netG(z, gen_labels)

        fake_pred, fake_aux = netD(gen_imgs.detach())
        d_fake_loss = adv_loss(fake_pred, fake) + aux_loss(fake_aux, gen_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        opt_D.step()

        # Train Generator
      
        opt_G.zero_grad()

        fake_pred, fake_aux = netD(gen_imgs)
        g_loss = adv_loss(fake_pred, valid) + aux_loss(fake_aux, gen_labels)

        g_loss.backward()
        opt_G.step()

        if i % 50 == 0:
            print(f"[{epoch+1}/{num_epochs}] LossD: {d_loss.item():.4f} LossG: {g_loss.item():.4f}")

    save_image(gen_imgs.data[:25], f"samples_acgan/epoch_{epoch+1}.png", nrow=5, normalize=True)

print("ACGAN Training Done!")