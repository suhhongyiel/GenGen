import pandas as pd
import numpy as np
import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from monai.transforms import Compose, ScaleIntensity, Resize, EnsureType

##############################################################
# 데이터 준비
##############################################################
adni_merge_path = "/home/Lab-tutorial/Demographic.csv"
adni_data = pd.read_csv(adni_merge_path)
adni_data["DX_bl"] = adni_data["DX_bl"].str.strip()

ADNI_DIR = "/research/02-ADNI/02-output/"
MICAPIPE_DIR = ADNI_DIR + "micapipe/"

matching_records = adni_data.reset_index(drop=True)
matching_records['Index'] = matching_records.index
matching_records['ADNI_MRI_PATH'] = MICAPIPE_DIR + 'sub-ADNI' + matching_records['PTID'] + '/' + \
                                     'ses-M00' + \
                                     '/xfm/sub-ADNI' + matching_records['PTID'] + '_' + \
                                     'ses-M00' + \
                                     '_from-nativepro_brain_to-MNI152_2mm_mode-image_desc-SyN_Warped.nii.gz'
matching_records['MRI_EXISTS'] = matching_records['ADNI_MRI_PATH'].apply(lambda x: os.path.exists(x))
print(matching_records['MRI_EXISTS'])

def get_center_axial_slice(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    z_center = data.shape[2] // 2
    slice_img = data[:, :, z_center]
    slice_img = slice_img[np.newaxis, ...]  # (C,H,W)
    return slice_img

image_size = 128
monai_transforms = Compose([
    ScaleIntensity(minv=0.0, maxv=1.0),
    Resize((image_size, image_size)),
    EnsureType()
])

class MRIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df[df['MRI_EXISTS']].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        mri_path = self.df.loc[idx, 'ADNI_MRI_PATH']
        slice_img = get_center_axial_slice(mri_path)
        if self.transform:
            slice_img = self.transform(slice_img)
        return slice_img

dataset = MRIDataset(matching_records, transform=monai_transforms)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


##############################################################
# 간단한 DCGAN Generator / Discriminator 정의
##############################################################

# 하이퍼파라미터
nz = 100  # latent vector size
ngf = 64  # generator feature map size
ndf = 64  # discriminator feature map size
nc = 1    # number of channels (grayscale)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input: latent vector z -> (nz, 1, 1)
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # (ngf*8, 4,4)
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # (ngf*4, 8,8)
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # (ngf*2, 16,16)
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf, 32,32)
            nn.ConvTranspose2d(ngf, nc, 4, 4, 3, bias=False),
            nn.Sigmoid() # output between [0,1]
            # (1, 128,128)
        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input: (nc, 128,128)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf,64,64)
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2,32,32)
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
            # (ndf*4,16,16)
        )
        
        # AdaptiveAvgPool2d를 통해 (ndf*4,16,16) -> (ndf*4,1,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # 이후 1x1 Conv로 단일 스칼라 출력
        self.final_conv = nn.Conv2d(ndf*4, 1, 1, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        x = self.avgpool(x)  # (ndf*4,1,1)
        x = self.final_conv(x) # (1,1,1)
        return self.sigmoid(x).view(-1,1)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(nz, ngf, nc).to(device)
netD = Discriminator(nc, ndf).to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))

fixed_noise = torch.randn(16, nz, 1, 1, device=device)

##############################################################
# Training Loop
##############################################################
num_epochs = 50

train_D_losses = []
train_G_losses = []
val_D_losses = []
val_G_losses = []

for epoch in range(num_epochs):
    # Train
    netG.train()
    netD.train()
    total_D_loss = 0
    total_G_loss = 0
    for i, real_data in enumerate(train_loader):
        real_data = real_data.to(device)  # shape: (B,1,128,128)
        b_size = real_data.size(0)
        
        # Label
        real_label = torch.ones(b_size, 1, device=device)
        fake_label = torch.zeros(b_size, 1, device=device)

        # Update Discriminator
        optimizerD.zero_grad()
        # Real
        output = netD(real_data)
        D_real_loss = criterion(output, real_label)

        # Fake
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_data = netG(noise)
        output = netD(fake_data.detach())
        D_fake_loss = criterion(output, fake_label)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizerD.step()

        # Update Generator
        optimizerG.zero_grad()
        output = netD(fake_data)
        G_loss = criterion(output, real_label)  # want fake to be classified as real
        G_loss.backward()
        optimizerG.step()

        total_D_loss += D_loss.item()
        total_G_loss += G_loss.item()

    avg_D_loss = total_D_loss / len(train_loader)
    avg_G_loss = total_G_loss / len(train_loader)

    # Validation
    netG.eval()
    netD.eval()
    val_D_loss_total = 0
    val_G_loss_total = 0

    with torch.no_grad():
        for val_real_data in val_loader:
            val_real_data = val_real_data.to(device)
            vb_size = val_real_data.size(0)
            real_label = torch.ones(vb_size, 1, device=device)
            fake_label = torch.zeros(vb_size, 1, device=device)

            # D loss on val set (real images)
            output = netD(val_real_data)
            val_D_real_loss = criterion(output, real_label)
            
            # D loss on val set (fake images)
            noise = torch.randn(vb_size, nz, 1, 1, device=device)
            val_fake_data = netG(noise)
            output = netD(val_fake_data)
            val_D_fake_loss = criterion(output, fake_label)
            val_D_loss = val_D_real_loss + val_D_fake_loss
            val_D_loss_total += val_D_loss.item()
            
            # G loss on val set (using fake_data)
            # Generate from fixed noise for validation if preferred, here just random
            val_output = netD(val_fake_data)
            val_G_loss = criterion(val_output, real_label)
            val_G_loss_total += val_G_loss.item()

    val_avg_D_loss = val_D_loss_total / len(val_loader)
    val_avg_G_loss = val_G_loss_total / len(val_loader)

    train_D_losses.append(avg_D_loss)
    train_G_losses.append(avg_G_loss)
    val_D_losses.append(val_avg_D_loss)
    val_G_losses.append(val_avg_G_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train D: {avg_D_loss:.4f}, G: {avg_G_loss:.4f} | Val D: {val_avg_D_loss:.4f}, G: {val_avg_G_loss:.4f}")

    # 주기적으로 생성된 이미지 샘플 저장
    if (epoch+1) % 2 == 0:
        with torch.no_grad():
            sample_fake = netG(fixed_noise).cpu().numpy()
        # 첫 번째 이미지를 시각화
        plt.figure(figsize=(10,10))
        for idx in range(min(16, sample_fake.shape[0])):
            plt.subplot(4,4,idx+1)
            plt.imshow(sample_fake[idx,0], cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Generated Samples Epoch {epoch+1}')
        plt.tight_layout()
        plt.savefig(f'generated_epoch_{epoch+1}.png')
        plt.close()

##############################################################
# 손실 곡선 플롯
##############################################################
plt.figure(figsize=(10,5))
plt.plot(train_D_losses, label='Train D Loss')
plt.plot(val_D_losses, label='Val D Loss')
plt.plot(train_G_losses, label='Train G Loss')
plt.plot(val_G_losses, label='Val G Loss')
plt.title('GAN Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('gan_loss_curve.png')
plt.close()

##############################################################
# CSV로 저장
##############################################################
import csv
with open('gan_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_D_loss', 'train_G_loss', 'val_D_loss', 'val_G_loss'])
    for i in range(num_epochs):
        writer.writerow([i+1, train_D_losses[i], train_G_losses[i], val_D_losses[i], val_G_losses[i]])

##############################################################
# Inference 결과 (마지막 epoch 기준)
##############################################################
with torch.no_grad():
    sample_fake = netG(fixed_noise).cpu().numpy()
plt.figure(figsize=(10,10))
for idx in range(min(16, sample_fake.shape[0])):
    plt.subplot(4,4,idx+1)
    plt.imshow(sample_fake[idx,0], cmap='gray')
    plt.axis('off')
plt.suptitle('Final Generated Samples')
plt.tight_layout()
plt.savefig('final_generated_samples.png')
plt.close()
