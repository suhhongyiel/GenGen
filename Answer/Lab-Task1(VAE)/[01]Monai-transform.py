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

#########################################
# 기존 데이터 준비
#########################################
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


#########################################
# MRI 로드 및 축 방향 슬라이스 추출
#########################################
def get_center_axial_slice(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()  # shape: (X, Y, Z)
    z_center = data.shape[2] // 2
    slice_img = data[:, :, z_center]
    # (H, W) -> (C, H, W)
    slice_img = slice_img[np.newaxis, ...]
    return slice_img

#########################################
# Monai Transform 정의
#########################################
image_size = 128
monai_transforms = Compose([
    ScaleIntensity(minv=0.0, maxv=1.0),
    Resize((image_size, image_size)),
    EnsureType()
])

#########################################
# 데이터셋 정의
#########################################
class MRIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df[df['MRI_EXISTS']].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        mri_path = self.df.loc[idx, 'ADNI_MRI_PATH']
        slice_img = get_center_axial_slice(mri_path)

        # Monai transform 적용
        if self.transform:
            slice_img = self.transform(slice_img)

        return slice_img

#########################################
# VAE 모델 정의
#########################################
class SimpleVAE(nn.Module):
    def __init__(self, image_size=128):
        super(SimpleVAE, self).__init__()
        self.fc1 = nn.Linear(image_size*image_size, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, image_size*image_size)
        self.image_size = image_size
        
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.image_size*self.image_size))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, reduction='mean'):
    # reduction='mean'으로 설정해서 값이 너무 크지 않게 조정
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, recon_x.size(-1)), reduction=reduction)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE.item(), KLD.item()

#########################################
# Train / Val 나누기
#########################################
dataset = MRIDataset(matching_records, transform=monai_transforms)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

#########################################
# 모델 학습
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleVAE(image_size=image_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

# metric 기록용 리스트
train_total_losses = []
train_bce_losses = []
train_kld_losses = []
val_total_losses = []
val_bce_losses = []
val_kld_losses = []

for epoch in range(num_epochs):
    model.train()
    train_total_loss = 0
    train_bce_loss = 0
    train_kld_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss, bce_val, kld_val = vae_loss_function(recon, batch, mu, logvar, reduction='mean')
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item()
        train_bce_loss += bce_val
        train_kld_loss += kld_val

    train_total_loss /= len(train_loader)
    train_bce_loss /= len(train_loader)
    train_kld_loss /= len(train_loader)

    # Validation
    model.eval()
    val_total_loss_ = 0
    val_bce_loss_ = 0
    val_kld_loss_ = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, bce_val, kld_val = vae_loss_function(recon, batch, mu, logvar, reduction='mean')
            val_total_loss_ += loss.item()
            val_bce_loss_ += bce_val
            val_kld_loss_ += kld_val

    val_total_loss_ /= len(val_loader)
    val_bce_loss_ /= len(val_loader)
    val_kld_loss_ /= len(val_loader)

    train_total_losses.append(train_total_loss)
    train_bce_losses.append(train_bce_loss)
    train_kld_losses.append(train_kld_loss)
    val_total_losses.append(val_total_loss_)
    val_bce_losses.append(val_bce_loss_)
    val_kld_losses.append(val_kld_loss_)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_total_loss:.4f}, Val Loss: {val_total_loss_:.4f}")

#########################################
# Loss Curve Plot
#########################################
plt.figure(figsize=(10,5))
plt.plot(train_total_losses, label='Train Loss')
plt.plot(val_total_losses, label='Val Loss')
plt.title('VAE Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.close()

# BCE, KLD curve (optional)
plt.figure(figsize=(10,5))
plt.plot(train_bce_losses, label='Train BCE')
plt.plot(val_bce_losses, label='Val BCE')
plt.plot(train_kld_losses, label='Train KLD')
plt.plot(val_kld_losses, label='Val KLD')
plt.title('BCE and KLD Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('metrics_curve.png')
plt.close()

#########################################
# CSV로 Metric 저장
#########################################
import csv
with open('metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_total_loss', 'train_bce', 'train_kld', 'val_total_loss', 'val_bce', 'val_kld'])
    for i in range(num_epochs):
        writer.writerow([i+1, train_total_losses[i], train_bce_losses[i], train_kld_losses[i],
                         val_total_losses[i], val_bce_losses[i], val_kld_losses[i]])

#########################################
# Inference 결과 시각화 (한 배치)
#########################################
model.eval()
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        recon, mu, logvar = model(batch)
        # 첫 번째 샘플에 대한 원본 vs 재구성 비교
        original = batch[0].cpu().squeeze().numpy()
        reconstructed = recon[0].cpu().view(1, image_size, image_size).numpy().squeeze()

        # Plot original and reconstructed
        plt.figure(figsize=(8,4))
        
        plt.subplot(1,2,1)
        plt.title('Original')
        plt.imshow(original, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.title('Reconstructed')
        plt.imshow(reconstructed, cmap='gray')
        plt.axis('off')

        plt.savefig('inference_example.png')
        plt.close()
        break
