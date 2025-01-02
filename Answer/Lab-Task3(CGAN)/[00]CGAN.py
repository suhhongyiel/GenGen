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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.functional import mean_absolute_error as mae_torch

##############################################################
# 데이터 준비
# 여기서 DX_bl 정보: AD=1.0, CN=0.0 으로 condition vector에 반영.
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

image_size = 64
monai_transforms = Compose([
    ScaleIntensity(minv=0.0, maxv=1.0),
    Resize((image_size, image_size, image_size)),
    EnsureType()
])

def load_nii(path):
    img = nib.load(path)
    data = img.get_fdata()
    data = np.expand_dims(data, axis=0) # (1,X,Y,Z)
    return data

class MRIDataset(Dataset):
    def __init__(self, df, transform=None, scale_funcs=None):
        self.df = df[df['MRI_EXISTS']].reset_index(drop=True)
        self.transform = transform
        self.scale_funcs = scale_funcs  # {'age': func, 'pteducat': func}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        mri_path = self.df.loc[idx, 'ADNI_MRI_PATH']
        data = load_nii(mri_path)
        if self.transform:
            data = self.transform(data)

        # DX_bl: AD=1.0, CN=0.0
        dx = self.df.loc[idx, 'DX_bl']
        dx_val = 1.0 if dx == 'AD' else 0.0

        age = self.df.loc[idx, 'AGE']
        ptedu = self.df.loc[idx, 'PTEDUCAT']

        if self.scale_funcs is not None:
            age_val = self.scale_funcs['age'](age)
            ptedu_val = self.scale_funcs['pteducat'](ptedu)
        else:
            age_val = age
            ptedu_val = ptedu

        # condition vector = [DX, Age, Education]
        # DX를 통해 AD/CN 생성 가능
        cond_vec = np.array([dx_val, age_val, ptedu_val], dtype=np.float32)
        return data, cond_vec

##############################################################
# Test 데이터 고정 (마지막 5개)
##############################################################
test_df = matching_records.tail(5).reset_index(drop=True)
train_val_df = matching_records.iloc[:-5].reset_index(drop=True)

train_val_dataset = MRIDataset(train_val_df, transform=monai_transforms)
test_dataset = MRIDataset(test_df, transform=monai_transforms)

# Train/Val Split
total_len = len(train_val_dataset)
train_size = int(0.7 * total_len)
val_size = total_len - train_size
train_dataset_, val_dataset_ = random_split(train_val_dataset, [train_size, val_size])

##############################################################
# Train/Val 데이터로부터 스케일링 파라미터 계산
##############################################################
def extract_cond_values(subset, dataset_obj):
    ages = []
    ptedus = []
    for idx in range(len(subset)):
        real_idx = subset.indices[idx]
        row = dataset_obj.df.iloc[real_idx]
        ages.append(row['AGE'])
        ptedus.append(row['PTEDUCAT'])
    return np.array(ages), np.array(ptedus)

train_ages, train_ptedu = extract_cond_values(train_dataset_, train_val_dataset)
val_ages, val_ptedu = extract_cond_values(val_dataset_, train_val_dataset)

all_ages = np.concatenate([train_ages, val_ages])
all_ptedu = np.concatenate([train_ptedu, val_ptedu])

age_min, age_max = all_ages.min(), all_ages.max()
ptedu_min, ptedu_max = all_ptedu.min(), all_ptedu.max()

def scale_age(age):
    return (age - age_min) / (age_max - age_min)

def scale_pteducat(p):
    return (p - ptedu_min) / (ptedu_max - ptedu_min)

# 스케일링 함수를 적용한 Dataset 재생성
train_dataset = MRIDataset(train_val_dataset.df.iloc[train_dataset_.indices], 
                           transform=monai_transforms, 
                           scale_funcs={'age':scale_age, 'pteducat':scale_pteducat})
val_dataset = MRIDataset(train_val_dataset.df.iloc[val_dataset_.indices], 
                         transform=monai_transforms, 
                         scale_funcs={'age':scale_age, 'pteducat':scale_pteducat})
test_dataset = MRIDataset(test_dataset.df, 
                          transform=monai_transforms, 
                          scale_funcs={'age':scale_age, 'pteducat':scale_pteducat})

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("Train:", len(train_dataset), "Val:", len(val_dataset), "Test:", len(test_dataset))

##############################################################
# 3D Conditional DCGAN
# Condition vector를 통해 AD(1) / CN(0) MRI를 Conditional 하게 생성 가능
##############################################################
nz = 100
cond_dim = 3
input_z_dim = nz + cond_dim
ngf = 16
ndf = 16
nc = 1

class Generator3D(nn.Module):
    def __init__(self, input_z_dim, ngf, nc):
        super(Generator3D, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(input_z_dim, ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ngf*8),
            nn.ReLU(True),
            # (ngf*8,4,4,4)
            nn.ConvTranspose3d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf*4),
            nn.ReLU(True),
            # (ngf*4,8,8,8)
            nn.ConvTranspose3d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf*2),
            nn.ReLU(True),
            # (ngf*2,16,16,16)
            nn.ConvTranspose3d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # (ngf,32,32,32)
            # 여기서 64³를 얻기 위해 한 번 더 stride=2 upsampling
            nn.ConvTranspose3d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
            # (nc,64,64,64)
        )

    def forward(self, z, c):
        c_reshaped = c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        zc = torch.cat([z, c_reshaped], dim=1)
        return self.main(zc)

class Discriminator3D(nn.Module):
    def __init__(self, nc, ndf, cond_dim):
        super(Discriminator3D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf,32,32,32)
            nn.Conv3d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2,16,16,16)
            nn.Conv3d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4,8,8,8)
        )

        self.cond_fc = nn.Sequential(
            nn.Linear(cond_dim, ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.final_conv = nn.Conv3d(ndf*8, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, c):
        feat = self.main(x) # (B, ndf*4,8,8,8)
        pooled_feat = self.avgpool(feat) # (B, ndf*4,1,1,1)
        c_feat = self.cond_fc(c).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        combined = torch.cat([pooled_feat, c_feat], dim=1) # (B,ndf*8,1,1,1)
        out = self.final_conv(combined)
        return torch.sigmoid(out).view(-1,1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator3D(input_z_dim, ngf, nc).to(device)
netD = Discriminator3D(nc, ndf, cond_dim).to(device)

criterion = nn.BCELoss()
# LR 조정 (옵션): lr=1e-4 또는 5e-5 시도
optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5,0.999))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5,0.999))


num_epochs = 100
train_D_losses = []
train_G_losses = []
val_D_losses = []
val_G_losses = []

# Validation fixed noise & cond
val_iter = iter(val_loader)
val_data, val_cond = next(val_iter)
val_cond = val_cond.to(device)
fixed_noise = torch.randn(val_data.size(0), nz, 1, 1, 1, device=device)

for epoch in range(num_epochs):
    netG.train()
    netD.train()
    total_D_loss = 0
    total_G_loss = 0
    for real_data, cond_vec in train_loader:
        real_data = real_data.to(device)
        cond_vec = cond_vec.to(device)
        b_size = real_data.size(0)

        real_label = torch.ones(b_size,1,device=device)
        fake_label = torch.zeros(b_size,1,device=device)

        # Update D
        optimizerD.zero_grad()
        output = netD(real_data, cond_vec)
        D_real_loss = criterion(output, real_label)

        noise = torch.randn(b_size, nz, 1,1,1, device=device)
        fake_data = netG(noise, cond_vec)
        output = netD(fake_data.detach(), cond_vec)
        D_fake_loss = criterion(output, fake_label)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        optimizerD.step()

        # Update G
        optimizerG.zero_grad()
        output = netD(fake_data, cond_vec)
        G_loss = criterion(output, real_label)
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
        for val_real_data, val_cond_vec in val_loader:
            val_real_data = val_real_data.to(device)
            val_cond_vec = val_cond_vec.to(device)
            vb_size = val_real_data.size(0)
            real_label = torch.ones(vb_size,1,device=device)
            fake_label = torch.zeros(vb_size,1,device=device)

            output = netD(val_real_data, val_cond_vec)
            val_D_real_loss = criterion(output, real_label)
            noise = torch.randn(vb_size, nz, 1,1,1, device=device)
            val_fake_data = netG(noise, val_cond_vec)
            output = netD(val_fake_data, val_cond_vec)
            val_D_fake_loss = criterion(output, fake_label)
            val_D_loss = val_D_real_loss + val_D_fake_loss
            val_D_loss_total += val_D_loss.item()

            val_output = netD(val_fake_data, val_cond_vec)
            val_G_loss_ = criterion(val_output, real_label)
            val_G_loss_total += val_G_loss_.item()

    val_avg_D_loss = val_D_loss_total / len(val_loader)
    val_avg_G_loss = val_G_loss_total / len(val_loader)

    train_D_losses.append(avg_D_loss)
    train_G_losses.append(avg_G_loss)
    val_D_losses.append(val_avg_D_loss)
    val_G_losses.append(val_avg_G_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train D: {avg_D_loss:.4f}, G: {avg_G_loss:.4f} | Val D: {val_avg_D_loss:.4f}, G: {val_avg_G_loss:.4f}")

plt.figure(figsize=(10,5))
plt.plot(train_D_losses, label='Train D')
plt.plot(val_D_losses, label='Val D')
plt.plot(train_G_losses, label='Train G')
plt.plot(val_G_losses, label='Val G')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('3D cGAN Loss Curves')
plt.legend()
plt.savefig('3d_cgan_loss_curve.png')
plt.close()

import csv
with open('3d_cgan_metrics.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch','train_D_loss','train_G_loss','val_D_loss','val_G_loss'])
    for i in range(num_epochs):
        writer.writerow([i+1, train_D_losses[i], train_G_losses[i], val_D_losses[i], val_G_losses[i]])

##############################################################
# Test Set Inference & Metrics (AD/CN 별로 생성 가능)
##############################################################
netG.eval()
num_test_samples = 5
test_psnr_list = []
test_ssim_list = []
test_mae_list = []

for i, (test_data, test_cond) in enumerate(test_loader):
    if i >= num_test_samples:
        break
    test_data = test_data.to(device)
    test_cond = test_cond.to(device)
    with torch.no_grad():
        # 원하는 이미지 타입 생성:
        # 예) AD 이미지 생성: test_cond[:,0] = 1.0
        #    CN 이미지 생성: test_cond[:,0] = 0.0
        # 아래는 원본 test_cond 그대로 사용
        noise = torch.randn(1, nz, 1,1,1, device=device)
        gen_data = netG(noise, test_cond)

    real_vol = test_data[0,0].cpu().numpy()
    fake_vol = gen_data[0,0].cpu().numpy()
    print(real_vol.shape, fake_vol.shape)

    real_vol = np.squeeze(real_vol) 
    fake_vol = np.squeeze(fake_vol)
    print("After shape: ", real_vol.shape, fake_vol.shape)

    cur_psnr = psnr(real_vol, fake_vol, data_range=1.0)
    cur_ssim = ssim(real_vol, fake_vol, data_range=1.0, multichannel=False)
    real_tensor = torch.tensor(real_vol, dtype=torch.float32)
    fake_tensor = torch.tensor(fake_vol, dtype=torch.float32)
    cur_mae = mae_torch(real_tensor, fake_tensor).item()

    test_psnr_list.append(cur_psnr)
    test_ssim_list.append(cur_ssim)
    test_mae_list.append(cur_mae)

    # NIfTI 저장
    gen_nii = nib.Nifti1Image(fake_vol, np.eye(4))
    nib.save(gen_nii, f'generated_test_{i}.nii.gz')

with open('3d_test_metrics.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['sample_idx','PSNR','SSIM','MAE'])
    for i in range(num_test_samples):
        writer.writerow([i, test_psnr_list[i], test_ssim_list[i], test_mae_list[i]])

print("3D test metrics saved to 3d_test_metrics.csv")


##############################################################
# AD 환자 MRI vs CN 환자 MRI 생성법
#
# 실제로 학습된 모델을 이용할 때, condition vector의 첫 번째 값(DX)으로 AD=1.0, CN=0.0을
# 원하는 대로 설정하여 동일한 noise로 AD/CN MRI를 비교할 수 있음.
#
# 예:
# c_AD = torch.tensor([[1.0, scaled_age, scaled_pteducat]], device=device)
# c_CN = torch.tensor([[0.0, scaled_age, scaled_pteducat]], device=device)
#
# noise = torch.randn(1, nz, 1,1,1, device=device)
# gen_AD = netG(noise, c_AD)
# gen_CN = netG(noise, c_CN)
#
# 이렇게 하면 동일한 latent vector에서도 AD/CN 조건만 바꿔 두 가지 타입의 이미지 생성 가능.
##############################################################
