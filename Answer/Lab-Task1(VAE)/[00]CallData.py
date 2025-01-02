import pandas as pd
import numpy as np
import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#########################################
# 기존 코드
#########################################
# 1. Demographic.csv 읽기 (ADNIMERGE.csv 역할을 하는 것으로 가정)
adni_merge_path = "/home/Lab-tutorial/Demographic.csv"  # ADNIMERGE.csv 파일 경로
adni_data = pd.read_csv(adni_merge_path)

# DX_bl 컬럼을 사용해 AD(1)와 CN(0)을 필터링
adni_data["DX_bl"] = adni_data["DX_bl"].str.strip()  # 문자열 공백 제거
ad_subjects = adni_data[adni_data["DX_bl"] == "AD"]
cn_subjects = adni_data[adni_data["DX_bl"] == "CN"]

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
# 추가 코드: MRI 로드 및 축 방향 슬라이스 추출
#########################################

# 중앙 슬라이스(축 방향: z-axis) 추출 함수
def get_center_axial_slice(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    # 데이터 shape: (X, Y, Z)
    # 중앙 z-slice 추출
    z_center = data.shape[2] // 2
    slice_img = data[:, :, z_center]
    # 정규화 (필요에 따라 수정 가능)
    slice_img = (slice_img - np.mean(slice_img)) / (np.std(slice_img) + 1e-8)
    # 2D 이미지를 (C,H,W) 형태로 맞추기 위해 채널 차원 추가
    slice_img = slice_img[np.newaxis, ...]
    return slice_img

#########################################
# VAE 모델 정의 (간단한 예시)
#########################################
class SimpleVAE(nn.Module):
    def __init__(self, image_size=128):
        super(SimpleVAE, self).__init__()
        # 인코더
        self.fc1 = nn.Linear(image_size*image_size, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        
        # 디코더
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

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, recon_x.size(-1)), reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

#########################################
# 데이터셋 및 데이터로더 정의
#########################################
class MRIDataset(Dataset):
    def __init__(self, df, transform=None, image_size=128):
        self.df = df
        self.transform = transform
        self.image_size = image_size

        # MRI_EXISTS가 True인 경우에만 필터
        self.df = self.df[self.df['MRI_EXISTS'] == True].reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        mri_path = self.df.loc[idx, 'ADNI_MRI_PATH']
        slice_img = get_center_axial_slice(mri_path)
        # 필요한 경우 리사이즈
        # 여기서는 단순히 Center Crop 또는 Resize를 예시로 torch와 torchvision 사용 가능
        # 여기서는 단순히 numpy로 이미지를 (image_size, image_size)로 리사이즈하는 예를 듭니다.
        # 실제로는 skimage.transform.resize 등을 사용하거나, torch transforms을 사용할 수 있습니다.
        from skimage.transform import resize
        slice_img = resize(slice_img, (1, self.image_size, self.image_size), mode='constant')
        

        # 다시 한 번 0~1 유지 확인 (resize 과정에서 interpolation으로 범위 밖 값이 나오지 않도록)
        slice_min = slice_img.min()
        slice_max = slice_img.max()
        if slice_max - slice_min > 0:
            slice_img = (slice_img - slice_min) / (slice_max - slice_min)
        else:
            slice_img = np.zeros_like(slice_img)

        # numpy -> torch tensor 변환
        slice_tensor = torch.tensor(slice_img, dtype=torch.float32)
        return slice_tensor


#########################################
# 모델 학습 예시
#########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 128

dataset = MRIDataset(matching_records, image_size=image_size)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SimpleVAE(image_size=image_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = vae_loss_function(recon, batch, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(dataloader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

#########################################
# 추론 예시 (한 배치의 재구성 결과 확인)
#########################################
model.eval()
with torch.no_grad():
    for batch in dataloader:
        batch = batch.to(device)
        recon, mu, logvar = model(batch)
        # recon 이미지를 numpy로 변환 후 시각화 등 가능
        # recon 이미지 shape: (B, image_size*image_size)
        recon_img = recon[0].cpu().view(1, image_size, image_size).numpy()
        # 여기서 matplotlib 등을 사용해 화면에 표시할 수 있습니다.
        # 예: 
        # import matplotlib.pyplot as plt
        # plt.imshow(recon_img[0], cmap='gray')
        # plt.show()
        break