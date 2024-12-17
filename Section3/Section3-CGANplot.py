import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# nii.gz 파일들이 저장된 디렉토리
nii_dir = "./"  # 실제 경로로 변경하세요
files = [f for f in os.listdir(nii_dir) if f.endswith(".nii.gz")]
files = sorted(files)  # 파일명 정렬 (옵션)

# 플롯 크기 및 레이아웃 설정
num_files = len(files)
cols = 5  # 한 행에 5개씩 표시 (파일 개수에 따라 조정)
rows = (num_files + cols - 1) // cols  # 필요한 행의 수 계산

fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
axes = axes.flatten() if num_files > 1 else [axes]

for i, fname in enumerate(files):
    nii_path = os.path.join(nii_dir, fname)
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    # 가운데 z-slice index
    z_center = data.shape[2] // 2
    slice_img = data[:, :, z_center]

    # 이미지 표시
    ax = axes[i]
    ax.imshow(slice_img.T, cmap='gray', origin='lower')
    ax.set_title(fname)
    ax.axis('off')

# 남는 subplot 비우기
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()

# figure 저장
save_path = "all_center_slices.png"
plt.savefig(save_path, dpi=300)
print(f"Figure saved to {save_path}")
