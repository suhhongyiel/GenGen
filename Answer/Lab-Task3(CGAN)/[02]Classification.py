import sys
import torch
import os
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob as gg
from sklearn.metrics import precision_score, recall_score, roc_auc_score, auc, f1_score, accuracy_score, precision_recall_curve, roc_curve
from AD_pretrained_utilities import CNN_8CL_B, CNN, torch_norm, loader, predict, \
    plot_complete_report, plot_auc_curve, plot_confusion_matrix
from scipy import ndimage
# ----------------------------
# 데이터 전처리 및 변환 함수
# ----------------------------
def nifti_to_npy(nifti_folder, output_folder, label=1):
    """NIfTI(.nii) 데이터를 .npy 형식으로 변환하며 라벨을 함께 저장"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    nifti_files = gg(os.path.join(nifti_folder, '*.nii*'))
    
    for file in nifti_files:
        img = nib.load(file).get_fdata()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))  # 정규화
        filename = os.path.basename(file).replace('.nii.gz', '.npy').replace('.nii', '.npy')
        
        # 이미지를 먼저 저장
        img_filename = os.path.join(output_folder, filename.replace('.npy', '_img.npy'))
        np.save(img_filename, img)
        
        # 라벨도 저장
        label_filename = os.path.join(output_folder, filename.replace('.npy', '_label.npy'))
        np.save(label_filename, label)
        
    print(f"Converted {len(nifti_files)} NIfTI files to .npy format with label {label}.")



# ----------------------------
# 테스트 함수
# ----------------------------
# 테스트 함수
def test_model(model, batch_size, data_path, output_dir, device=torch.device('cpu')):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 데이터 로드 (라벨 파일만 포함)
    dataset = gg(os.path.join(data_path, '*_label.npy'))
    if len(dataset) == 0:
        print('No data found in the folder. Please check your path.')
        sys.exit()
    
    # preprocessing=True로 설정하여 이미지 크기 조정 활성화
    test_data = loader(dataset, transform=torch_norm, preprocessing=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # 예측 수행
    y_test, yp0, yp1, y_pred = predict(model, test_loader, device)
    
    # 클래스 수 확인
    unique_classes = np.unique(y_test)
    if len(unique_classes) < 2:
        print(f"Warning: Only one class ({unique_classes[0]}) present in y_test. Some metrics will be undefined.")
    
    # 평가
    precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall_0 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    
    # 결과 저장
    report_df = pd.DataFrame({
        'Precision_0': [precision_0], 
        'Precision_1': [precision_1],
        'Recall_0': [recall_0], 
        'Recall_1': [recall_1],
        'F1': [f1], 
        'Accuracy': [acc]
    })
    report_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    print("Evaluation completed and results saved!")

# AD_pretrained_utilities.py

def resize_data_volume_by_scale(data, scale, order=1):
    """
    Resize the data based on the provided scale.
    :param scale: float between 0 and 1 or list of scales for each dimension.
    :param order: The order of the spline interpolation, default is 1 (linear).
    """
    if isinstance(scale, float):
        scale_list = [scale, scale, scale]
    else:
        scale_list = scale
    return ndimage.zoom(data, scale_list, order=order)

def img_processing(image, scaling=0.5, final_size=[73, 96, 96]):
    """
    Resize and normalize the image.
    :param image: 3D numpy array.
    :param scaling: Initial scaling factor.
    :param final_size: Desired final size [D, H, W].
    """
    # 초기 스케일링
    image = resize_data_volume_by_scale(image, scale=scaling, order=1)
    # 최종 크기로 조정
    new_scaling = [final_size[i] / image.shape[i] for i in range(3)]
    final_image = resize_data_volume_by_scale(image, scale=new_scaling, order=1)
    return final_image



# ----------------------------
# 메인 실행
# ----------------------------
if __name__ == '__main__':
    # 설정
    nifti_input_dir = '/home/Lab-tutorial/Lab-Task3(CGAN)/'  # NIfTI 데이터 폴더 경로
    npy_output_dir = './converted_npy'             # NIfTI -> NPY 변환 폴더
    pretrained_model_path = './AD_pretrained_weights.pt'  # 사전 학습 모델 경로
    results_dir = './results/'                     # 결과 저장 경로
    batch_size = 1                                 # 배치 사이즈

    # 데이터 전처리: NIfTI -> NPY
    nifti_to_npy(nifti_input_dir, npy_output_dir)

    # 모델 불러오기
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = CNN_8CL_B()
    model = CNN(config).to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    model.eval()

    # 테스트 수행
    test_model(model, batch_size, npy_output_dir, results_dir, device)
