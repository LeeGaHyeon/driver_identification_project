import os
import glob
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import Efficient  # 학습된 EfficientNet VI 모델 가져오기

# Grad-CAM 클래스
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.detach().cpu().numpy()

# 이미지 전처리 함수
def preprocess_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# Grad-CAM 실행 함수
def generate_gradcam_for_folder(model_path, input_folder, output_folder, image_size=224):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 학습된 EfficientNet VI 모델 로드
    model = Efficient(enet_type='tf_efficientnet_b0_ns', out_dim=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Grad-CAM 대상 레이어 설정
    target_layer = model.enet.features[-1]  # EfficientNet 마지막 Conv 레이어
    grad_cam = GradCAM(model, target_layer)

    # 입력 폴더에서 이미지 목록 가져오기
    image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        # 이미지 전처리
        input_tensor = preprocess_image(image_path, image_size=image_size).to(device)
        cam = grad_cam.generate_cam(input_tensor)

        # 원본 이미지 불러오기
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size, image_size))  # 모델 입력 크기에 맞게 리사이즈

        # Grad-CAM Heatmap 생성
        heatmap = cv2.resize(cam, (image_size, image_size))  # Grad-CAM 결과 크기 조정
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        # Heatmap과 원본 이미지 합성
        superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

        # 저장 경로 설정
        file_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, file_name)

        # Grad-CAM 결과 저장
        cv2.imwrite(output_path, superimposed_img)

        print(f"Grad-CAM 생성 완료: {output_path}")

# 실행
if __name__ == '__main__':
    model_path = './weights/1217_fine_VI_w10_best_fold_1.pth.pth'  # 학습된 VI 모델 경로
    input_folder = '/home/mmc/disk/driver_identification_old/images_auto/v_w10_auto/choimingi_auto/1'  # 입력 이미지 폴더
    output_folder = '/home/mmc/disk/driver_identification_old/images_auto/grad_cam/v_w10_auto/choimingi_auto/1'  # Grad-CAM 결과 저장 폴더

    generate_gradcam_for_folder(model_path, input_folder, output_folder)
