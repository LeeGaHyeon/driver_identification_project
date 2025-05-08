import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import glob
import pandas as pd
from model import Efficient  # 모델 클래스 불러오기

# 모델 로드 함수
def load_model(model_path, device):
    model = Efficient(enet_type='Efficient', out_dim=4)  # 클래스 수 = 4
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 평가 모드
    model.to(device)
    return model

# 이미지 전처리 함수
def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 배치 차원 추가

# Confidence 계산 및 기록
def calculate_confidence(model, image_paths, device):
    results = []
    for img_path in image_paths:
        image = preprocess_image(img_path, image_size=224).to(device)
        with torch.no_grad():
            outputs = model(image)  # 모델 예측
            probabilities = F.softmax(outputs, dim=1)  # 확률 계산
            confidence, predicted_class = torch.max(probabilities, dim=1)  # 가장 높은 확률
            results.append({'file_name': img_path, 'confidence': confidence.item()})
    return results

# 메인 실행 함수
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 경로
    model_path = './weights/1217_fine_MI_w10_best_fold_1.pth.pth'

    # ADAS 이미지 디렉토리 경로
    adas_dirs = [
        '/home/mmc/disk/driver_identification_old/images_auto/m_w10_auto/houjonguk_auto/2'
    ]

    # 이미지 경로 수집
    image_paths = []
    for adas_dir in adas_dirs:
        image_paths.extend(glob.glob(os.path.join(adas_dir, '*.jpg')))  # 모든 jpg 이미지 수집

    # 모델 로드
    print("Loading model...")
    model = load_model(model_path, device)

    # Confidence 계산
    print("Calculating confidence scores...")
    results = calculate_confidence(model, image_paths, device)

    # 결과를 DataFrame으로 저장
    df = pd.DataFrame(results)
    output_csv = './adas_confidence/mi/houjonguk_auto_2_confidence.csv'
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == '__main__':
    main()