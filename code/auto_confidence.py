# import pandas as pd
# import re

# # CSV 파일 불러오기
# input_file = './adas_confidence/mi/choimingi_auto_1_confidence.csv'
# output_file = './adas_confidence/mi/choimingi_auto_confidence_with_time_fixed.csv'

# # CSV 파일 읽기
# df = pd.read_csv(input_file)

# # 파일명에서 영상 번호 추출 함수
# def extract_video_number(file_name):
#     match = re.search(r'cut_(\d+)of', file_name)  # 영상 번호 추출
#     if match:
#         return int(match.group(1))  # 정수형 영상 번호 반환
#     return -1  # 추출 실패 시

# # 영상 번호 컬럼 추가
# df['video_number'] = df['file_name'].apply(extract_video_number)

# # 시간 구간 계산 (영상 번호가 1부터 시작하므로 -1 해서 보정)
# df['time_sec'] = (df['video_number'] - 1) * 10 + 10

# # 영상 번호별 confidence 평균 계산
# grouped = df.groupby('time_sec')['confidence'].mean().reset_index()

# # 결과 저장
# grouped.to_csv(output_file, index=False)
# print(f"10초 단위로 시간 라벨링된 confidence 평균을 저장했습니다: {output_file}")

# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
# import os
# import glob
# import pandas as pd
# from model import Cross_model  # Cross 모델 클래스 불러오기

# # 모델 로드 함수
# def load_cross_model(model_path, device):
#     model = Cross_model(out_dim=4, valid_name='1')  # 클래스 수 = 4
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()  # 평가 모드
#     model.to(device)
#     return model

# # 이미지 전처리 함수
# def preprocess_image(image_path, image_size):
#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     image = Image.open(image_path).convert('RGB')
#     return transform(image)

# # Confidence 계산 함수
# def calculate_cross_confidence(model, visual_paths, optical_paths, device):
#     results = []
#     for vis_path, opt_path in zip(visual_paths, optical_paths):
#         visual_img = preprocess_image(vis_path, image_size=224).unsqueeze(0).unsqueeze(1).to(device)
#         optical_img = preprocess_image(opt_path, image_size=224).unsqueeze(0).unsqueeze(1).to(device)

#         with torch.no_grad():
#             outputs = model(visual_img, optical_img, device)
#             probabilities = F.softmax(outputs, dim=1)  # 확률 계산
#             confidence, predicted_class = torch.max(probabilities, dim=1)  # 가장 높은 확률
#             results.append({'file_name': vis_path, 'confidence': confidence.item()})
#     return results

# # 메인 실행 함수
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # 모델 경로
#     model_path = '/home/mmc/disk/driver_identification_old/weights/1217_cross_w10_best_fold_1.pth.pth'

#     # ADAS 이미지 디렉토리 경로
#     visual_dir = '/home/mmc/disk/driver_identification_old/images_auto/v_w10_auto/houjonguk_auto/1'
#     optical_dir = '/home/mmc/disk/driver_identification_old/images_auto/m_w10_auto/houjonguk_auto/1'

#     # 이미지 경로 수집
#     visual_paths = sorted(glob.glob(os.path.join(visual_dir, '*.jpg')))
#     optical_paths = sorted(glob.glob(os.path.join(optical_dir, '*.jpg')))

#     # 모델 로드
#     print("Loading Cross model...")
#     model = load_cross_model(model_path, device)

#     # Confidence 계산
#     print("Calculating confidence scores...")
#     results = calculate_cross_confidence(model, visual_paths, optical_paths, device)

#     # 결과를 DataFrame으로 저장
#     df = pd.DataFrame(results)
#     output_csv = './adas_confidence/cross/houjonguk_auto_1_confidence.csv'
#     df.to_csv(output_csv, index=False)
#     print(f"Results saved to {output_csv}")

# if __name__ == '__main__':
#     main()

import pandas as pd
import re

# CSV 파일 불러오기
input_file = './adas_confidence/cross/houjonguk_auto_1_confidence.csv'
output_file = './adas_confidence/cross/houjonguk_auto_cross_confidence_with_time.csv'


# CSV 파일 읽기
df = pd.read_csv(input_file)

# 영상 번호 추출 함수 (cut_{번호}of 형태에서 번호만 가져오기)
def extract_video_number(file_name):
    match = re.search(r'cut_(\d+)of', file_name)
    if match:
        return int(match.group(1))  # 영상 번호 반환
    return -1

# 영상 번호 컬럼 추가
df['video_number'] = df['file_name'].apply(extract_video_number)

# 영상 번호별 confidence 평균 계산
avg_confidence = df.groupby('video_number')['confidence'].mean().reset_index()

# 시간 라벨링: 영상 번호에 맞춰 10초 단위로 설정
avg_confidence['time_sec'] = avg_confidence['video_number'] * 10

# 결과 컬럼 정리
final_df = avg_confidence[['time_sec', 'confidence']]

# 결과 저장
final_df.to_csv(output_file, index=False)
print(f"시간 라벨링된 영상당 평균 confidence를 저장했습니다: {output_file}")

