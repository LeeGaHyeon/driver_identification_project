import numpy as np

# 주어진 교차 검증 결과 데이터
cross_validation_results = {
    'leekanghyuk_2': [89, 86, 26, 182, 79, 69, 89],
    'kimgangsu': [86, 60, 83, 161, 118, 109, 83],
    'kangjihyun': [82, 73, 65, 181, 107, 95, 107],
    'hurhongjun': [82, 78, 24, 191, 107, 106, 106],
    'leeeunseo': [78, 47, 53, 188, 94, 96, 96],
    'chunjihun': [77, 83, 61, 197, 82, 100, 100],
    'leegihun': [77, 36, 56, 82, 51, 81, 81],
    'ohseunghun': [74, 67, 43, 196, 93, 99, 99],
    'seosanghyeok': [73, 80, 85, 199, 98, 96, 96],
    'jojeongduk': [73, 84, 39, 153, 61, 92, 92],
    'leeseunglee': [70, 78, 67, 224, 104, 67, 67],
    'leejaeho_2': [65, 69, 55, 173, 105, 87, 87],
    'leegahyeon_2': [64, 72, 58, 187, 101, 96, 96],
    'kangminjae': [62, 83, 36, 208, 98, 109, 109],
    'cheonaeji': [62, 57, 25, 188, 70, 96, 96],
    'leeyunguel_2': [58, 84, 59, 168, 90, 99, 99],
    'leegahyeon': [56, 33, 61, 80, 34, 48, 48],
    'kimminju': [54, 22, 40, 113, 61, 69, 69],
    'choimingi': [51, 6, 12, 76, 56, 23, 23],
    'leejaeho': [51, 20, 14, 117, 79, 53, 53],
    'leegihun_2': [46, 65, 69, 142, 74, 97, 97],
    'baekseungdo': [43, 45, 54, 218, 65, 95, 95],
    'simboseok': [33, 42, 46, 126, 57, 64, 64],
    'anseonyeong': [24, 26, 54, 171, 91, 117, 117],
    'leekanghyuk': [16, 19, 25, 99, 59, 26, 26],
    'jeongyubin': [11, 29, 57, 107, 69, 44, 44],
    'leeyunguel': [21, 33, 33, 88, 65, 60, 60]
}

# 각 클래스의 평균 잘못 분류된 횟수를 계산
average_errors = {cls: np.mean(errors) for cls, errors in cross_validation_results.items()}

# 평균 잘못 분류된 횟수가 적은 상위 15개 클래스를 선택
sorted_classes = sorted(average_errors, key=average_errors.get)
top_15_classes = sorted_classes[:17]

print("성능이 좋은 상위 15개 클래스:")
for cls in top_15_classes:
    print(f"{cls}: 평균 {average_errors[cls]:.2f}개")

# 성능이 좋은 15개 클래스를 제외한 나머지 클래스 제거
removed_classes = sorted_classes[15:]

print("\n제거된 클래스:")
for cls in removed_classes:
    print(f"{cls}: 평균 {average_errors[cls]:.2f}개")
