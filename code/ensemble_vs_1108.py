import pandas as pd
from sklearn.metrics import f1_score, average_precision_score, accuracy_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np

def calculate_map(y_true, y_scores, num_classes):
    
    #각 클래스에 대해 AP 계산
    average_precisions = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        ap = np.trapz(recall, precision)  # AP를 적분으로 계산
        average_precisions.append(ap)
    # 모든 클래스에 대해 AP 평균 계산하여 mAP 반환

    return np.mean(average_precisions)

def evaluate_ensemble(rounds=4, courses=3):
    # 라운드 및 코스에 대한 전체 성능을 저장할 변수
    total_accuracy, total_top3_accuracy, total_f1_score, total_map_score = 0, 0, 0, 0
    total_course_accuracy, total_course_top3_accuracy, total_course_f1, total_course_map = 0, 0, 0, 0

    # 라운드별 평가
    for round_num in range(1, rounds + 1):
        # 비디오와 CAN 데이터만 로드
        vi_round_avg_C = pd.read_csv(f'./prob/0902_AugAug2Cross_w30_logits_{round_num}.csv')
        can_round_C_avg_prob = pd.read_csv(f'./prob/sensor_round_{round_num}_avg_prob.csv')

        val_col = f'val_{round_num}'
        merged_df = vi_round_avg_C.merge(can_round_C_avg_prob, on=val_col, suffixes=('_vi', '_can'))

        label_columns = [str(i) for i in range(15)]
        for label in label_columns:
            merged_df[label + '_avg'] = (merged_df[label + '_vi'] + merged_df[label + '_can']) / 2

        merged_df['predicted_label'] = merged_df[[label + '_avg' for label in label_columns]].idxmax(axis=1)
        merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_avg', '').astype(int)
        accuracy = (merged_df[val_col] == merged_df['predicted_label']).mean()

        merged_df['top_3_labels'] = merged_df[[label + '_avg' for label in label_columns]].apply(
            lambda row: row.nlargest(3).index.str.replace('_avg', '').astype(int).tolist(), axis=1)
        top3_accuracy = merged_df.apply(lambda row: row[val_col] in row['top_3_labels'], axis=1).mean()

        f1 = f1_score(merged_df[val_col], merged_df['predicted_label'], average='macro')

        # y_true = label_binarize(merged_df[val_col], classes=[int(label) for label in label_columns])
        # y_scores = merged_df[[label + '_avg' for label in label_columns]].values
        # map_score = calculate_map(y_true, y_scores, num_classes=len(label_columns))

        ######################## 1107

        y_true_bin = label_binarize(merged_df[val_col], classes=[int(label) for label in label_columns])

        # y_scores: 각 클래스에 대한 예측 점수 배열
        y_scores = merged_df[[label + '_avg' for label in label_columns]].values

        # mAP 계산
        average_precisions = []
        for i in range(y_true_bin.shape[1]):  # 각 클래스별 반복
            # Precision-Recall 곡선 계산
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            # 곡선 아래 면적(AP) 계산
            ap = np.trapz(recall, precision)  # trapz로 AP 직접 계산
            average_precisions.append(ap)

        # mAP 계산 (모든 클래스에 대해 AP 평균)
        map_score = np.mean(average_precisions)

        # sklearn 제공 함수로 mAP 계산 (검증)
        map_score_sklearn = average_precision_score(y_true_bin, y_scores, average='macro')

        ######################## 1107
        print(f"Round {round_num} - Accuracy: {accuracy * 100:.2f}%")
        print(f"Round {round_num} - Top-3 Accuracy: {top3_accuracy * 100:.2f}%")
        print(f"Round {round_num} - F1-Score: {f1 * 100:.2f}%")
        print(f"Round {round_num} - Mean Average Precision (mAP): {map_score * 100:.2f}%\n")

        total_accuracy += accuracy
        total_top3_accuracy += top3_accuracy
        total_f1_score += f1
        total_map_score += map_score

    # 코스별 평가
    for course_label in ['A', 'B', 'C'][:courses]:
        vi_course_avg_C = pd.read_csv(f'./prob/0902_AugAug2Cross_w30_logits_{course_label}.csv')
        can_course_C_avg_prob = pd.read_csv(f'./prob/sensor_course_{course_label}_avg_prob.csv')

        val_col = f'val_{course_label}'
        merged_df = vi_course_avg_C.merge(can_course_C_avg_prob, on=val_col, suffixes=('_vi', '_can'))

        for label in label_columns:
            merged_df[label + '_avg'] = (merged_df[label + '_vi'] + merged_df[label + '_can']) / 2

        merged_df['predicted_label'] = merged_df[[label + '_avg' for label in label_columns]].idxmax(axis=1)
        merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_avg', '').astype(int)
        accuracy = (merged_df[val_col] == merged_df['predicted_label']).mean()

        merged_df['top_3_labels'] = merged_df[[label + '_avg' for label in label_columns]].apply(
            lambda row: row.nlargest(3).index.str.replace('_avg', '').astype(int).tolist(), axis=1)
        top3_accuracy = merged_df.apply(lambda row: row[val_col] in row['top_3_labels'], axis=1).mean()

        f1 = f1_score(merged_df[val_col], merged_df['predicted_label'], average='macro')

        y_true = label_binarize(merged_df[val_col], classes=[int(label) for label in label_columns])
        y_scores = merged_df[[label + '_avg' for label in label_columns]].values
        map_score = calculate_map(y_true, y_scores, num_classes=len(label_columns))

        print(f"Course {course_label} - Accuracy: {accuracy * 100:.2f}%")
        print(f"Course {course_label} - Top-3 Accuracy: {top3_accuracy * 100:.2f}%")
        print(f"Course {course_label} - F1-Score: {f1 * 100:.2f}%")
        print(f"Course {course_label} - Mean Average Precision (mAP): {map_score * 100:.2f}%\n")

        total_course_accuracy += accuracy
        total_course_top3_accuracy += top3_accuracy
        total_course_f1 += f1
        total_course_map += map_score

    # 각 라운드의 평균 성능 출력
    print(f"\nAverage across {rounds} rounds:")
    print(f"Accuracy: {(total_accuracy / rounds) * 100:.2f}%")
    print(f"Top-3 Accuracy: {(total_top3_accuracy / rounds) * 100:.2f}%")
    print(f"F1-Score: {(total_f1_score / rounds) * 100:.2f}%")
    print(f"Mean Average Precision (mAP): {(total_map_score / rounds) * 100:.2f}%")

    # 각 코스의 평균 성능 출력
    print(f"\nAverage across {courses} courses:")
    print(f"Accuracy: {(total_course_accuracy / courses) * 100:.2f}%")
    print(f"Top-3 Accuracy: {(total_course_top3_accuracy / courses) * 100:.2f}%")
    print(f"F1-Score: {(total_course_f1 / courses) * 100:.2f}%")
    print(f"Mean Average Precision (mAP): {(total_course_map / courses) * 100:.2f}%")

if __name__ == "__main__":
    evaluate_ensemble()
