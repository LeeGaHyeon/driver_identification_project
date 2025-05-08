import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np

def calculate_map(y_true, y_scores, num_classes):
    # 각 클래스에 대해 AP 계산
    average_precisions = []
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        ap = np.trapz(recall, precision)  # AP를 적분으로 계산
        average_precisions.append(ap)
    # 모든 클래스에 대해 AP 평균 계산하여 mAP 반환
    return np.mean(average_precisions)

def ensemble_metrics(rounds=4, courses=3):
    total_accuracy, total_top3_accuracy, total_f1_score, total_map_score = 0, 0, 0, 0
    total_course_accuracy, total_course_top3_accuracy, total_course_f1, total_course_map = 0, 0, 0, 0

    # 라운드별 평가
    for round_num in range(1, rounds + 1):
        # 비디오와 CAN 데이터만 로드
        vi_round_avg_C = pd.read_csv(f'./prob/0902_AugAug2Cross_w30_logits_{round_num}.csv')
        can_round_C_avg_prob = pd.read_csv(f'./prob/can_round_{round_num}_avg_prob.csv')

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

        y_true = label_binarize(merged_df[val_col], classes=[int(label) for label in label_columns])
        y_scores = merged_df[[label + '_avg' for label in label_columns]].values
        map_score = calculate_map(y_true, y_scores, num_classes=len(label_columns))

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
        can_course_C_avg_prob = pd.read_csv(f'./prob/can_course_{course_label}_avg_prob.csv')

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
    avg_accuracy = total_accuracy / rounds
    avg_top3_accuracy = total_top3_accuracy / rounds
    avg_f1_score = total_f1_score / rounds
    avg_map_score = total_map_score / rounds

    print(f"\nAverage across {rounds} rounds:")
    print(f"Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy: {avg_top3_accuracy * 100:.2f}%")
    print(f"F1-Score: {avg_f1_score * 100:.2f}%")
    print(f"Mean Average Precision (mAP): {avg_map_score * 100:.2f}%")

    # 각 코스의 평균 성능 출력
    avg_course_accuracy = total_course_accuracy / courses
    avg_course_top3_accuracy = total_course_top3_accuracy / courses
    avg_course_f1 = total_course_f1 / courses
    avg_course_map = total_course_map / courses

    print(f"\nAverage across {courses} courses:")
    print(f"Accuracy: {avg_course_accuracy * 100:.2f}%")
    print(f"Top-3 Accuracy: {avg_course_top3_accuracy * 100:.2f}%")
    print(f"F1-Score: {avg_course_f1 * 100:.2f}%")
    print(f"Mean Average Precision (mAP): {avg_course_map * 100:.2f}%")

if __name__ == "__main__":
    ensemble_metrics()
