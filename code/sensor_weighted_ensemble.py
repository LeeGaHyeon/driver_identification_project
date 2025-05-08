# import pandas as pd
# from sklearn.metrics import f1_score
# import itertools

# def evaluate_ensemble(sensor_weight, can_weight, course_dfs, label_columns):
#     total_accuracy = 0
#     total_top3_accuracy = 0
#     total_f1 = 0
#     total_map = 0
    
#     course_performances = []
    
#     for course_df in course_dfs:
#         merged_df = course_df['df']
#         val_col = course_df['val_col']  # 각 회차 또는 코스의 'val' 컬럼 이름 가져오기
        
#         # Weighted average probabilities (Sensor + CAN)
#         for label in label_columns:
#             if f'{label}_sensor' in merged_df.columns and f'{label}_can' in merged_df.columns:
#                 merged_df[label + '_weighted_avg'] = (
#                     merged_df[label + '_sensor'] * sensor_weight + 
#                     merged_df[label + '_can'] * can_weight
#                 )

#         # Predicted label
#         weighted_avg_columns = [label + '_weighted_avg' for label in label_columns if label + '_weighted_avg' in merged_df.columns]
#         if weighted_avg_columns:
#             merged_df['predicted_label'] = merged_df[weighted_avg_columns].idxmax(axis=1)
#             merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_weighted_avg', '').astype(int)
#         else:
#             print(f"No weighted columns found for {course_df['name']}")

#         # Accuracy
#         merged_df['correct'] = merged_df[val_col] == merged_df['predicted_label']
#         accuracy = merged_df['correct'].mean()

#         # Top-3 accuracy
#         merged_df['top_3_labels'] = merged_df[weighted_avg_columns].apply(
#             lambda row: row.nlargest(3).index.str.replace('_weighted_avg', '').astype(int).tolist(), axis=1)
#         merged_df['top_3_correct'] = merged_df.apply(lambda row: row[val_col] in row['top_3_labels'], axis=1)
#         top3_accuracy = merged_df['top_3_correct'].mean()

#         # F1-Score
#         f1 = f1_score(merged_df[val_col], merged_df['predicted_label'], average='macro')

#         # Mean Average Precision (mAP)
#         def average_precision_at_k(row, k):
#             top_k_labels = row['top_3_labels'][:k]
#             try:
#                 rank = top_k_labels.index(row[val_col]) + 1  # Find the rank (1-based index)
#                 return 1.0 / rank  # Precision at the rank
#             except ValueError:
#                 return 0.0  # If the true label is not in the top_k, return 0.0

#         merged_df['ap_at_3'] = merged_df.apply(lambda row: average_precision_at_k(row, 3), axis=1)
#         map_score = merged_df['ap_at_3'].mean()
        
#         course_performances.append({
#             'course': course_df['name'],
#             'accuracy': accuracy,
#             'top3_accuracy': top3_accuracy,
#             'f1': f1,
#             'map': map_score
#         })
        
#         total_accuracy += accuracy
#         total_top3_accuracy += top3_accuracy
#         total_f1 += f1
#         total_map += map_score
    
#     # 평균 계산
#     num_courses = len(course_dfs)
#     avg_accuracy = total_accuracy / num_courses
#     avg_top3_accuracy = total_top3_accuracy / num_courses
#     avg_f1 = total_f1 / num_courses
#     avg_map = total_map / num_courses

#     return avg_accuracy, avg_top3_accuracy, avg_f1, avg_map, course_performances

# def find_best_tradeoff_weights_and_evaluate():
#     # 회차 데이터 로드
#     round_dfs = [
#         {'name': '1회차', 'val_col': 'val_1', 'df': pd.read_csv('./prob/sensor_round_1_avg_prob.csv').merge(
#             pd.read_csv('./prob/can_round_1_avg_prob.csv'), on='val_1', suffixes=('_sensor', '_can'))},
        
#         {'name': '2회차', 'val_col': 'val_2', 'df': pd.read_csv('./prob/sensor_round_2_avg_prob.csv').merge(
#             pd.read_csv('./prob/can_round_2_avg_prob.csv'), on='val_2', suffixes=('_sensor', '_can'))},
        
#         {'name': '3회차', 'val_col': 'val_3', 'df': pd.read_csv('./prob/sensor_round_3_avg_prob.csv').merge(
#             pd.read_csv('./prob/can_round_3_avg_prob.csv'), on='val_3', suffixes=('_sensor', '_can'))},
        
#         {'name': '4회차', 'val_col': 'val_4', 'df': pd.read_csv('./prob/sensor_round_4_avg_prob.csv').merge(
#             pd.read_csv('./prob/can_round_4_avg_prob.csv'), on='val_4', suffixes=('_sensor', '_can'))}
#     ]

#     # 코스 데이터 로드
#     course_dfs = [
#         {'name': 'A 코스', 'val_col': 'val_A', 'df': pd.read_csv('./prob/sensor_course_A_avg_prob.csv').merge(
#             pd.read_csv('./prob/can_course_A_avg_prob.csv'), on='val_A', suffixes=('_sensor', '_can'))},
        
#         {'name': 'B 코스', 'val_col': 'val_B', 'df': pd.read_csv('./prob/sensor_course_B_avg_prob.csv').merge(
#             pd.read_csv('./prob/can_course_B_avg_prob.csv'), on='val_B', suffixes=('_sensor', '_can'))},
        
#         {'name': 'C 코스', 'val_col': 'val_C', 'df': pd.read_csv('./prob/sensor_course_C_avg_prob.csv').merge(
#             pd.read_csv('./prob/can_course_C_avg_prob.csv'), on='val_C', suffixes=('_sensor', '_can'))}
#     ]
    
#     label_columns = [str(i) for i in range(15)]
    
#     # 최적의 가중치 찾기
#     best_weights = None
#     best_tradeoff_performance = -1
#     best_tradeoff_round_performances = None
#     best_tradeoff_course_performances = None
    
#     # sensor_weight와 can_weight의 조합을 테스트 (가중치 합은 1이 되도록 설정)
#     for sensor_weight, can_weight in itertools.product([x / 20.0 for x in range(21)], repeat=2):
#         if abs(sensor_weight + can_weight - 1.0) < 1e-6:  # 가중치의 합이 1인 경우만
            
#             # 회차별 평가
#             avg_round_accuracy, avg_round_top3_accuracy, avg_round_f1, avg_round_map, round_performances = evaluate_ensemble(
#                 sensor_weight, can_weight, round_dfs, label_columns)
            
#             round_performance = (avg_round_accuracy + avg_round_top3_accuracy + avg_round_f1 + avg_round_map) / 4
            
#             # 코스별 평가
#             avg_course_accuracy, avg_course_top3_accuracy, avg_course_f1, avg_course_map, course_performances = evaluate_ensemble(
#                 sensor_weight, can_weight, course_dfs, label_columns)
            
#             course_performance = (avg_course_accuracy + avg_course_top3_accuracy + avg_course_f1 + avg_course_map) / 4
            
#             # 회차별 성능과 코스별 성능 간의 평균 성능을 구하여 Trade-off 성능을 계산
#             tradeoff_performance = (round_performance + course_performance) / 2
            
#             # 최적의 Trade-off 성능 찾기
#             if tradeoff_performance > best_tradeoff_performance:
#                 best_tradeoff_performance = tradeoff_performance
#                 best_weights = (sensor_weight, can_weight)
#                 best_tradeoff_round_performances = round_performances
#                 best_tradeoff_course_performances = course_performances
    
#     # 최적 가중치 출력
#     print(f"Best Trade-off Weights: sensor={best_weights[0]}, can={best_weights[1]}")
#     print(f"Best Trade-off Performance (Average of Rounds and Courses): {best_tradeoff_performance:.4f}")
    
#     # 최적 Trade-off 성능 (회차별)
#     if best_tradeoff_round_performances:
#         print("\nRound Performances:")
#         for round_perf in best_tradeoff_round_performances:
#             print(f"Round {round_perf['course']} - Accuracy: {round_perf['accuracy']:.4f}, "
#                   f"Top-3 Accuracy: {round_perf['top3_accuracy']:.4f}, F1-Score: {round_perf['f1']:.4f}, "
#                   f"mAP: {round_perf['map']:.4f}")
    
#     # 최적 Trade-off 성능 (코스별)
#     if best_tradeoff_course_performances:
#         print("\nCourse Performances:")
#         for course_perf in best_tradeoff_course_performances:
#             print(f"Course {course_perf['course']} - Accuracy: {course_perf['accuracy']:.4f}, "
#                   f"Top-3 Accuracy: {course_perf['top3_accuracy']:.4f}, F1-Score: {course_perf['f1']:.4f}, "
#                   f"mAP: {course_perf['map']:.4f}")

# if __name__ == "__main__":
#     find_best_tradeoff_weights_and_evaluate()

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve
from sklearn.preprocessing import label_binarize
import itertools
import numpy as np

def evaluate_ensemble(sensor_weight, can_weight, course_dfs, label_columns):
    total_accuracy = 0
    total_top3_accuracy = 0
    total_f1 = 0
    total_map = 0
    
    course_performances = []
    
    for course_df in course_dfs:
        merged_df = course_df['df']
        val_col = course_df['val_col']  # 각 회차 또는 코스의 'val' 컬럼 이름 가져오기
        
        # Weighted average probabilities (Sensor + CAN)
        for label in label_columns:
            if f'{label}_sensor' in merged_df.columns and f'{label}_can' in merged_df.columns:
                merged_df[label + '_weighted_avg'] = (
                    merged_df[label + '_sensor'] * sensor_weight + 
                    merged_df[label + '_can'] * can_weight
                )

        # Predicted label
        weighted_avg_columns = [label + '_weighted_avg' for label in label_columns if label + '_weighted_avg' in merged_df.columns]
        if weighted_avg_columns:
            merged_df['predicted_label'] = merged_df[weighted_avg_columns].idxmax(axis=1)
            merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_weighted_avg', '').astype(int)
        else:
            print(f"No weighted columns found for {course_df['name']}")

        # Accuracy
        merged_df['correct'] = merged_df[val_col] == merged_df['predicted_label']
        accuracy = merged_df['correct'].mean()

        # Top-3 accuracy
        merged_df['top_3_labels'] = merged_df[weighted_avg_columns].apply(
            lambda row: row.nlargest(3).index.str.replace('_weighted_avg', '').astype(int).tolist(), axis=1)
        merged_df['top_3_correct'] = merged_df.apply(lambda row: row[val_col] in row['top_3_labels'], axis=1)
        top3_accuracy = merged_df['top_3_correct'].mean()

        # F1-Score
        f1 = f1_score(merged_df[val_col], merged_df['predicted_label'], average='macro')

        # Mean Average Precision (mAP)
        y_true_bin = label_binarize(merged_df[val_col], classes=[int(label) for label in label_columns])
        y_scores = merged_df[weighted_avg_columns].values
        average_precisions = []
        for i in range(len(label_columns)):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            ap = np.trapz(recall, precision)
            average_precisions.append(ap)
        map_score = np.mean(average_precisions)
        
        course_performances.append({
            'course': course_df['name'],
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'f1': f1,
            'map': map_score
        })
        
        total_accuracy += accuracy
        total_top3_accuracy += top3_accuracy
        total_f1 += f1
        total_map += map_score
    
    # 평균 계산
    num_courses = len(course_dfs)
    avg_accuracy = total_accuracy / num_courses
    avg_top3_accuracy = total_top3_accuracy / num_courses
    avg_f1 = total_f1 / num_courses
    avg_map = total_map / num_courses

    return avg_accuracy, avg_top3_accuracy, avg_f1, avg_map, course_performances

def find_best_tradeoff_weights_and_evaluate():
    # 회차 데이터 로드
    round_dfs = [
        {'name': '1회차', 'val_col': 'val_1', 'df': pd.read_csv('./prob/sensor_round_1_avg_prob.csv').merge(
            pd.read_csv('./prob/can_round_1_avg_prob.csv'), on='val_1', suffixes=('_sensor', '_can'))},
        
        {'name': '2회차', 'val_col': 'val_2', 'df': pd.read_csv('./prob/sensor_round_2_avg_prob.csv').merge(
            pd.read_csv('./prob/can_round_2_avg_prob.csv'), on='val_2', suffixes=('_sensor', '_can'))},
        
        {'name': '3회차', 'val_col': 'val_3', 'df': pd.read_csv('./prob/sensor_round_3_avg_prob.csv').merge(
            pd.read_csv('./prob/can_round_3_avg_prob.csv'), on='val_3', suffixes=('_sensor', '_can'))},
        
        {'name': '4회차', 'val_col': 'val_4', 'df': pd.read_csv('./prob/sensor_round_4_avg_prob.csv').merge(
            pd.read_csv('./prob/can_round_4_avg_prob.csv'), on='val_4', suffixes=('_sensor', '_can'))}
    ]

    # 코스 데이터 로드
    course_dfs = [
        {'name': 'A 코스', 'val_col': 'val_A', 'df': pd.read_csv('./prob/sensor_course_A_avg_prob.csv').merge(
            pd.read_csv('./prob/can_course_A_avg_prob.csv'), on='val_A', suffixes=('_sensor', '_can'))},
        
        {'name': 'B 코스', 'val_col': 'val_B', 'df': pd.read_csv('./prob/sensor_course_B_avg_prob.csv').merge(
            pd.read_csv('./prob/can_course_B_avg_prob.csv'), on='val_B', suffixes=('_sensor', '_can'))},
        
        {'name': 'C 코스', 'val_col': 'val_C', 'df': pd.read_csv('./prob/sensor_course_C_avg_prob.csv').merge(
            pd.read_csv('./prob/can_course_C_avg_prob.csv'), on='val_C', suffixes=('_sensor', '_can'))}
    ]
    
    label_columns = [str(i) for i in range(15)]
    
    # 최적의 가중치 찾기
    best_weights = None
    best_tradeoff_performance = -1
    best_tradeoff_round_performances = None
    best_tradeoff_course_performances = None
    
    # sensor_weight와 can_weight의 조합을 테스트 (가중치 합은 1이 되도록 설정)
    for sensor_weight, can_weight in itertools.product([x / 20.0 for x in range(21)], repeat=2):
        if abs(sensor_weight + can_weight - 1.0) < 1e-6:  # 가중치의 합이 1인 경우만
            
            # 회차별 평가
            avg_round_accuracy, avg_round_top3_accuracy, avg_round_f1, avg_round_map, round_performances = evaluate_ensemble(
                sensor_weight, can_weight, round_dfs, label_columns)
            
            round_performance = (avg_round_accuracy + avg_round_top3_accuracy + avg_round_f1 + avg_round_map) / 4
            
            # 코스별 평가
            avg_course_accuracy, avg_course_top3_accuracy, avg_course_f1, avg_course_map, course_performances = evaluate_ensemble(
                sensor_weight, can_weight, course_dfs, label_columns)
            
            course_performance = (avg_course_accuracy + avg_course_top3_accuracy + avg_course_f1 + avg_course_map) / 4
            
            # 회차별 성능과 코스별 성능 간의 평균 성능을 구하여 Trade-off 성능을 계산
            tradeoff_performance = (round_performance + course_performance) / 2
            
            # 최적의 Trade-off 성능 찾기
            if tradeoff_performance > best_tradeoff_performance:
                best_tradeoff_performance = tradeoff_performance
                best_weights = (sensor_weight, can_weight)
                best_tradeoff_round_performances = round_performances
                best_tradeoff_course_performances = course_performances
    
    # 최적 가중치 출력
    print(f"Best Trade-off Weights: sensor={best_weights[0]}, can={best_weights[1]}")
    print(f"Best Trade-off Performance (Average of Rounds and Courses): {best_tradeoff_performance:.4f}")
    
    # 최적 Trade-off 성능 (회차별)
    if best_tradeoff_round_performances:
        print("\nRound Performances:")
        for round_perf in best_tradeoff_round_performances:
            print(f"Round {round_perf['course']} - Accuracy: {round_perf['accuracy']:.4f}, "
                  f"Top-3 Accuracy: {round_perf['top3_accuracy']:.4f}, F1-Score: {round_perf['f1']:.4f}, "
                  f"mAP: {round_perf['map']:.4f}")
    
    # 최적 Trade-off 성능 (코스별)
    if best_tradeoff_course_performances:
        print("\nCourse Performances:")
        for course_perf in best_tradeoff_course_performances:
            print(f"Course {course_perf['course']} - Accuracy: {course_perf['accuracy']:.4f}, "
                  f"Top-3 Accuracy: {course_perf['top3_accuracy']:.4f}, F1-Score: {course_perf['f1']:.4f}, "
                  f"mAP: {course_perf['map']:.4f}")

if __name__ == "__main__":
    find_best_tradeoff_weights_and_evaluate()

