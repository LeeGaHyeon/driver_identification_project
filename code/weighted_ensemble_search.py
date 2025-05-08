# 코스

# import pandas as pd
# from sklearn.metrics import f1_score
# import itertools

# def evaluate_ensemble(vi_weight, sensor_weight, can_weight, course_dfs, label_columns):
#     total_accuracy = 0
#     total_top3_accuracy = 0
#     total_f1 = 0
#     total_map = 0
    
#     course_performances = []
    
#     for course_df in course_dfs:
#         merged_df = course_df['df']
#         val_col = course_df['val_col']  # 각 코스의 'val' 컬럼 이름 가져오기
        
#         # Weighted average probabilities
#         for label in label_columns:
#             merged_df[label + '_weighted_avg'] = (
#                 merged_df[label + '_vi'] * vi_weight + 
#                 merged_df[label + '_sensor'] * sensor_weight + 
#                 merged_df[label] * can_weight
#             )

#         # Predicted label
#         merged_df['predicted_label'] = merged_df[[label + '_weighted_avg' for label in label_columns]].idxmax(axis=1)
#         merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_weighted_avg', '').astype(int)

#         # Accuracy
#         merged_df['correct'] = merged_df[val_col] == merged_df['predicted_label']
#         accuracy = merged_df['correct'].mean()

#         # Top-3 accuracy
#         merged_df['top_3_labels'] = merged_df[[label + '_weighted_avg' for label in label_columns]].apply(
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


# import pandas as pd
# from sklearn.metrics import f1_score
# import itertools

# def evaluate_simple_ensemble(ensemble_cols, course_dfs, label_columns):
#     course_performances = []
#     total_accuracy = 0
#     total_top3_accuracy = 0
#     total_f1 = 0
#     total_map = 0
    
#     for course_df in course_dfs:
#         merged_df = course_df['df']
#         val_col = course_df['val_col']  # 각 코스의 'val' 컬럼 이름 가져오기
        
#         # Simple average of the specified ensemble columns
#         for label in label_columns:
#             merged_df[label + '_avg'] = merged_df[[label + col_suffix for col_suffix in ensemble_cols]].mean(axis=1)

#         # Predicted label
#         merged_df['predicted_label'] = merged_df[[label + '_avg' for label in label_columns]].idxmax(axis=1)
#         merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_avg', '').astype(int)

#         # Accuracy
#         merged_df['correct'] = merged_df[val_col] == merged_df['predicted_label']
#         accuracy = merged_df['correct'].mean()
#         total_accuracy += accuracy

#         # Top-3 accuracy
#         merged_df['top_3_labels'] = merged_df[[label + '_avg' for label in label_columns]].apply(
#             lambda row: row.nlargest(3).index.str.replace('_avg', '').astype(int).tolist(), axis=1)
#         merged_df['top_3_correct'] = merged_df.apply(lambda row: row[val_col] in row['top_3_labels'], axis=1)
#         top3_accuracy = merged_df['top_3_correct'].mean()
#         total_top3_accuracy += top3_accuracy

#         # F1-Score
#         f1 = f1_score(merged_df[val_col], merged_df['predicted_label'], average='macro')
#         total_f1 += f1

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
#         total_map += map_score
        
#         course_performances.append({
#             'course': course_df['name'],
#             'accuracy': accuracy,
#             'top3_accuracy': top3_accuracy,
#             'f1': f1,
#             'map': map_score
#         })
    
#     num_courses = len(course_dfs)
#     avg_accuracy = total_accuracy / num_courses
#     avg_top3_accuracy = total_top3_accuracy / num_courses
#     avg_f1 = total_f1 / num_courses
#     avg_map = total_map / num_courses

#     return course_performances, avg_accuracy, avg_top3_accuracy, avg_f1, avg_map

# def find_best_weights_and_simple_ensembles():
#     # A, B, C 코스 데이터 로드
#     course_dfs = [
#         {'name': 'A', 'val_col': 'val_A', 'df': pd.read_csv('./prob/0902_AugAug2Cross_w30_logits_A.csv').merge(
#             pd.read_csv('./prob/sensor_course_A_avg_prob.csv'), on='val_A', suffixes=('_vi', '_sensor')).merge(
#             pd.read_csv('./prob/can_course_A_avg_prob.csv'), on='val_A', suffixes=('', '_can'))},
        
#         {'name': 'B', 'val_col': 'val_B', 'df': pd.read_csv('./prob/0902_AugAug2Cross_w30_logits_B.csv').merge(
#             pd.read_csv('./prob/sensor_course_B_avg_prob.csv'), on='val_B', suffixes=('_vi', '_sensor')).merge(
#             pd.read_csv('./prob/can_course_B_avg_prob.csv'), on='val_B', suffixes=('', '_can'))},
        
#         {'name': 'C', 'val_col': 'val_C', 'df': pd.read_csv('./prob/0902_AugAug2Cross_w30_logits_C.csv').merge(
#             pd.read_csv('./prob/sensor_course_C_avg_prob.csv'), on='val_C', suffixes=('_vi', '_sensor')).merge(
#             pd.read_csv('./prob/can_course_C_avg_prob.csv'), on='val_C', suffixes=('', '_can'))}
#     ]
    
#     label_columns = [str(i) for i in range(15)]
    
#     # 최적의 가중치 찾기
#     best_weights = None
#     best_performance = -1
#     best_course_performances = None
    
#     for vi_weight, sensor_weight, can_weight in itertools.product([x / 20.0 for x in range(21)], repeat=3):
#         if abs(vi_weight + sensor_weight + can_weight - 1.0) < 1e-6:  # 가중치의 합이 1인 경우만
#             avg_accuracy, avg_top3_accuracy, avg_f1, avg_map, course_performances = evaluate_ensemble(
#                 vi_weight, sensor_weight, can_weight, course_dfs, label_columns)
            
#             performance = (avg_accuracy + avg_top3_accuracy + avg_f1 + avg_map) / 4  # 성능 평균
            
#             if performance > best_performance:
#                 best_performance = performance
#                 best_weights = (vi_weight, sensor_weight, can_weight)
#                 best_course_performances = course_performances
    
#     print(f"Best Weights: vi={best_weights[0]}, sensor={best_weights[1]}, can={best_weights[2]}")
#     print(f"Best Performance (Average of Accuracy, Top-3 Accuracy, F1-Score, mAP): {best_performance:.4f}")
    
#     # 최적 가중치 조합으로 코스별 성능 출력
#     if best_course_performances:
#         for course_perf in best_course_performances:
#             print(f"Course {course_perf['course']} - Accuracy: {course_perf['accuracy']:.4f}, "
#                   f"Top-3 Accuracy: {course_perf['top3_accuracy']:.4f}, F1-Score: {course_perf['f1']:.4f}, "
#                   f"mAP: {course_perf['map']:.4f}")
    
#     # Simple ensemble 조합별 성능 평가 및 출력
#     simple_ensembles = {
#         "vi+can": ["_vi", ""],
#         "vi+sensor": ["_vi", "_sensor"],
#         "sensor+can": ["_sensor", ""]
#     }
    
#     for ensemble_name, ensemble_cols in simple_ensembles.items():
#         print(f"\nEvaluating Simple Ensemble: {ensemble_name}")
#         course_performances, avg_accuracy, avg_top3_accuracy, avg_f1, avg_map = evaluate_simple_ensemble(
#             ensemble_cols, course_dfs, label_columns)
        
#         for course_perf in course_performances:
#             print(f"Course {course_perf['course']} - Accuracy: {course_perf['accuracy']:.4f}, "
#                   f"Top-3 Accuracy: {course_perf['top3_accuracy']:.4f}, F1-Score: {course_perf['f1']:.4f}, "
#                   f"mAP: {course_perf['map']:.4f}")
        
#         print(f"\nAverage Performance for {ensemble_name} across all courses - "
#               f"Accuracy: {avg_accuracy:.4f}, Top-3 Accuracy: {avg_top3_accuracy:.4f}, "
#               f"F1-Score: {avg_f1:.4f}, mAP: {avg_map:.4f}")

# if __name__ == "__main__":
#     find_best_weights_and_simple_ensembles()

# 회차
import pandas as pd
from sklearn.metrics import f1_score
import itertools

def evaluate_ensemble(vi_weight, sensor_weight, can_weight, course_dfs, label_columns):
    total_accuracy = 0
    total_top3_accuracy = 0
    total_f1 = 0
    total_map = 0
    
    course_performances = []
    
    for course_df in course_dfs:
        merged_df = course_df['df']
        val_col = course_df['val_col']  # 각 회차의 'val' 컬럼 이름 가져오기
        
        # Weighted average probabilities
        for label in label_columns:
            merged_df[label + '_weighted_avg'] = (
                merged_df[label + '_vi'] * vi_weight + 
                merged_df[label + '_sensor'] * sensor_weight + 
                merged_df[label] * can_weight
            )

        # Predicted label
        merged_df['predicted_label'] = merged_df[[label + '_weighted_avg' for label in label_columns]].idxmax(axis=1)
        merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_weighted_avg', '').astype(int)

        # Accuracy
        merged_df['correct'] = merged_df[val_col] == merged_df['predicted_label']
        accuracy = merged_df['correct'].mean()

        # Top-3 accuracy
        merged_df['top_3_labels'] = merged_df[[label + '_weighted_avg' for label in label_columns]].apply(
            lambda row: row.nlargest(3).index.str.replace('_weighted_avg', '').astype(int).tolist(), axis=1)
        merged_df['top_3_correct'] = merged_df.apply(lambda row: row[val_col] in row['top_3_labels'], axis=1)
        top3_accuracy = merged_df['top_3_correct'].mean()

        # F1-Score
        f1 = f1_score(merged_df[val_col], merged_df['predicted_label'], average='macro')

        # Mean Average Precision (mAP)
        def average_precision_at_k(row, k):
            top_k_labels = row['top_3_labels'][:k]
            try:
                rank = top_k_labels.index(row[val_col]) + 1  # Find the rank (1-based index)
                return 1.0 / rank  # Precision at the rank
            except ValueError:
                return 0.0  # If the true label is not in the top_k, return 0.0

        merged_df['ap_at_3'] = merged_df.apply(lambda row: average_precision_at_k(row, 3), axis=1)
        map_score = merged_df['ap_at_3'].mean()
        
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

def evaluate_simple_ensemble(ensemble_cols, course_dfs, label_columns):
    course_performances = []
    total_accuracy = 0
    total_top3_accuracy = 0
    total_f1 = 0
    total_map = 0
    
    for course_df in course_dfs:
        merged_df = course_df['df']
        val_col = course_df['val_col']  # 각 회차의 'val' 컬럼 이름 가져오기
        
        # Simple average of the specified ensemble columns
        for label in label_columns:
            merged_df[label + '_avg'] = merged_df[[label + col_suffix for col_suffix in ensemble_cols]].mean(axis=1)

        # Predicted label
        merged_df['predicted_label'] = merged_df[[label + '_avg' for label in label_columns]].idxmax(axis=1)
        merged_df['predicted_label'] = merged_df['predicted_label'].str.replace('_avg', '').astype(int)

        # Accuracy
        merged_df['correct'] = merged_df[val_col] == merged_df['predicted_label']
        accuracy = merged_df['correct'].mean()
        total_accuracy += accuracy

        # Top-3 accuracy
        merged_df['top_3_labels'] = merged_df[[label + '_avg' for label in label_columns]].apply(
            lambda row: row.nlargest(3).index.str.replace('_avg', '').astype(int).tolist(), axis=1)
        merged_df['top_3_correct'] = merged_df.apply(lambda row: row[val_col] in row['top_3_labels'], axis=1)
        top3_accuracy = merged_df['top_3_correct'].mean()
        total_top3_accuracy += top3_accuracy

        # F1-Score
        f1 = f1_score(merged_df[val_col], merged_df['predicted_label'], average='macro')
        total_f1 += f1

        # Mean Average Precision (mAP)
        def average_precision_at_k(row, k):
            top_k_labels = row['top_3_labels'][:k]
            try:
                rank = top_k_labels.index(row[val_col]) + 1  # Find the rank (1-based index)
                return 1.0 / rank  # Precision at the rank
            except ValueError:
                return 0.0  # If the true label is not in the top_k, return 0.0

        merged_df['ap_at_3'] = merged_df.apply(lambda row: average_precision_at_k(row, 3), axis=1)
        map_score = merged_df['ap_at_3'].mean()
        total_map += map_score
        
        course_performances.append({
            'course': course_df['name'],
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'f1': f1,
            'map': map_score
        })
    
    num_courses = len(course_dfs)
    avg_accuracy = total_accuracy / num_courses
    avg_top3_accuracy = total_top3_accuracy / num_courses
    avg_f1 = total_f1 / num_courses
    avg_map = total_map / num_courses

    return course_performances, avg_accuracy, avg_top3_accuracy, avg_f1, avg_map

def find_best_weights_and_simple_ensembles():
    # 1, 2, 3, 4 회차 데이터 로드
    course_dfs = [
        {'name': '1회차', 'val_col': 'val_1', 'df': pd.read_csv('./prob/0902_AugAug2Cross_w30_logits_1.csv').merge(
            pd.read_csv('./prob/sensor_round_1_avg_prob.csv'), on='val_1', suffixes=('_vi', '_sensor')).merge(
            pd.read_csv('./prob/can_round_1_avg_prob.csv'), on='val_1', suffixes=('', '_can'))},
        
        {'name': '2회차', 'val_col': 'val_2', 'df': pd.read_csv('./prob/0902_AugAug2Cross_w30_logits_2.csv').merge(
            pd.read_csv('./prob/sensor_round_2_avg_prob.csv'), on='val_2', suffixes=('_vi', '_sensor')).merge(
            pd.read_csv('./prob/can_round_2_avg_prob.csv'), on='val_2', suffixes=('', '_can'))},
        
        {'name': '3회차', 'val_col': 'val_3', 'df': pd.read_csv('./prob/0902_AugAug2Cross_w30_logits_3.csv').merge(
            pd.read_csv('./prob/sensor_round_3_avg_prob.csv'), on='val_3', suffixes=('_vi', '_sensor')).merge(
            pd.read_csv('./prob/can_round_3_avg_prob.csv'), on='val_3', suffixes=('', '_can'))},
        
        {'name': '4회차', 'val_col': 'val_4', 'df': pd.read_csv('./prob/0902_AugAug2Cross_w30_logits_4.csv').merge(
            pd.read_csv('./prob/sensor_round_4_avg_prob.csv'), on='val_4', suffixes=('_vi', '_sensor')).merge(
            pd.read_csv('./prob/can_round_4_avg_prob.csv'), on='val_4', suffixes=('', '_can'))}
    ]
    
    label_columns = [str(i) for i in range(15)]
    
    # 최적의 가중치 찾기
    best_weights = None
    best_performance = -1
    best_course_performances = None
    
    for vi_weight, sensor_weight, can_weight in itertools.product([x / 20.0 for x in range(21)], repeat=3):
        if abs(vi_weight + sensor_weight + can_weight - 1.0) < 1e-6:  # 가중치의 합이 1인 경우만
            avg_accuracy, avg_top3_accuracy, avg_f1, avg_map, course_performances = evaluate_ensemble(
                vi_weight, sensor_weight, can_weight, course_dfs, label_columns)
            
            performance = (avg_accuracy + avg_top3_accuracy + avg_f1 + avg_map) / 4  # 성능 평균
            
            if performance > best_performance:
                best_performance = performance
                best_weights = (vi_weight, sensor_weight, can_weight)
                best_course_performances = course_performances
    
    print(f"Best Weights: vi={best_weights[0]}, sensor={best_weights[1]}, can={best_weights[2]}")
    print(f"Best Performance (Average of Accuracy, Top-3 Accuracy, F1-Score, mAP): {best_performance:.4f}")
    
    # 최적 가중치 조합으로 회차별 성능 출력
    if best_course_performances:
        for course_perf in best_course_performances:
            print(f"Round {course_perf['course']} - Accuracy: {course_perf['accuracy']:.4f}, "
                  f"Top-3 Accuracy: {course_perf['top3_accuracy']:.4f}, F1-Score: {course_perf['f1']:.4f}, "
                  f"mAP: {course_perf['map']:.4f}")
    
    # Simple ensemble 조합별 성능 평가 및 출력
    simple_ensembles = {
        "vi+can": ["_vi", ""],
        "vi+sensor": ["_vi", "_sensor"],
        "sensor+can": ["_sensor", ""]
    }
    
    for ensemble_name, ensemble_cols in simple_ensembles.items():
        print(f"\nEvaluating Simple Ensemble: {ensemble_name}")
        course_performances, avg_accuracy, avg_top3_accuracy, avg_f1, avg_map = evaluate_simple_ensemble(
            ensemble_cols, course_dfs, label_columns)
        
        for course_perf in course_performances:
            print(f"Round {course_perf['course']} - Accuracy: {course_perf['accuracy']:.4f}, "
                  f"Top-3 Accuracy: {course_perf['top3_accuracy']:.4f}, F1-Score: {course_perf['f1']:.4f}, "
                  f"mAP: {course_perf['map']:.4f}")
        
        print(f"\nAverage Performance for {ensemble_name} across all rounds - "
              f"Accuracy: {avg_accuracy:.4f}, Top-3 Accuracy: {avg_top3_accuracy:.4f}, "
              f"F1-Score: {avg_f1:.4f}, mAP: {avg_map:.4f}")

if __name__ == "__main__":
    find_best_weights_and_simple_ensembles()
