import os

def count_data_per_round(base_dir, shuffle_dir):
    optical_dir = os.path.join(base_dir, 'optical_window30')
    visual_dir = os.path.join(base_dir, 'visual_window30')

    shuffle_optical_dir = os.path.join(shuffle_dir, 'optical_window30')
    shuffle_visual_dir = os.path.join(shuffle_dir, 'visual_window30')

    total_optical_count_base = 0
    total_visual_count_base = 0
    total_optical_count_shuffle = 0
    total_visual_count_shuffle = 0
    mismatch_found = False

    # 각 폴더에서 데이터 수 확인
    def count_files_in_round(optical_round_path, visual_round_path):
        optical_files = os.listdir(optical_round_path)
        visual_files = os.listdir(visual_round_path)
        return len(optical_files), len(visual_files)

    for person_id in os.listdir(optical_dir):
        optical_person_path = os.path.join(optical_dir, person_id)
        visual_person_path = os.path.join(visual_dir, person_id)
        shuffle_optical_person_path = os.path.join(shuffle_optical_dir, person_id)
        shuffle_visual_person_path = os.path.join(shuffle_visual_dir, person_id)

        if os.path.isdir(optical_person_path) and os.path.isdir(visual_person_path) and \
           os.path.isdir(shuffle_optical_person_path) and os.path.isdir(shuffle_visual_person_path):
            for section in ['A', 'B', 'C']:
                optical_section_path = os.path.join(optical_person_path, section)
                visual_section_path = os.path.join(visual_person_path, section)
                shuffle_optical_section_path = os.path.join(shuffle_optical_person_path, section)
                shuffle_visual_section_path = os.path.join(shuffle_visual_person_path, section)

                if os.path.isdir(optical_section_path) and os.path.isdir(visual_section_path) and \
                   os.path.isdir(shuffle_optical_section_path) and os.path.isdir(shuffle_visual_section_path):
                    for scenario in ['bump', 'corner']:
                        optical_scenario_path = os.path.join(optical_section_path, scenario)
                        visual_scenario_path = os.path.join(visual_section_path, scenario)
                        shuffle_optical_scenario_path = os.path.join(shuffle_optical_section_path, scenario)
                        shuffle_visual_scenario_path = os.path.join(shuffle_visual_section_path, scenario)

                        if os.path.isdir(optical_scenario_path) and os.path.isdir(visual_scenario_path) and \
                           os.path.isdir(shuffle_optical_scenario_path) and os.path.isdir(shuffle_visual_scenario_path):
                            for round_folder in os.listdir(optical_scenario_path):
                                optical_round_path = os.path.join(optical_scenario_path, round_folder)
                                visual_round_path = os.path.join(visual_scenario_path, round_folder)
                                shuffle_optical_round_path = os.path.join(shuffle_optical_scenario_path, round_folder)
                                shuffle_visual_round_path = os.path.join(shuffle_visual_scenario_path, round_folder)

                                if os.path.isdir(optical_round_path) and os.path.isdir(visual_round_path) and \
                                   os.path.isdir(shuffle_optical_round_path) and os.path.isdir(shuffle_visual_round_path):

                                    # base 폴더와 shuffle 폴더에서 데이터 수 계산
                                    base_optical_count, base_visual_count = count_files_in_round(optical_round_path, visual_round_path)
                                    shuffle_optical_count, shuffle_visual_count = count_files_in_round(shuffle_optical_round_path, shuffle_visual_round_path)

                                    # 총 갯수 카운트
                                    total_optical_count_base += base_optical_count
                                    total_visual_count_base += base_visual_count
                                    total_optical_count_shuffle += shuffle_optical_count
                                    total_visual_count_shuffle += shuffle_visual_count

                                    # 데이터 갯수가 다를 경우 출력
                                    if base_optical_count != shuffle_optical_count or base_visual_count != shuffle_visual_count:
                                        mismatch_found = True
                                        print(f"Mismatch found in Person: {person_id}, Section: {section}, Scenario: {scenario}, Round: {round_folder}")
                                        print(f"  Base Optical count: {base_optical_count}, Shuffle Optical count: {shuffle_optical_count}")
                                        print(f"  Base Visual count: {base_visual_count}, Shuffle Visual count: {shuffle_visual_count}")
                                        print(f"  Base Optical Path: {optical_round_path}")
                                        print(f"  Base Visual Path: {visual_round_path}")
                                        print(f"  Shuffle Optical Path: {shuffle_optical_round_path}")
                                        print(f"  Shuffle Visual Path: {shuffle_visual_round_path}\n")

    if not mismatch_found:
        print("No mismatch found between base and shuffle data.")

    # 총 갯수 출력
    print(f"Total Visual Data Count (Base): {total_visual_count_base}")
    print(f"Total Optical Data Count (Base): {total_optical_count_base}")
    print(f"Total Visual Data Count (Shuffle): {total_visual_count_shuffle}")
    print(f"Total Optical Data Count (Shuffle): {total_optical_count_shuffle}")

# Usage
base_dir = "./images/"
shuffle_dir = "./images_shuffle/"
count_data_per_round(base_dir, shuffle_dir)
