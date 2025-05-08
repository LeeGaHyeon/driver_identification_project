import os

def count_data_per_round(base_dir, output_file):
    optical_dir = os.path.join(base_dir, 'optical_window30')
    visual_dir = os.path.join(base_dir, 'visual_window30')
    
    with open(output_file, 'w') as f:
        for person_id in os.listdir(optical_dir):
            optical_person_path = os.path.join(optical_dir, person_id)
            visual_person_path = os.path.join(visual_dir, person_id)
            
            if os.path.isdir(optical_person_path) and os.path.isdir(visual_person_path):
                f.write(f"Person ID: {person_id}\n")
                for section in ['A', 'B', 'C']:
                    optical_section_path = os.path.join(optical_person_path, section)
                    visual_section_path = os.path.join(visual_person_path, section)
                    
                    if os.path.isdir(optical_section_path) and os.path.isdir(visual_section_path):
                        f.write(f"  Section: {section}\n")
                        for scenario in ['bump', 'corner']:
                            optical_scenario_path = os.path.join(optical_section_path, scenario)
                            visual_scenario_path = os.path.join(visual_section_path, scenario)
                            
                            if os.path.isdir(optical_scenario_path) and os.path.isdir(visual_scenario_path):
                                f.write(f"    Scenario: {scenario}\n")
                                for round_folder in os.listdir(optical_scenario_path):
                                    optical_round_path = os.path.join(optical_scenario_path, round_folder)
                                    visual_round_path = os.path.join(visual_scenario_path, round_folder)
                                    
                                    if os.path.isdir(optical_round_path) and os.path.isdir(visual_round_path):
                                        optical_files = os.listdir(optical_round_path)
                                        visual_files = os.listdir(visual_round_path)
                                        
                                        f.write(f"      Round: {round_folder}\n")
                                        f.write(f"        Optical data count: {len(optical_files)}\n")
                                        f.write(f"        Visual data count: {len(visual_files)}\n")

# Usage
base_dir = "./images/"
output_file = "images_shuffle_data_count_수정후.txt"
count_data_per_round(base_dir, output_file)