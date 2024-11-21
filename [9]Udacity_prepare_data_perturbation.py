import os
import json
import shutil

def find_subfolders_with_string(root_folder, search_string):
    matching_folders = []
    
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if search_string in dirname:
                matching_folders.append(os.path.join(dirpath, dirname))
    
    return matching_folders

def find_specific_folder(folders_list, second_search_string):
    for folder in folders_list:
        if second_search_string in os.path.basename(folder):
            return folder
    return None

# Function to process each JSON file
def process_json_file(input_json_path,json_folder, start_idx, output_folder):
    # Extract the base name and base directory
    base_name = os.path.basename(input_json_path).split('_logs')[0]
    base_dir = os.path.dirname(input_json_path)
    matching_folders = find_subfolders_with_string(base_dir, base_name)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the input JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Process the data and create a new JSON file for each frame and its associated label
    for entry in data:
        perturbation=entry.get('scenario', [])["perturbation_function"]
        scale=entry.get('scenario', [])["perturbation_scale"]
        frames = entry.get('frames', [])
        pid_actions = entry.get('pid_actions', [])
        ctes = entry.get('xte', [])

        images_string=f"{base_name}__{perturbation}_{scale}_perturbed"
        image_folder_path=find_specific_folder(matching_folders, images_string)

        if len(frames) != len(pid_actions):
            print(f"Mismatch in lengths of frames and pid_actions: {len(frames)} vs {len(pid_actions)}")
            continue
        print(f"Reading {len(frames)} frames")
        dropped=0
        previous_cte=10
        
        for frame, action, cte in zip(frames, pid_actions, ctes):
            # Ensure action has at least two values
            # if abs(cte)>previous_cte:
            #     # psrint(f"Dropped frame due to increasing cte: {previous_cte}->{cte}")
            #     dropped+=1
            #     previous_cte=abs(cte)
            #     continue
            action = action[0]
            previous_cte=abs(cte)

            # Create a new JSON object for each frame
            simple_json = {
                "user/angle": str(action[0]), 
                "user/throttle": str(action[1])
            }

            # Generate JSON file name with naming convention: record_000001.json
            json_output_path = os.path.join(output_folder, f'record_{start_idx:06d}.json')
            with open(json_output_path, 'w') as out_file:
                json.dump(simple_json, out_file, indent=4)

            # Copy the corresponding image with the new naming convention
            image_src_path = os.path.join(image_folder_path, f'{frame}.jpg')
            image_dst_path = os.path.join(output_folder, f'image_{start_idx:06d}.jpg')
            if os.path.exists(image_src_path):
                shutil.copy(image_src_path, image_dst_path)
            else:
                print(f"Image file not found: {image_src_path}")

            start_idx += 1
        
        print(f'Saving files of {images_string} (Dropped {dropped} frames)')

    return start_idx

# Folder containing JSON files
json_folder = 'output_logs/test_perturbation_extended3'
output_folder = '/examples/models/datasets/continue_train_original5'

# Get list of all JSON files in the folder
input_json_files = [os.path.join(json_folder, file) for file in os.listdir(json_folder) if file.endswith('.json')]

# Starting index for file naming
start_idx = 1

# Process each JSON file
for input_json_file in input_json_files:
    start_idx = process_json_file(input_json_file,json_folder, start_idx, output_folder)

print(f'Processed all files and saved to {output_folder}')