import os
import json
import shutil

# Function to process each JSON file
def process_json_file(input_json_path, start_idx, output_folder):
    # Extract the base name and base directory
    base_name = os.path.basename(input_json_path).split('_logs')[0]
    base_dir = os.path.dirname(input_json_path)

    # Define the output folder and image folder paths
    image_folder_name = f'{base_name}___0_original'
    image_folder_path = os.path.join(base_dir, image_folder_name)

    # Check if the image folder exists
    if not os.path.exists(image_folder_path):
        print(f"Image folder not found: {image_folder_path}")
        return start_idx

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the input JSON file
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Process the data and create a new JSON file for each frame and its associated label
    for entry in data:
        frames = entry.get('frames', [])
        pid_actions = entry.get('pid_actions', [])
        ctes = entry.get('xte', [])

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
        
        print(f'Saving files of {image_folder_name} (Dropped {dropped} frames)')
        print("Now run [examples/models/train_dave2.py --model ./checkpoints/original.h5] to train the nominal model")

    return start_idx

# Folder containing JSON files
json_folder = 'output_logs/train_nominal'
output_folder = 'examples/models/datasets/train_dataset'

# Get list of all JSON files in the folder
input_json_files = [os.path.join(json_folder, file) for file in os.listdir(json_folder) if file.endswith('.json')]

# Starting index for file naming
start_idx = 1

# Process each JSON file
for input_json_file in input_json_files:
    start_idx = process_json_file(input_json_file, start_idx, output_folder)

print(f'Processed all files and saved to {output_folder}')