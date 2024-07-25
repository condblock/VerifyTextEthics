import json
import glob

def load_and_merge_data(file_pattern, output_file):
    merged_data = []
    for file_path in glob.glob(file_pattern):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            merged_data.extend(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

file_pattern = 'dataset/train/talksets-train-*.json'
output_file = 'dataset/train/merged_train_data.json'

load_and_merge_data(file_pattern, output_file)

print(f"Data merged and saved to {output_file}")
