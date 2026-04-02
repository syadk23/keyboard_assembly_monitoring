import os
from pathlib import Path

def update_labels(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    removed_count = 0
    
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        try:
            class_id = int(parts[0])
            
            # Change label 15 to 11
            if class_id == 15:
                parts[0] = "11"
                updated_lines.append(" ".join(parts) + "\n")
                print(f"  Changed class ID 15 to 11 in {file_path}")
            # Keep only labels with class ID <= 10
            elif class_id <= 10:
                updated_lines.append(" ".join(parts) + "\n")
            else:
                removed_count += 1
                print(f"  Removed class ID {class_id} from {file_path}")
        except ValueError:
            # Invalid class ID format, skip this line
            removed_count += 1
            print(f"  Removed invalid line from {file_path}: {line.strip()}")

    with open(file_path, "w") as f:
        f.writelines(updated_lines)

    if removed_count > 0:
        print(f"Updated {file_path} - removed {removed_count} labels")

def process_label_folder(label_dir):
    """Process all label files in a directory"""
    label_path = Path(label_dir)
    if not label_path.exists():
        print(f"Directory not found: {label_dir}")
        return
    
    txt_files = list(label_path.glob("*.txt"))
    print(f"Found {len(txt_files)} label files in {label_dir}")
    
    for file_path in txt_files:
        update_labels(str(file_path))

def process_all_splits(dataset_dir="dataset"):
    """Process all train, valid, and test splits"""
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        label_dir = f"{dataset_dir}/{split}/labels"
        label_path = Path(label_dir)
        
        if label_path.exists():
            print(f"\nProcessing {split} split...")
            process_label_folder(label_dir)
        else:
            print(f"Skipping {split} - directory not found: {label_dir}")

if __name__ == "__main__":
    # Process all splits
    print("Changing label 15 to 11 and removing labels with class ID > 10...")
    process_all_splits("switch_dataset")
    print("\n✅ Done!")
