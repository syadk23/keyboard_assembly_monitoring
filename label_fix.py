import os

def update_labels(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        class_id = parts[0]
        if class_id == '15':
            parts[0] = '65'


        updated_lines.append(" ".join(parts) + "\n")

    with open(file_path, "w") as f:
        f.writelines(updated_lines)

    print(f"Updated {file_path}")

def process_label_folder(label_dir):
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(label_dir, filename)
            update_labels(file_path)

# 🔁 Replace this with your actual path
label_folder = "New folder"

process_label_folder(label_folder)
