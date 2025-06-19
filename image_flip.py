import os
import cv2

def flip_yolo_labels(yolo_labels, flip_type):
    new_labels = []
    for label in yolo_labels:
        cls, x, y, w, h = map(float, label.strip().split())
        if flip_type == "horizontal":
            x = 1 - x
        elif flip_type == "vertical":
            y = 1 - y
        new_labels.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return new_labels

def process_image(img_path, label_path, output_img_path, output_label_path, flip_type="horizontal"):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error loading image: {img_path}")
        return

    # Flip image
    flip_code = 1 if flip_type == "horizontal" else 0
    flipped_img = cv2.flip(img, flip_code)

    # Save flipped image
    cv2.imwrite(output_img_path, flipped_img)

    # Read original label file
    if not os.path.exists(label_path):
        print(f"⚠️ Label file missing for {img_path}")
        return

    with open(label_path, "r") as f:
        labels = f.readlines()

    # Flip bounding boxes
    new_labels = flip_yolo_labels(labels, flip_type)

    # Save updated label file
    with open(output_label_path, "w") as f:
        f.write("\n".join(new_labels))

    print(f"✅ Saved: {output_img_path} and {output_label_path}")

# Example usage
img_folder = "pcb/"
label_folder = "pcblabel/"
output_img_folder = "flipped_images/"
output_label_folder = "flipped_labels/"
os.makedirs(output_img_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

for img_name in os.listdir(img_folder):
    if not img_name.endswith((".jpg", ".png")):
        continue
    img_path = os.path.join(img_folder, img_name)
    label_path = os.path.join(label_folder, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

    output_img_path = os.path.join(output_img_folder, "flipped_" + img_name)
    output_label_path = os.path.join(output_label_folder, "flipped_" + img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

    process_image(img_path, label_path, output_img_path, output_label_path, flip_type="horizontal")
