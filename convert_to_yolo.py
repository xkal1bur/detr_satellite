import os
from pathlib import Path

print(os.getcwd())

# Define tus clases
class_map = {'small-vehicle': 0, 'ship': 1, 'large-vehicle': 2, 'plane': 3,
             'harbor': 4, 'storage-tank': 5, 'tennis-court': 6, 'swimming-pool': 7,
             'bridge': 8, 'helicopter': 9, 'basketball-court': 10, 'roundabout': 11,
             'baseball-diamond': 12, 'soccer-ball-field': 13, 'ground-track-field': 14,
             'container-crane': 15}

def polygon_to_yolo(coords, img_w, img_h):
    xs = coords[::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

def process_dir(txt_dir, img_dir, out_dir, img_ext='png'):
    os.makedirs(out_dir, exist_ok=True)
    for txt_file in os.listdir(txt_dir):
        if not txt_file.endswith('.txt'):
            continue
        img_path = os.path.join(img_dir, txt_file.replace('.txt', f'.{img_ext}'))
        if not os.path.exists(img_path):
            print(f"Image not found for: {txt_file}")
            continue
        from PIL import Image
        img = Image.open(img_path)
        w, h = img.size

        lines_out = []
        with open(os.path.join(txt_dir, txt_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('imagesource') or line.startswith('gsd:'):
                    continue  # Salta líneas de metadatos o vacías

                parts = line.split()
                try:
                    coords = list(map(float, parts[:8]))
                except ValueError:
                    print(f"Skipping invalid line in {txt_file}: {line}")
                    continue
                label = parts[8]
                if label not in class_map:
                    continue
                x_center, y_center, width, height = polygon_to_yolo(coords, w, h)
                cls = class_map[label]
                lines_out.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")


        with open(os.path.join(out_dir, txt_file), 'w') as f:
            f.write('\n'.join(lines_out))

# Cambia rutas según tu estructura
process_dir('data/train_labs', 'data/train', 'data2/train/labels_complete')
process_dir('data/val_labs', 'data/val', 'data2/val/labels_complete')
