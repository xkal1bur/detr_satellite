import os
from pathlib import Path

print(os.getcwd())

# Define tus clases
class_map = {'small-vehicle': 0, 'ship': 1, 'large-vehicle': 2, 'plane': 3,
             'harbor': 4, 'storage-tank': 5, 'tennis-court': 6, 'swimming-pool': 7,
             'bridge': 8, 'helicopter': 9, 'basketball-court': 10, 'roundabout': 11,
             'baseball-diamond': 12, 'soccer-ball-field': 13, 'ground-track-field': 14,
             'container-crane': 15}

def process_dir(txt_dir, img_dir, out_dir_poly, img_ext='png'):
    os.makedirs(out_dir_poly, exist_ok=True)
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

        lines_out_yolo = []
        lines_out_poly = []
        with open(os.path.join(txt_dir, txt_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('imagesource') or line.startswith('gsd:'):
                    continue

                parts = line.split()
                try:
                    coords = list(map(float, parts[:8]))
                except ValueError:
                    print(f"Skipping invalid line in {txt_file}: {line}")
                    continue
                label = parts[8]
                if label not in class_map:
                    continue
                cls = class_map[label]
                # OBB normalizado
                norm_coords = [coords[i] / w if i % 2 == 0 else coords[i] / h for i in range(8)]
                norm_coords_str = " ".join(f"{v:.6f}" for v in norm_coords)
                lines_out_poly.append(f"{cls} {norm_coords_str}")

        with open(os.path.join(out_dir_poly, txt_file), 'w') as f:
            f.write('\n'.join(lines_out_poly))

# Cambia rutas según tu estructura
process_dir(
    'data/train_labs', 'data/train',
    'data2/train/labels_complete_poly'
)
process_dir(
    'data/val_labs', 'data/val',
    'data2/val/labels_complete_poly'
)
