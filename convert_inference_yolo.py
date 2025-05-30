import os
import pandas as pd
import cv2

# === CONFIGURACIÓN ===
img_dir = "data2/test/images"
#label_dir = "runs_/predict_target_poly/target_infer/labels"
label_dir = "labels_1024"
output_obb_csv = "submission.csv"

class_names = [
    'small-vehicle', 'ship', 'large-vehicle', 'plane', 'harbor', 'storage-tank',
    'tennis-court', 'swimming-pool', 'bridge', 'helicopter', 'basketball-court',
    'roundabout', 'baseball-diamond', 'soccer-ball-field', 'ground-track-field',
    'container-crane'
]

# --- Obtener todos los IDs desde img_dir ---
all_ids = [os.path.splitext(f)[0] for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

rows_obb = []

for image_id in all_ids:
    image_path = os.path.join(img_dir, image_id + ".png")
    label_path = os.path.join(label_dir, image_id + ".txt")

    img = cv2.imread(image_path)
    if img is None or not os.path.exists(label_path):
        rows_obb.append({"Id": image_id, "Predicted": " "})
        continue

    h, w = img.shape[:2]
    with open(label_path, "r") as f:
        lines = f.readlines()

    preds_obb = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        cls_id = int(parts[0])
        obb_norm = list(map(float, parts[1:9]))
        conf = float(parts[9]) if len(parts) > 9 else 1.0
        label = class_names[cls_id]
        # Desnormalizar
        obb_abs = [obb_norm[i] * w if i % 2 == 0 else obb_norm[i] * h for i in range(8)]
        obb_str = f"{label} {conf:.2f} " + " ".join(f"{p:.2f}" for p in obb_abs)
        preds_obb.append(obb_str)

    rows_obb.append({
        "Id": image_id,
        "Predicted": "; ".join(preds_obb)
    })

pd.DataFrame(rows_obb).to_csv(output_obb_csv, index=False)

print(f"✅ Guardado {output_obb_csv}")