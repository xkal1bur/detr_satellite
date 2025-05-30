import os
import cv2
import pandas as pd
from ultralytics import YOLO

# === Configuración ===
model = YOLO(f'runs/obb/train/weights/best.pt')
img_dir = 'data2/val/images'
image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

# Mapa de clases
class_names = [
    'small-vehicle', 'ship', 'large-vehicle', 'plane', 'harbor', 'storage-tank',
    'tennis-court', 'swimming-pool', 'bridge', 'helicopter', 'basketball-court',
    'roundabout', 'baseball-diamond', 'soccer-ball-field', 'ground-track-field',
    'container-crane'
]

rows_obb = []

for file in image_files:
    path = os.path.join(img_dir, file)
    img = cv2.imread(path)
    h, w = img.shape[:2]

    results = model(path)[0]
    preds_obb = []

    print(f"Procesando {file}... Detecciones: {getattr(results, 'boxes', None)}")
    if getattr(results, "boxes", None) is not None and len(results.boxes) > 0:
        print(f"  → {len(results.boxes)} detecciones")
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            obb_norm = box.xyxyxyxy[0].tolist()
            obb_abs = [obb_norm[i] * w if i % 2 == 0 else obb_norm[i] * h for i in range(8)]
            obb_str = " ".join(f"{p:.2f}" for p in obb_abs)
            preds_obb.append(f"{class_names[cls_id]} {conf:.2f} {obb_str}")
    else:
        print("  → Sin detecciones")

    img_id = os.path.splitext(file)[0]
    rows_obb.append({'Id': img_id, 'Predicted': '; '.join(preds_obb)})

pd.DataFrame(rows_obb).to_csv('submission_obb.csv', index=False)

print("✅ Archivo generado: submission_obb.csv (formato OBB en píxeles)")