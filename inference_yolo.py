import os
import cv2
import pandas as pd
from ultralytics import YOLO

# === Configuración ===
ultimo_train = 9
model = YOLO(f'runs/detect/train{ultimo_train}/weights/best.pt')
img_dir = 'data2/val/images'
image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

# Mapa de clases
class_names = [
    'small-vehicle', 'ship', 'large-vehicle', 'plane', 'harbor', 'storage-tank',
    'tennis-court', 'swimming-pool', 'bridge', 'helicopter', 'basketball-court',
    'roundabout', 'baseball-diamond', 'soccer-ball-field', 'ground-track-field',
    'container-crane'
]

# === Función para convertir bbox -> OBB (cuadrado sin rotación) ===
def bbox_to_obb(xc, yc, w, h):
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = y1
    x3 = x2
    y3 = yc + h / 2
    x4 = x1
    y4 = y3
    return [x1, y1, x2, y2, x3, y3, x4, y4]

# === Inicializar los datos para los CSVs ===
rows_yolo = []
rows_obb = []

# === Procesar imágenes ===
for file in image_files:
    path = os.path.join(img_dir, file)
    img = cv2.imread(path)
    h, w = img.shape[:2]

    results = model(path)[0]
    preds_yolo = []
    preds_obb = []

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        xc, yc, bw, bh = box.xywhn[0].tolist()

        # Convertir a absolutos
        xc_abs, yc_abs = xc * w, yc * h
        bw_abs, bh_abs = bw * w, bh * h

        # === Formato YOLO (xywh normalizado) ===
        preds_yolo.append(f"{class_names[cls_id]} {conf:.2f} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        # === Formato OBB ===
        obb = bbox_to_obb(xc_abs, yc_abs, bw_abs, bh_abs)
        obb_str = ' '.join(f"{int(p)}" for p in obb)
        preds_obb.append(f"{class_names[cls_id]} {conf:.2f} {obb_str}")

    img_id = os.path.splitext(file)[0]
    rows_yolo.append({'Id': img_id, 'Predicted': '; '.join(preds_yolo)})
    rows_obb.append({'Id': img_id, 'Predicted': '; '.join(preds_obb)})

# === Guardar CSVs ===
pd.DataFrame(rows_yolo).to_csv('submission_yolo.csv', index=False)
pd.DataFrame(rows_obb).to_csv('submission_obb.csv', index=False)

print("✅ Archivos generados:")
print(" - submission_yolo.csv (formato YOLO xywh)")
print(" - submission_obb.csv (formato OBB en píxeles)")
