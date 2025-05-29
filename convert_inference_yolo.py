import os
import pandas as pd
import cv2

# === CONFIGURACIÓN ===
img_dir = "data2/test/images"
label_dir = "runs/predict_target/target_infer/labels"  # donde están los .txt
output_yolo_csv = "submission_target_yolo.csv"
output_obb_csv = "submission_target_obb.csv"

# Orden de clases
class_names = [
    'small-vehicle', 'ship', 'large-vehicle', 'plane', 'harbor', 'storage-tank',
    'tennis-court', 'swimming-pool', 'bridge', 'helicopter', 'basketball-court',
    'roundabout', 'baseball-diamond', 'soccer-ball-field', 'ground-track-field',
    'container-crane'
]

# Función para pasar de bbox a obb
def bbox_to_obb(cx, cy, w, h):
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = y1
    x3 = x2
    y3 = cy + h / 2
    x4 = x1
    y4 = y3
    return [x1, y1, x2, y2, x3, y3, x4, y4]

# === Procesar archivos ===
rows_yolo = []
rows_obb = []

for file in sorted(os.listdir(label_dir)):
    if not file.endswith(".txt"):
        continue

    image_id = os.path.splitext(file)[0]
    label_path = os.path.join(label_dir, file)
    image_path = os.path.join(img_dir, image_id + ".png")

    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Imagen no encontrada: {image_path}")
        continue
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    preds_yolo = []
    preds_obb = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue

        cls_id = int(parts[0])
        cx, cy, bw, bh, conf = map(float, parts[1:])
        label = class_names[cls_id]

        # YOLO format (normalizado)
        yolo_str = f"{label} {conf:.2f} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        preds_yolo.append(yolo_str)

        # Convertir a OBB en coordenadas absolutas
        cx_abs = cx * w
        cy_abs = cy * h
        bw_abs = bw * w
        bh_abs = bh * h
        obb_coords = bbox_to_obb(cx_abs, cy_abs, bw_abs, bh_abs)
        obb_str = f"{label} {conf:.2f} " + " ".join(str(int(p)) for p in obb_coords)
        preds_obb.append(obb_str)

    rows_yolo.append({
        "Id": image_id,
        "Predicted": "; ".join(preds_yolo)
    })
    rows_obb.append({
        "Id": image_id,
        "Predicted": "; ".join(preds_obb)
    })

# === Guardar CSVs ===
pd.DataFrame(rows_yolo).to_csv(output_yolo_csv, index=False)
pd.DataFrame(rows_obb).to_csv(output_obb_csv, index=False)

print(f"✅ Guardado {output_yolo_csv} y {output_obb_csv}")
