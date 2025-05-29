import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

# === Configura esto ===
#image_path = "data2/val/images/P0018.png"
image_path = "data/target/P0003.png"
#ground_truth_txt_path = "data2/val/labels/P0018.txt"
ground_truth_txt_path = "data2/val/labels/P0003.txt"
#submission_csv_path = "submission_yolo.csv"
submission_csv_path = "submission_target_yolo.csv"
class_names = [
    'small-vehicle', 'ship', 'large-vehicle', 'plane', 'harbor', 'storage-tank',
    'tennis-court', 'swimming-pool', 'bridge', 'helicopter', 'basketball-court',
    'roundabout', 'baseball-diamond', 'soccer-ball-field', 'ground-track-field',
    'container-crane'
]

# === Obtener ID de la imagen ===
image_id = os.path.splitext(os.path.basename(image_path))[0]  # "P0018"

# === Cargar imagen ===
image = cv2.imread(image_path)
assert image is not None, f"Error cargando imagen: {image_path}"
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]

# === Dibujar ground truth ===
if os.path.exists(ground_truth_txt_path):
    with open(ground_truth_txt_path, 'r') as f:
        gt_lines = f.readlines()

    for line in gt_lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        class_id = int(class_id)
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Rojo para ground truth
        label = class_names[class_id] if class_id < len(class_names) else str(class_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# === Leer predicciones del CSV ===
df = pd.read_csv(submission_csv_path)
row = df[df['Id'] == image_id]
if not row.empty:
    prediction_str = row.iloc[0]['Predicted']
    if isinstance(prediction_str, str) and prediction_str.strip():
        predictions = prediction_str.strip().split(';')
        for pred in predictions:
            pred = pred.strip()
            if not pred:
                continue
            parts = pred.split()
            label = parts[0]
            conf = float(parts[1])
            x_center, y_center, width, height = map(float, parts[2:])

            x_center *= w
            y_center *= h
            width *= w
            height *= h
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Verde para predicciones
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
else:
    print(f"❌ No se encontró predicción para {image_id} en el CSV")

# === Mostrar resultado ===
plt.figure(figsize=(12, 8))
plt.imshow(image)
plt.axis('off')
plt.title(f"YOLO - Predicciones (verde) vs GT (rojo) - {image_id}")
plt.show()
