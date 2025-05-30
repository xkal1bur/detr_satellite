import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
# Configura tu path de imagen
#image_path = "data2/val/images/P0003.png"

## Fixear P1210, P1212, P1619 y P1750

### explorar toda la carpeta data/target/*.png y mostrar uno aleatorio
image_path = random.choice([
    os.path.join("data/target", f)
    for f in os.listdir("data/target") if f.endswith('.png')
])
#image_path = "data/target/P1750.png"
image_id = os.path.splitext(os.path.basename(image_path))[0]  # "P0029"

# Cargar CSV con predicciones
#df = pd.read_csv("submission_obb.csv")
#df = pd.read_csv("submission.csv")
df = pd.read_csv("submission.csv")

# Buscar la fila correspondiente
row = df[df['Id'] == image_id]
if row.empty:
    print(f"❌ No se encontraron predicciones para la imagen {image_id}")
    exit()

prediction_str = row.iloc[0]['Predicted']
if pd.isna(prediction_str) or prediction_str.strip() == "":
    print(f"⚠️ Predicción vacía para {image_id}")
    exit()

# Leer imagen
img = cv2.imread(image_path)
assert img is not None, f"Imagen no encontrada: {image_path}"
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Dibujar cada OBB
detections = prediction_str.strip().split(';')

for det in detections:
    det = det.strip()
    if not det:
        continue
    parts = det.split()
    label = parts[0]
    conf = float(parts[1])
    coords = list(map(float, parts[2:]))

    # Agrupar coordenadas por pares (x, y)
    points = np.array(coords, dtype=np.int32).reshape((4, 2))

    # Dibujar el polígono
    cv2.polylines(img, [points], isClosed=True, color=(255, 0, 0), thickness=2)

    # Poner la etiqueta con confianza
    text = f"{label} {conf:.2f}"
    cv2.putText(img, text, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Mostrar la imagen
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.title(f"Inferencias OBB - {image_id}")
plt.show()
