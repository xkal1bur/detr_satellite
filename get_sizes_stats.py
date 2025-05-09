import os
import torch
from torchvision.io import read_image
import numpy as np

img_dir = 'data/target' # Evaluation :v
img_files = sorted(os.listdir(img_dir))

heights = []
widths = []

for fname in img_files:
    path = os.path.join(img_dir, fname)
    try:
        img_tensor = read_image(path)  # (C, H, W)
        _, h, w = img_tensor.shape
        heights.append(h)
        widths.append(w)
    except Exception as e:
        print(f"Error con {fname}: {e}")

# Convertir a numpy para estadísticas
heights = np.array(heights)
widths = np.array(widths)

# Función auxiliar para imprimir stats
def print_stats(name, arr):
    print(f"\n📊 Estadísticas de {name}:")
    print(f"- Count:     {len(arr)}")
    print(f"- Min:       {np.min(arr)}")
    print(f"- Max:       {np.max(arr)}")
    print(f"- Mean:      {np.mean(arr):.2f}")
    print(f"- Median:    {np.median(arr)}")
    print(f"- Std Dev:   {np.std(arr):.2f}")
    print(f"- Q1 (25%):  {np.percentile(arr, 25)}")
    print(f"- Q3 (75%):  {np.percentile(arr, 75)}")

print_stats("Alturas (H)", heights)
print_stats("Anchos (W)", widths)

""" TRAIN
📊 Estadísticas de Alturas (H):
- Count:     1129
- Min:       346
- Max:       8115
- Mean:      2200.56
- Median:    1835.0
- Std Dev:   1375.13
- Q1 (25%):  1098.0
- Q3 (75%):  3241.0

📊 Estadísticas de Anchos (W):
- Count:     1129
- Min:       387
- Max:       12029
- Mean:      2329.80
- Median:    1849.0
- Std Dev:   1553.10
- Q1 (25%):  1139.0
- Q3 (75%):  3525.0
"""


""" VAL
📊 Estadísticas de Alturas (H):
- Count:     282
- Min:       278
- Max:       7282
- Mean:      2104.13
- Median:    1624.0
- Std Dev:   1364.20
- Q1 (25%):  1067.0
- Q3 (75%):  3181.75

📊 Estadísticas de Anchos (W):
- Count:     282
- Min:       420
- Max:       10478
- Mean:      2204.08
- Median:    1685.0
- Std Dev:   1505.21
- Q1 (25%):  1059.25
- Q3 (75%):  3212.75
"""

""" TARGET
📊 Estadísticas de Alturas (H):
- Count:     458
- Min:       511
- Max:       6759
- Mean:      2210.98
- Median:    1705.5
- Std Dev:   1415.34
- Q1 (25%):  1098.0
- Q3 (75%):  3418.5

📊 Estadísticas de Anchos (W):
- Count:     458
- Min:       353
- Max:       13383
- Mean:      2398.01
- Median:    1871.5
- Std Dev:   1666.13
- Q1 (25%):  1086.75
- Q3 (75%):  4000.0
"""
