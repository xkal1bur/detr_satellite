import os
import shutil
import random

img_names = sorted(os.listdir("train"))
lab_names = [img.replace(".png", ".txt") for img in img_names]  # Added dots before file extensions

os.makedirs("val", exist_ok=True)
os.makedirs("val_labs", exist_ok=True)

n_val = int(0.2 * len(img_names))

idx_val = random.sample(range(len(img_names)), n_val)

for idx in idx_val:
    img_source = f"train/{img_names[idx]}"
    lab_source = f"train_labs/{lab_names[idx]}"
    img_dest = f"val/{img_names[idx]}"
    lab_dest = f"val_labs/{lab_names[idx]}"
    
    if os.path.exists(img_source):
        shutil.move(img_source, img_dest)
    if os.path.exists(lab_source):
        shutil.move(lab_source, lab_dest)
