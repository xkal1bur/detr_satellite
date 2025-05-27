import os
import shutil
import random

img_dir = "train_extracted"
lab_dir = "train_extracted_labs"
img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
lab_names = [img.replace(".png", ".txt") for img in img_names]  # Added dots before file extensions

os.makedirs("train", exist_ok=True)
os.makedirs("train_labs", exist_ok=True)

os.makedirs("val", exist_ok=True)
os.makedirs("val_labs", exist_ok=True)

# use a seed for random sampling
random.seed(42)
idxs = list(range(len(img_names)))
random.shuffle(idxs)
n_val = int(0.2 * len(img_names))
# use the random seed
idx_val = set(idxs[:n_val])

for i, (img, lab) in enumerate(zip(img_names, lab_names)):
    img_source = os.path.join(img_dir, img)
    lab_source = os.path.join(lab_dir, lab)
    if i in idx_val:
        img_dest = os.path.join("val", img)
        lab_dest = os.path.join("val_labs", lab)
    else:
        img_dest = os.path.join("train", img)
        lab_dest = os.path.join("train_labs", lab)
    shutil.copy(img_source, img_dest)
    if os.path.exists(lab_source):
        shutil.copy(lab_source, lab_dest)
