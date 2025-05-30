import os
import shutil

def copy_subset(img_src_dir, label_src_dir, img_dst_dir, label_dst_dir, num_files):
    os.makedirs(img_dst_dir, exist_ok=True)
    if label_dst_dir:
        os.makedirs(label_dst_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(img_src_dir) if f.endswith(('.jpg', '.png'))])[:num_files]

    for f in files:
        img_src = os.path.join(img_src_dir, f)
        shutil.copy(img_src, os.path.join(img_dst_dir, f))

        if label_src_dir and label_dst_dir:
            label_name = os.path.splitext(f)[0] + '.txt'
            label_src = os.path.join(label_src_dir, label_name)
            if os.path.exists(label_src):
                shutil.copy(label_src, os.path.join(label_dst_dir, label_name))
            else:
                print(f"⚠️ Label not found for image {f}, skipping label.")

    print(f"✅ Copiados {len(files)} archivos de {img_src_dir} a {img_dst_dir}")

# Configura cuántos archivos usar
num_files_ = 100
tamano_val = int(num_files_ * 0.2)

# Subsets
copy_subset(
    img_src_dir='data/train',
    label_src_dir='data2/train/labels_complete_poly',
    img_dst_dir='data2/train/images',
    label_dst_dir='data2/train/labels',
    num_files=num_files_
)

copy_subset(
    img_src_dir='data/val',
    label_src_dir='data2/val/labels_complete_poly',
    img_dst_dir='data2/val/images',
    label_dst_dir='data2/val/labels',
    num_files=tamano_val
)

# Este subset no tiene labels
copy_subset(
    img_src_dir='data/target',
    label_src_dir=None,
    img_dst_dir='data2/test/images',
    label_dst_dir=None,
    num_files=tamano_val
)
