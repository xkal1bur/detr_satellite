import torch
import torchvision.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_image_with_boxes(image_tensor, boxes, labels, font_size=12, box_color='red', label_color='white'):
    """
    Dibuja una imagen con sus cajas delimitadoras y etiquetas sobre ella.
    
    Args:
    - image_tensor: Tensor de imagen en formato [C, H, W]
    - boxes: Lista de coordenadas de las cajas de delimitación [xmin, ymin, xmax, ymax]
    - labels: Lista de etiquetas correspondientes a las cajas
    - font_size: Tamaño de la fuente de la etiqueta
    - box_color: Color de las cajas
    - label_color: Color del texto de las etiquetas
    """
    image = image_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x_coords = [box[0], box[2], box[4], box[6], box[0]]
        y_coords = [box[1], box[3], box[5], box[7], box[1]]
        ax.plot(x_coords, y_coords, linewidth=2, color=box_color)
        ax.text(box[0], box[1], f'{label[0]}', fontsize=font_size, color=label_color,
                bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

if __name__ == "__main__":
    image_path = "./data/train/P0212.png"
    image_tensor = io.read_image(image_path)

    labels_path = "./data/train_labs/P0212.txt"
    with open(labels_path, 'r') as f:
        next(f)
        gsd_raw = next(f)
        gsd = float(gsd_raw.split(':')[1].strip())
        lines = f.readlines()

    boxes = []
    labels = []
    for line in lines:
        parts = line.split()
        coords = list(map(float, parts[:8]))
        label = parts[8]
        incognite = int(parts[9])
        boxes.append(coords)
        labels.append((label, incognite))

    plot_image_with_boxes(image_tensor, boxes, labels)
