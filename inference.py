import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from model import create_model, CLASSES
from pathlib import Path

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "out/checkpoint.pth"
image_path = "data/val/P0002.png"  # Change to your image path
means = torch.tensor([0.3428944945335388, 0.34861984848976135, 0.31862562894821167])
stds = torch.tensor([0.1603400707244873, 0.15214884281158447, 0.14776213467121124])
num_classes = 17  # 16 + background

# --- Preprocessing ---
def preprocess_image(img_path):
    image = read_image(img_path)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] == 4:
        image = image[:3, :, :]
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((512, 512), antialias=True, interpolation=v2.InterpolationMode.BICUBIC),
        v2.Normalize(mean=means, std=stds)
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# --- Load Model ---
model = create_model(num_classes=num_classes, pretrained=True)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# --- Inference ---
img_tensor = preprocess_image(image_path).to(device)
with torch.no_grad():
    outputs = model(img_tensor)

# --- Postprocess ---
# Get probabilities and boxes
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]  # Remove "no object" class
keep = probas.max(-1).values > 0.7  # Confidence threshold

boxes = outputs['pred_boxes'][0, keep].cpu()
scores = probas[keep].cpu()
labels = scores.argmax(-1)

# Print results
for box, label, score in zip(boxes, labels, scores.max(-1).values):
    print(f"Label: {CLASSES[label]}, Score: {score:.2f}, Box: {box.tolist()}")

def bbox_to_polygon(box):
    # box: [x_min, y_min, x_max, y_max] -> [x0, y0, x1, y0, x1, y1, x0, y1]
    x_min, y_min, x_max, y_max = box.tolist()
    return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

boxes_poly = [bbox_to_polygon(box) for box in boxes]

# Optionally, visualize results
from visualize import plot_image_with_boxes, unnormalize_image
img = read_image(image_path)
img = img[:3] if img.shape[0] == 4 else img
plot_image_with_boxes(unnormalize_image(img, mean=means, std=stds), boxes_poly, [CLASSES[l] for l in labels])