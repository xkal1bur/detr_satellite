import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from model import create_model, CLASSES
from pathlib import Path
import csv

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "out/checkpoint.pth"
images_dir = Path("data/val")  # Carpeta con imágenes a inferir
means = torch.tensor([0.3428944945335388, 0.34861984848976135, 0.31862562894821167])
stds = torch.tensor([0.1603400707244873, 0.15214884281158447, 0.14776213467121124])
num_classes = 16  # SOLO tus clases reales

# --- Preprocessing ---
def preprocess_image(img_path):
    image = read_image(str(img_path))
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
model = create_model(num_classes=num_classes, pretrained=False)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# --- Inference loop ---
submission_rows = []
image_paths = sorted(list(images_dir.glob("*.png")))[:5]  # Cambia extensión si es necesario

for img_path in image_paths:
    img_tensor = preprocess_image(img_path).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    boxes = outputs['pred_boxes'][0, keep].cpu()
    scores = probas[keep].cpu()
    labels = scores.argmax(-1)

    img_id = img_path.stem
    print(f"\nImagen: {img_path.stem}")
    print("Probabilidades (primeras 5 queries):")
    print(probas[:5])  # Muestra las probabilidades de las primeras 5 queries
    print("Clases predichas (primeras 5 queries):")
    print([CLASSES[i] for i in labels[:5]])
    preds = []
    for box, label, score in zip(boxes, labels, scores.max(-1).values):
        coords = " ".join([f"{v:.2f}" for v in box.tolist()])
        preds.append(f"{CLASSES[label]} {score:.2f} {coords}")

    pred_str = "; ".join(preds)
    submission_rows.append([img_id, pred_str])

# --- Guardar CSV ---
with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Id", "Predicted"])
    writer.writerows(submission_rows)

print("¡Archivo submission.csv generado!")