from dataclasses import dataclass
import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet101
import torchvision.transforms as T
from torchvision.transforms import v2
import requests
from PIL import Image
import matplotlib.pyplot as plt

@dataclass
class DERTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


# Nuestras clases
CLASSES = [
    'background',  # 0
    'small-vehicle',  # 1
    'ship',  # 2
    'large-vehicle',  # 3
    'plane',  # 4
    'harbor',  # 5
    'storage-tank',  # 6
    'tennis-court',  # 7
    'swimming-pool',  # 8
    'bridge',  # 9
    'helicopter',  # 10
    'basketball-court',  # 11
    'roundabout',  # 12
    'baseball-diamond',  # 13
    'soccer-ball-field',  # 14
    'ground-track-field',  # 15
    'container-crane'  # 16
]

class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.linear_bbox = nn.Linear(hidden_dim, 8)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings with larger initial size
        self.row_embed = nn.Parameter(torch.rand(100, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(100, hidden_dim // 2))

    def get_position_embeddings(self, H, W):
        # Interpolate position embeddings to match feature map size
        col_embed = nn.functional.interpolate(
            self.col_embed.unsqueeze(0).unsqueeze(0),
            size=(W, self.col_embed.shape[1]),
            mode='bilinear',
            align_corners=True
        ).squeeze(0).squeeze(0)
        
        row_embed = nn.functional.interpolate(
            self.row_embed.unsqueeze(0).unsqueeze(0),
            size=(H, self.row_embed.shape[1]),
            mode='bilinear',
            align_corners=True
        ).squeeze(0).squeeze(0)
        
        return row_embed, col_embed

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        row_embed, col_embed = self.get_position_embeddings(H, W)
        
        pos = torch.cat([
            col_embed.unsqueeze(0).repeat(H, 1, 1),
            row_embed.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                           self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()} # si no esta normalziada, no usar sigmoid

def modify_classifier_weights(state_dict, num_classes=16):
    """
    Modifica los pesos del clasificador y los positional embeddings para manejar el nuevo número de clases
    y el nuevo tamaño de embeddings. Además adapta la capa linear_bbox para polígonos (8 puntos).
    """
    # Modificar los pesos del clasificador
    orig_weight = state_dict['linear_class.weight']
    orig_bias = state_dict['linear_class.bias']
    
    # Crear nuevos tensores con el número deseado de clases (+1 para la clase no-objeto)
    new_weight = torch.zeros((num_classes + 1, orig_weight.size(1)), dtype=orig_weight.dtype)
    new_bias = torch.zeros(num_classes + 1, dtype=orig_bias.dtype)
    
    # Copiar los pesos de las clases que queremos mantener
    class_mapping = {
        0: 0,  # background/no-object
        3: 1,  # car -> small-vehicle
        8: 3,  # truck -> large-vehicle
        5: 4,  # airplane -> plane
    }
    
    for old_idx, new_idx in class_mapping.items():
        new_weight[new_idx] = orig_weight[old_idx]
        new_bias[new_idx] = orig_bias[old_idx]
    
    # Inicializar el resto de pesos con xavier uniform
    nn.init.xavier_uniform_(new_weight[len(class_mapping):])
    nn.init.zeros_(new_bias[len(class_mapping):])
    
    # Actualizar el state dict para el clasificador
    state_dict['linear_class.weight'] = new_weight
    state_dict['linear_class.bias'] = new_bias

    # Adaptar linear_bbox para 8 puntos (polígono)
    orig_bbox_weight = state_dict['linear_bbox.weight']
    orig_bbox_bias = state_dict['linear_bbox.bias']

    new_bbox_weight = torch.zeros((8, orig_bbox_weight.size(1)), dtype=orig_bbox_weight.dtype)
    new_bbox_bias = torch.zeros(8, dtype=orig_bbox_bias.dtype)

    # Copia los 4 primeros (cx, cy, w, h) en los primeros 4
    new_bbox_weight[:4] = orig_bbox_weight
    new_bbox_bias[:4] = orig_bbox_bias

    # Inicializa los 4 nuevos (para los otros vértices del polígono)
    nn.init.xavier_uniform_(new_bbox_weight[4:])
    nn.init.zeros_(new_bbox_bias[4:])

    state_dict['linear_bbox.weight'] = new_bbox_weight
    state_dict['linear_bbox.bias'] = new_bbox_bias

    # Redimensionar los positional embeddings
    for key in ['row_embed', 'col_embed']:
        if key in state_dict:
            old_embed = state_dict[key]
            new_size = [100, old_embed.size(1)]  # New size is 100 x hidden_dim//2
            
            # Crear un nuevo tensor más grande
            new_embed = torch.zeros(new_size, dtype=old_embed.dtype)
            
            # Copiar los valores existentes
            new_embed[:old_embed.size(0)] = old_embed
            
            # Inicializar los nuevos valores con xavier uniform
            if old_embed.size(0) < new_size[0]:
                nn.init.xavier_uniform_(new_embed[old_embed.size(0):])
            
            # Actualizar el state dict
            state_dict[key] = new_embed
    
    return state_dict

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
#transform = T.Compose([
#    T.Resize(800),
#    T.ToTensor(),
#    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#])

means = torch.tensor([0.3428944945335388, 0.34861984848976135, 0.31862562894821167])
stds = torch.tensor([0.1603400707244873, 0.15214884281158447, 0.14776213467121124])

image_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((1850, 1850), antialias=True, interpolation=v2.InterpolationMode.BICUBIC),
    v2.Normalize(mean=means, std=stds)
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def rescale_polygons(out_polygons, size):
    """Convierte polígonos normalizados [0,1] a escala de imagen"""
    img_w, img_h = size
    scale = torch.tensor([img_w, img_h] * 4, dtype=torch.float32)
    return out_polygons * scale


def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    #assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    #bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    bboxes_scaled = rescale_polygons(outputs['pred_boxes'][0, keep], im.size)

    return probas[keep], bboxes_scaled

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)


# use a img for my dataset in data/val/P0002.png
im = Image.open('data/val/P0002.png')

# Cargar y modificar el modelo
detr = DETRdemo(num_classes=16)  # 16 clases + 1 de fondo
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)

# Modificar los pesos del clasificador
state_dict = modify_classifier_weights(state_dict)

# Cargar el estado modificado
detr.load_state_dict(state_dict)
detr.eval()

scores, boxes = detect(im, detr, image_transform) # transform

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, poly, c in zip(prob, boxes.tolist(), COLORS * 100):
        xys = [(poly[i], poly[i+1]) for i in range(0, 8, 2)]
        xys.append(xys[0])  # cerrar el polígono
        xs, ys = zip(*xys)
        ax.plot(xs, ys, color=c, linewidth=2)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(poly[0], poly[1], text, fontsize=12,
            bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()
    
#plot_results(im, scores, boxes)

def create_model(num_classes=16, pretrained=True):
    """
    Create and initialize the DETR model.
    Args:
        num_classes (int): Number of object classes (excluding background)
        pretrained (bool): Whether to initialize from COCO pretrained weights
    Returns:
        model (DETRdemo): The initialized model
    """
    model = DETRdemo(num_classes=num_classes) # no sumar 1 acá
    
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
            map_location='cpu', check_hash=True)
        
        # Modify the classifier and positional embedding weights
        state_dict = modify_classifier_weights(state_dict, num_classes)
        
        # Load the modified state dict
        model.load_state_dict(state_dict)
    
    return model