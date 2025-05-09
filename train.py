import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
from torchvision.transforms import v2


# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
init_from = 'scratch'                 # 'scratch' | 'resume' | 'detr-resnet50'
eval_interval = 2000  # cada cuantos intervalos se evalua
log_interval = 1 # cada cuantos intervalos haremos logs :v
eval_iters = 200 # iteraciones de eval

# Datos
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "val")
num_workers = 2 # para el data loader
batch_size = 12                       # if gradient_accumulation_steps > 1, este es micro-batch size
gradient_accumulation_steps = 5 * 8   # para simular batch más grande 

# model
n_classes   = 999 + 1                   # clases + fondo
num_queries = 100
backbone    = 'resnet50'

# transformer
embed_dim      = 256                  # x8 in feedforward
n_heads        = 8
enc_layers     = 6
dec_layers     = 6
dropout        = 0.1 # arbitrario señor :v

# losses & bipartite matching
set_cost_class = 1                    # peso coste clasificación 
set_cost_bbox  = 5                    # peso coste bbox en matching 
set_cost_giou  = 2                    # peso coste giou en matching 
bbox_loss_coef = 5                    # coef de loss L1 bbox 
giou_loss_coef = 2                    # coef de loss giou 
eos_coef       = 0.1                  # peso para clase “no object” 
aux_loss       = True                 # usar pérdidas auxiliares en cada capa 
remove_difficult = False              # ignorar anotaciones difíciles 

# adamw & learning rate decay
lr            = 1e-4
weight_decay  = 1e-4
epochs        = 300
grad_clip     = 0.1
# to-do: warmup, min_lr, decay_iters, 

# system 
device    = "cuda"
dtype     = 'bfloat16' if torch.cuda.is_available() \
            and torch.cuda.is_bf16_supported() \
            else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile   = True           # usa pytorch 2.0 para compilar el modelo e ir más rápido
backend   = 'nccl'         # 'nccl', 'gloo', etc.


# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------


ddp = int(os.environ.get('RANK', -1)) != -1 # ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# TO-DO
# tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
# print(f"tokens per iteration will be: {tokens_per_iter:,}")


if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(53 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# DataLoader 
transform = v2.Compose([
    v2.ToImage()
])

import numpy as np
#wrap_dataset_for_transforms_v2()
class SatellitalDataset(Dataset):
    def __init__(self, img_dir, annot_dir, transform=None, target_transform=None):
        self.annot_dir = annot_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_names = sorted(os.listdir(img_dir))
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        image = read_image(f"{self.img_dir}{self.img_names[idx]}")

        boxes, label, incognite_value ,gsd = self._read_annot(f"{self.annot_dir}{self.img_names[idx].replace("png","txt")}")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            boxes = self.target_transform(boxes)
            
        boxes = [box.tolist() for box in boxes]
        return image, boxes, label
    
    def _read_annot(self, annot_path):
        with open(annot_path, 'r') as f:
            next(f)
            gsd_raw = next(f)
            gsd = float(gsd_raw.split(':')[1].strip())
            lines = f.readlines()
        boxes = []
        labels = []
        incognite = []
        for line in lines:
            parts = line.split()
            coords = np.array(parts[:8], dtype=np.float32)
            label = parts[8]
            incg = int(parts[9])
            
            boxes.append(coords)
            labels.append(label)
            incognite.append(incg)
        return boxes, labels, incognite, gsd
        

train_dataset = SatellitalDataset("data/train/", "data/train_labs/")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
img, boxes, labels  = next(iter(train_loader))

from visualize import plot_image_with_boxes
plot_image_with_boxes(img[0], boxes, labels)





if ddp:
    destroy_process_group()