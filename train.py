import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
from torchvision.transforms import v2
from model import create_model, CLASSES
from matcher import HungarianMatcher
from losses import DETRLoss
import time
from pathlib import Path
import math
import sys
from collections import deque
from datetime import datetime, timedelta


# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
init_from = 'detr-resnet50'                 # 'scratch' | 'resume' | 'detr-resnet50'
eval_interval = 2000  # cada cuantos intervalos se evalua
log_interval = 1 # cada cuantos intervalos haremos logs :v
eval_iters = 200 # iteraciones de eval

# Datos
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "val")
num_workers = 2 # para el data loader
batch_size = 1 #12                       # if gradient_accumulation_steps > 1, este es micro-batch size
gradient_accumulation_steps = 5 * 8   # para simular batch más grande 

# model
n_classes   = 16 + 1                   # clases + fondo
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
eos_coef       = 0.1                  # peso para clase "no object" 
aux_loss       = True                 # usar pérdidas auxiliares en cada capa 
remove_difficult = False              # ignorar anotaciones difíciles 

# adamw & learning rate decay
lr            = 1e-4
weight_decay  = 1e-4
epochs        = 1 #300
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


#ddp = int(os.environ.get('RANK', -1)) != -1 # ddp run?
ddp = False
'''
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
    ddp_world_size = 1'''

master_process = True # for now, we will not use ddp
seed_offset = 0 # for now, we will not use ddp
ddp_world_size = 1 # for now, we will not use ddp
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
class BoxScaler:
    def __init__(self, target_size):
        self.target_size = target_size
    
    def __call__(self, data):
        boxes, original_size = data
        h_scale = self.target_size[0] / original_size[0]
        w_scale = self.target_size[1] / original_size[1]

        scaled_boxes = []
        for box in boxes:
            scaled_box = []
            for i, coord in enumerate(box):
                scale = w_scale if i % 2 == 0 else h_scale
                scaled_box.append(coord * scale)
            scaled_boxes.append(scaled_box)
        return scaled_boxes

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
        image = read_image(os.path.join(self.img_dir, self.img_names[idx]))
        original_size = (image.shape[1], image.shape[2])

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3, :, :]  # descarta canal alfa

        if self.transform:
            image = self.transform(image)

        boxes, labels, incognite_value, gsd = self._read_annot(os.path.join(self.annot_dir, self.img_names[idx].replace("png", "txt")))
        if self.target_transform:
            boxes = self.target_transform((boxes, original_size))

        return image, boxes, labels
    
    def _read_annot(self, annot_path):
        with open(annot_path, 'r') as f:
            next(f)
            gsd_raw = next(f)
            gsd_str = gsd_raw.split(':')[1].strip()
            try:
                gsd = float(gsd_str)
            except ValueError:
                gsd = 0.0  # o -1, según prefieras
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
        
def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, boxes, labels

means = torch.tensor([0.3428944945335388, 0.34861984848976135, 0.31862562894821167])
stds = torch.tensor([0.1603400707244873, 0.15214884281158447, 0.14776213467121124])

image_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((512, 512), antialias=True, interpolation=v2.InterpolationMode.BICUBIC),
    v2.Normalize(mean=means, std=stds)
])

target_transform = BoxScaler(target_size=(512, 512))

train_dataset = SatellitalDataset(
    #img_dir=train_dir,
    # escoger solo las primeras 10 imágenes para entrenamiento
    img_dir=os.path.join(data_dir, "train")[:15],
    #annot_dir=os.path.join(data_dir, "train_labs"),
    # escoger solo las primeras 10 imágenes para entrenamiento
    annot_dir=os.path.join(data_dir, "train_labs")[:15],
    transform=image_transform,
    target_transform=target_transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_fn
)

img, boxes, labels = next(iter(train_loader))

from visualize import plot_image_with_boxes, unnormalize_image
img_to_plot = unnormalize_image(img[0], mean=means, std=stds)
#plot_image_with_boxes(img_to_plot, boxes[0], labels[0])

def polygon_to_bbox(box):
    # box: [8] -> [x_min, y_min, x_max, y_max]
    xs = box[0::2] # coordenadas x
    ys = box[1::2] # coordenadas y
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max, y_max]

'''
def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    if world_size < 2:
        return input_dict
    
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict'''

def reduce_dict(input_dict, average=True):
    # Si torch.distributed no está disponible o no se ha inicializado, simplemente devuelve el input
    if not torch.distributed.is_available():
        return input_dict

    if not torch.distributed.is_initialized():
        return input_dict

    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return input_dict

    with torch.inference_mode():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict



def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for samples, boxes, labels in metric_logger.log_every(data_loader, print_freq=10, header=header):
        samples = samples.to(device)
        
        # Convert boxes and labels to the format expected by DETR
        targets = []
        for box_list, label_list in zip(boxes, labels):
            # Convert string labels to indices
            label_indices = torch.tensor([CLASSES.index(label) for label in label_list], dtype=torch.long)
            # Convert boxes to tensor and normalize if needed
            #box_tensor = torch.tensor(box_list, dtype=torch.float32)
            # Convert boxes to tensor and normalize if needed
            if len(box_list) == 0:
                box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            else:
                box_tensor = torch.tensor([polygon_to_bbox(box) for box in box_list], dtype=torch.float32)
                if box_tensor.ndim == 1:
                    box_tensor = box_tensor.unsqueeze(0)  # shape [1, 4] si solo hay una caja
            # Create target dict
            target = {
                'boxes': box_tensor.to(device),
                'labels': label_indices.to(device)
            }
            targets.append(target)

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = {'loss_ce': 1, 'loss_bbox': bbox_loss_coef, 'loss_giou': giou_loss_coef}
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = loss_dict #reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                  for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def add_meter(self, name, meter):
        if name not in self.meters:
            self.meters[name] = meter
        else:
            raise ValueError(f"Meter with name {name} already exists.")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header is not None:
            print(header)
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue()
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        
        for obj in iterable:
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                print("{} {:{}d}/{} {}".format(header, i, len(str(len(iterable))), len(iterable), str(self)))
            i += 1
            end = time.time()

class SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value = self.global_avg)

def main():
    # Initialize distributed training if needed
    #if ddp:
    #    init_process_group(backend=backend)
        
    # Create model with proper initialization
    if init_from == 'scratch':
        model = create_model(num_classes=n_classes, pretrained=False)
    elif init_from == 'detr-resnet50':
        model = create_model(num_classes=n_classes, pretrained=True)
    elif init_from == 'resume':
        model = create_model(num_classes=n_classes, pretrained=False)
        checkpoint = torch.load(output_dir / 'checkpoint.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    model.to(device)
    
    #if ddp:
    #    model = DDP(model, device_ids=[ddp_local_rank])
    
    # Create matcher and criterion
    matcher = HungarianMatcher(
        cost_class=set_cost_class,
        cost_bbox=set_cost_bbox,
        cost_giou=set_cost_giou)
    
    criterion = DETRLoss(
        matcher=matcher,
        num_classes=n_classes,
        eos_coef=eos_coef)
    
    criterion.to(device)
    
    # Create optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                      if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters()
                      if "backbone" in n and p.requires_grad],
            "lr": lr * 0.1,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                weight_decay=weight_decay)
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
    
    # Create output directory
    output_dir = Path(out_dir)
    if master_process:
        os.makedirs(output_dir, exist_ok=True)
    
    # Resume training if requested
    start_epoch = 0
    if init_from == 'resume':
        checkpoint = torch.load(output_dir / 'checkpoint.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    
    print("Start training")
    start_time = time.time()
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        #if ddp:  # Set epoch for proper shuffling
        #    train_loader.sampler.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=grad_clip)
        
        lr_scheduler.step()
        
        if master_process:
            # Save checkpoint every epoch
            checkpoint_path = output_dir / 'checkpoint.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': config
            }, checkpoint_path)
            
            # Save a numbered checkpoint periodically
            if (epoch + 1) % 10 == 0:
                checkpoint_path = output_dir / f"checkpoint_{epoch:04}.pth"
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': config
                }, checkpoint_path)
            
            print(f"Epoch {epoch} training stats:")
            for k, v in train_stats.items():
                print(f"\t{k}: {v:.6f}")
    
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    #if ddp:
    #    destroy_process_group()

if __name__ == '__main__':
    main()
