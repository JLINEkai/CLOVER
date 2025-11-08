import pdb, os
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from PathBLIP.dataset import Quiltdataset, ImageTextContrastiveCollator
from lavis.models import load_model
from trainer import Trainer

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# set cuda devices
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
# device = "cuda:0,1,2,3" if torch.cuda.is_available() else "cpu"


train_config = {
    'num_epochs': 20,
    'warmup': 0.1,
    'lr': 2e-5,
    'weight_decay': 1e-4,
    'eval_batch_size': 8,
    'eval_steps': 1000,
    'save_steps': 1000,
}

train_dataset = Quiltdataset("../BLIP/LAVIS-main/quilt.csv")
train_collate_fn = ImageTextContrastiveCollator()
train_dataloader = DataLoader(train_dataset,
    batch_size=8,
    collate_fn=train_collate_fn,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True
    )

val_dataset = Quiltdataset("../test_samples.csv")
val_collate_fn = ImageTextContrastiveCollator()

val_dataloader = DataLoader(val_dataset,
    batch_size=4,
    collate_fn=val_collate_fn,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    )

# parser = argparse.ArgumentParser(description='training')
# parser.add_argument('--local_rank', type=int, help='local rank for dist')
# args = parser.parse_args()
torch.distributed.init_process_group(backend='nccl')
# print(args.local_rank)
# world_size = torch.cuda.device_count()
local_rank = torch.distributed.get_rank()
# print(local_rank)
torch.cuda.set_device(local_rank)


# parser.add_argument("--local-rank", type=int)
# args = parser.parse_args()
# if 'LOCAL_RANK' not in os.environ:
#     os.environ['LOCAL_RANK'] = str(args.local_rank)

model = load_model("blip2", "pretrain", checkpoint="../BLIP/blip2_pretrained.pth")
    # model.load_state_dict(torch.load('./checkpoints/vision_text_pretrain/t5/epoch10.pth',map_location='cpu'),strict=False)
model.cuda()
model_save_path = f'../BLIP/LAVIS-main/checkpoints/VL'
trainer = Trainer()
trainer.train(
    model,
    train_dataset,
    val_dataset,
    local_rank,
    warmup_ratio=train_config['warmup'],
    epochs=train_config['num_epochs'],
    optimizer_params={'lr':train_config['lr']},
    output_path=model_save_path,
    weight_decay=train_config['weight_decay'],
    use_amp=True,
    )


    