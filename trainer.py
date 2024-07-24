import os
from typing import List, Dict, Type
import math
import torch
from torch.optim import Optimizer
import transformers
import torch.nn as nn
from PathBLIP.dataset import ImageTextContrastiveCollator
from tqdm import tqdm
WEIGHTS_NAME = "pytorch_model.bin"

class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None):
        pass

    def train(self,
        model,
        train_dataset,
        eval_dataset,
        local_rank,
        epochs: int = 1,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        output_path: str = './checkpoints/vision_text_pretrain',
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.accumulation_steps = accumulation_steps
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        train_collate_fn = ImageTextContrastiveCollator()        
        
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False, pin_memory=True, num_workers=4,
                                                   batch_size=36, drop_last=True, collate_fn=train_collate_fn, sampler=train_sampler)

        steps_per_epoch = len(dataloader)
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        # Prepare optimizers
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)


        skip_scheduler = False
        for epoch in range(epochs):
            data_iterator = iter(dataloader)
            with tqdm(total=steps_per_epoch, disable= local_rank != 0) as pbar:
                for train_iter in range(steps_per_epoch):
                    model.zero_grad()
                    model.train()              
                    data = next(data_iterator)
                    data['image'] = data['image'].cuda()
                    if use_amp:
                        with autocast():
                            loss = model(data)
                        loss_value = loss['loss']
                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss = model(data)
                        loss_value = loss['loss'] / self.accumulation_steps
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                    # pbar.set_postfix('Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}'.format(epoch,epochs,train_iter,steps_per_epoch,loss_value))  # 输入一个字典，显示实验指标
                    # tqdm.write('Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}'.format(epoch,epochs,train_iter,steps_per_epoch,loss_value), end="\r")
                    pbar.set_description(f"Epoch {epoch+1}/{epochs}, Batch {train_iter}/{steps_per_epoch} - Loss: {loss_value:.4f}")
                    pbar.update(1)  # 更新进度条
                    # pbar.update(1)
                    # print('Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}'.format(epoch,epochs,train_iter,steps_per_epoch,loss_value))
                    

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()
            if local_rank == 0:        
                self._save_ckpt(model,epoch,output_path)


    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _save_ckpt(self, model, epoch, save_dir):
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, 'epoch{}.pth'.format(epoch)))
        print("Save: ", os.path.join(save_dir, 'epoch{}.pth'.format(epoch)))