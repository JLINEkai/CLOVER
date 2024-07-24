from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import torch
import pickle
import json
from lavis.datasets.datasets.base_dataset import BaseDataset
import random
class ImageTextContrastiveCollator:
    def __init__(self):
        return
    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['image'].append(data['image'])
            inputs['question'].append(data['question'])
            inputs['answer'].append(data['answer'])
        
        return inputs
    
PATH = "./instruction_data_15k_a.json"

class QuiltVQAdataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, ann_paths, vis_root):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        with open(PATH, "r", encoding="utf-8") as f:
            self.data = json.load(f)  
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        question = self.data[index]['question']
        question = "Now that you are a pathologist, please answer the following questions based on the images. " + question
        answer = self.data[index]['answer']
        img_path = self.data[index]['path']

        img_path = os.path.join("../quilt_1m_path", img_path)
                    
        image = Image.open(img_path).convert('RGB')
        image = self.vis_processor(image)
        answer = self.text_processor(answer)
        question = self.text_processor(question)
        
        
        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
        }

         
        
    
    
    