from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import torch
import pickle
from lavis.datasets.datasets.base_dataset import BaseDataset

class ImageTextContrastiveCollator:
    def __init__(self):
        return
    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['image'].append(data['image'])
            inputs['question'].append(data['question'])
            inputs['answer'].append(data['answer'])
            

        # inputs['image'] = torch.stack(inputs['image'])

        return inputs

pkl_path = '../PathVQA/pvqa/qas/test_open_qa.pkl'



class PVQAdataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, ann_paths, vis_root):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)

        
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        question = self.data[index]['sent']
        question = "Now that you are a pathologist, please answer the following questions based on the images. "+question
        # question = question + "please output yes or no."
        
        # question = "Now that you are a pathologist, please answer the following questions based on the images. " + question
        answer = list(self.data[index]['label'].keys())[0]
        img_path = os.path.join('../PathVQA/pvqa/images', 'test', self.data[index]['img_id'])+".jpg"
            
        image = Image.open(img_path).convert('RGB')
        image = self.vis_processor(image)
        answer = self.text_processor(answer)
        question = self.text_processor(question)
        
        
        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
        }

         
        

    
    
    