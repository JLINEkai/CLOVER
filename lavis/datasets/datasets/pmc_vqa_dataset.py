from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import torch
import pickle
import json
import io
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


        return inputs


#####################################
class PMCVQAdataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, ann_paths, vis_root):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        

        df = pd.read_parquet("../train-00000-of-00001-e5107276f24d7201.parquet")
        self.data =  df[df["answer_type"] == "CLOSED"]

        
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        question = self.data.iloc[index]['question']

        question = "Now that you are a pathologist, please answer the following questions based on the images. please only answer yes or no." + question

        answer = self.data.iloc[index]['answer']

        
        image = Image.open(io.BytesIO(self.data.iloc[index]['image']['bytes']))
        
        image = self.vis_processor(image)
        answer = self.text_processor(answer)
        question = self.text_processor(question)
        
        
        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
        }
        


    
    
    