from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import torch
import pickle
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
            

        # inputs['image'] = torch.stack(inputs['image'])

        return inputs

IMG_PATH = "../MIL/CLAM-master"

# test_negative_files = ['N202409303001.ibl', "N202410335001.ibl", 'N202410282002.ibl', 'N202409292001.ibl', "N202410275001.ibl", "N202410264002.ibl", "N202410272001.ibl", "N202409316001.ibl","N202410284003.ibl", "N202410265002.ibl", "N202410283001.ibl", "N202410315001.ibl", "N202410266002.ibl"]
# test_positive_files = [ "N202409298003.ibl",  'N202410307001.ibl', 'N202410296001.ibl', 'N202410300001.ibl', 'N202409294001.ibl']

test_positive_files_int = ['N202410318001.ibl', 'N202410285002.ibl', 'N202410307001.ibl', 'N202410284001.ibl', 'N202410293002.ibl', 'N202410305001.ibl', 'N202410302001.ibl', 'N202409289001.ibl', 'N202410286001.ibl'] 
test_positive_files_sto = ['NULL-20240422-144501.ibl', 'NULL-20240422-143615.ibl']
test_negative_files_int = ['N202409296001.ibl', 'N202410304001.ibl', 'N202410310001.ibl', 'N202409317001.ibl', 'N202410285001.ibl', 'N202409293001.ibl', 'N202410283002.ibl', 'N202410282002.ibl']
test_negative_files_sto = ['N202410267001.ibl', 'N202410277001.ibl', 'N202410271001.ibl', 'N202410276001.ibl']


train_positive_int = ['N202409294001.ibl']
train_negative_int = ['N202410316001.ibl']
train_positive_sto = ['N202410334001.ibl']
train_negative_sto = ['N202410273001.ibl']


class STOdataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, ann_paths, vis_root):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.data_img = []
        self.data_label = []
        for x in os.listdir(os.path.join(IMG_PATH, "positive")): 
            # if x not in test_positive_files_int and x not in test_positive_files_sto:
            # if x in test_positive_files_int:
            if x in test_positive_files_sto:
            # if x in train_positive_int or x in train_positive_sto:
                for img in os.listdir(os.path.join(IMG_PATH, "positive", x)):

                    
                    self.data_img.append(os.path.join(IMG_PATH, "positive", x, img))
                    self.data_label.append(1)
        print("positive len*****************", len(self.data_img))
        
        
        for x in os.listdir(os.path.join(IMG_PATH, "negative")):
            # if x not in test_negative_files_int and x not in test_negative_files_sto:
            # if x in test_negative_files_int:
            if x in test_negative_files_sto:
            # if x in train_negative_int or x in train_negative_sto:
            
                for img in os.listdir(os.path.join(IMG_PATH, "negative", x)):
                    self.data_img.append(os.path.join(IMG_PATH, "negative", x, img))
                    self.data_label.append(0)
        print("all len*****************", len(self.data_img))
        

        
        
    def __len__(self):
        return len(self.data_img)
    def __getitem__(self, index):
        # question = self.data[index]['sent']
        question_lists = ['Is this pathological image showing a negative or positive result?', 'Does this pathological image indicate a negative or positive outcome?', 
                'Can you tell if this pathological image is negative or positive?', 'Is the result in this pathological image negative or positive?', 
                'Does this image of pathology suggest something negative or positive?', 'Is this pathological image reflecting a positive or negative result?',
                'Does this image show a positive or negative pathology?', 'Is the pathology in this image considered negative or positive?',
                'Can you determine whether this pathological image is positive or negative?', 'Is this pathological image revealing a positive or negative condition?'
                ]
        question = random.choice(question_lists)
        question = "Now that you are a pathologist, please answer the following questions based on the images. " + question_lists[0]

        if self.data_label[index] == 0:
            answer = "this is a negative pathological image"
        else:
            answer = "this is a positive pathological image"
            

        img_path = self.data_img[index]    
        image = Image.open(img_path).convert('RGB')
        image = self.vis_processor(image)
        answer = self.text_processor(answer)
        question = self.text_processor(question)
        
        
        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
        }
         
        
        
if __name__ == '__main__':
    
    
    question = ['Is this pathological image showing a negative or positive result?', 'Does this pathological image indicate a negative or positive outcome?', 
                'Can you tell if this pathological image is negative or positive?', 'Is the result in this pathological image negative or positive?', 
                'Does this image of pathology suggest something negative or positive?', 'Is this pathological image reflecting a positive or negative result?',
                'Does this image show a positive or negative pathology?', 'Is the pathology in this image considered negative or positive?',
                'Can you determine whether this pathological image is positive or negative?', 'Is this pathological image revealing a positive or negative condition?'
                ]
    # test = PVQAdataset()
    print(random.choice(question))
    # print(test.__len__())
    # print(test.__getitem__(0))
    # print(test.__getitem__(1))
    
    
    