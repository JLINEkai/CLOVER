from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import torch
import pickle

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
pkl_path = '../PathVQA/pvqa/qas/test_vqa.pkl'
class PVQAdataset(Dataset):
    def __init__(self):
        # self.df = pd.read_csv(csv_path)
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
       
        normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        question = self.data[index]['sent']
        answer = list(self.data[index]['label'].keys())[0]
        img_path = os.path.join('../PathVQA/pvqa/images', 'test', self.data[index]['img_id'])+".jpg"
        return {
            "image": img_path,
            "question": question,
            "answer": answer,
        }
        # return {
        #     "image": img_path,
        #     "text_input": caption,
        #     "text_output": caption,
        # }
         
        
        
if __name__ == '__main__':
    test = PVQAdataset()
    print(test.__len__())
    print(test.__getitem__(0))
    print(test.__getitem__(1))
    
    
    