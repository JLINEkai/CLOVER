from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import torch

class ImageTextContrastiveCollator:
    def __init__(self):
        return
    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['image'].append(data['image'])
            inputs['text_input'].append(data['text_input'])
            inputs['text_output'].append(data['text_output'])
            

        # inputs['image'] = torch.stack(inputs['image'])

        return inputs

class Quiltdataset(Dataset):
    def __init__(self):
        # self.df = pd.read_csv(csv_path)
        self.df = pd.read_csv('../BLIP/LAVIS-main/quilt.csv')
        self.df = self.df.dropna(axis=0, subset=['pathology'])[400000:]
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
        return len(self.df)
    def __getitem__(self, index):
        caption = self.df.iloc[index]['caption']
        if type(caption) == float:
            caption = "This is a image about the pathology."
        img_path = self.df.iloc[index]['image_path']
        # img_path = os.path.join("../", img_path)
        # image = Image.open(img_path).convert('RGB')
        # image = self.transform(image)
        # caption = self.text_processor(caption)
        # img = self.transform(img)
        
        caption = caption.split()
        prefix = caption[:int(len(caption) * 0.2)]
        subfix = caption[int(len(caption) * 0.2):]
        prefix = " ".join(prefix)
        subfix = " ".join(subfix)
        return {
            "image": img_path,
            "text_input": prefix,
            "text_output": subfix,
        }
        # return {
        #     "image": img_path,
        #     "text_input": caption,
        #     "text_output": caption,
        # }
         
        
        
if __name__ == '__main__':
    test = Quiltdataset()
    print(test.__len__())
    print(test.__getitem__(0))
    print(test.__getitem__(1))
    
    
    