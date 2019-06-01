import os

import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data

# target transform
class ScoreNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, score, weight):
        score = (score - self.mean) / self.std
        
        return score, weight

def collate(samples):
    imgs = []
    scores = []
    weights = []
    
    for sample in samples:
        imgs.append(sample[0])
        scores.append(sample[1][0])
        weights.append(sample[1][1])
    
    imgs = torch.stack(imgs)
    scores = torch.tensor(scores, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)
    
    # imgs is a tensor, while annos is a tuple (scores, weights)
    return imgs, (scores, weights)

class ScoreDataset(data.Dataset):
    def __init__(self, root_path, crop_set='crop', split_name='score-1', phase='train', transforms=None, target_transforms=None, weight=False):
        self.img_path = os.path.join(root_path, crop_set)

        if phase == 'train' and weight == True:
            self.anno_file = os.path.join(root_path, '{}_weight_{}.csv'.format(
                split_name,
                phase
            ))
        else:
            self.anno_file = os.path.join(root_path, '{}_{}.csv'.format(
                split_name,
                phase
            ))
        self.anno_df = pd.read_csv(self.anno_file)
        
        self.weight = weight
        self.transforms = transforms
        self.target_transforms = target_transforms
        
    def __getitem__(self, index):
        anno = self.anno_df.iloc[index]
        
        image_file = os.path.join(
            self.img_path,
            '{}_{}.jpg'.format(anno['id'], anno['serial'])
        )
        
        img = Image.open(image_file).convert('RGB') # TODO : check channel
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        score = anno['score']
        if self.weight:
            weight = anno['weight']
        else:
            weight = 1.
        
        if self.target_transforms is not None:
            score, weight = self.target_transforms(score, weight)
            
        return img, (score, weight)

    def __len__(self):
        return len(self.anno_df)
    