import os
import json
import cv2
import numpy as np
from PIL import Image
from torch.utils import data

def make_dataset(root):
    imgs = []
    labels_dir = os.path.join(root, 'labels')
    for img_name in os.listdir(labels_dir):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(root, 'images', img_name)
        label_path = os.path.join(root, 'labels', img_name)
        json_path = os.path.join(root, 'json', f'{base_name}.json')
        imgs.append((img_path, label_path, json_path))
    return imgs

class ImageFolder_DFMS(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, json_path = self.imgs[index]
        
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')

        with open(json_path, 'r') as f:
            json_data = json.load(f)
 
        shape = json_data['shapes'][0]
        (x1, y1), (x2, y2) = shape['points']
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
        
        x1, x2 = sorted([int(round(x1)), int(round(x2))])
        y1, y2 = sorted([int(round(y1)), int(round(y2))])
        
        w, h = img.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        img_array = np.zeros((h, w, 3), dtype=np.uint8)  
        img_np = np.array(img) 
        img_array[y1:y2, x1:x2, :] = img_np[y1:y2, x1:x2, :]  
        img_local = Image.fromarray(img_array).convert('RGB')    
        img_local = img_local.resize((256, 256))  

        if self.joint_transform:
            img, target = self.joint_transform(img, target)
            
        if self.transform:
            img = self.transform(img)
            
        if self.target_transform:
            target = self.target_transform(target)
            img_local = self.target_transform(img_local)            

        return img, target, img_local
    
    def __len__(self):
        return len(self.imgs)