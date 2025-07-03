import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from argparse import ArgumentParser
import pdb
from functools import partial

num_points = 15
average_position = torch.tensor([
    [0.0000, 0.1503],
    [0.2500, 0.1507],
    [0.5000, 0.1504],
    [0.7500, 0.1490],
    [1.0000, 0.1468],
    [0.0000, 0.7490],
    [0.2500, 0.7468],
    [0.5000, 0.7446],
    [0.7500, 0.7426],
    [1.0000, 0.7407],
    [0.0000, 0.7931],
    [0.2500, 0.7915],
    [0.5000, 0.7897],
    [0.7500, 0.7876],
    [1.0000, 0.7848]
], dtype=torch.float32)

class IdentityModule(torch.nn.Module):
    def forward(self, x):
        return x

class PointPredHead(nn.Module):
    def __init__(self, in_features, num_points, dropout, arch, std) -> None:
        super().__init__()
        self.arch = arch
        self.std = std
        if 'resnet' in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_points // 3),
                nn.Tanh()
            )
        elif 'convnext' in arch:
            self.trunk = nn.Sequential(
                nn.Dropout(dropout),
                nn.Flatten(1),
                partial(nn.LayerNorm, eps=1e-6)(in_features),
                nn.Linear(in_features, num_points // 3),
                nn.Tanh()
            )
        else:
            raise NotImplementedError

        self.init_param()

    def init_param(self):
        if 'resnet' in self.arch:
            nn.init.normal_(self.trunk[1].weight, std=self.std)
            nn.init.zeros_(self.trunk[1].bias)
        elif 'convnext' in self.arch:
            nn.init.normal_(self.trunk[3].weight, std=self.std)
            nn.init.zeros_(self.trunk[3].bias)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.trunk(x)

class Three_model(nn.Module):
    def __init__(self, arch) -> None:
        super().__init__()
        self.model_intimaup = self.load_single_model(arch)
        self.model_intimadown = self.load_single_model(arch)
        self.model_media = self.load_single_model(arch)

    def load_single_model(self, arch):
        if 'resnet' in arch:
            model = models.__dict__[arch](weights="IMAGENET1K_V2")
            in_features = model.fc.in_features
            model.fc = nn.Identity()
        elif 'convnext' in arch:
            model = models.__dict__[arch](weights="IMAGENET1K_V1")
            in_features = model.classifier[2].in_features
            model.classifier = IdentityModule()
        else:
            raise NotImplementedError

        actors = PointPredHead(in_features, num_points, 0.0, arch, 0.001)
        model.add_module("actors", actors)
        return model

    def forward(self, x):
        intimaup_feature = self.model_intimaup(x)
        intimadown_feature = self.model_intimadown(x)
        media_feature = self.model_media(x)

        feature = torch.cat((intimaup_feature, intimadown_feature, media_feature), 1)
        
        intimaup = self.model_intimaup.actors(intimaup_feature)
        intimadown = self.model_intimadown.actors(intimadown_feature)
        media = self.model_media.actors(media_feature)

        pred_offset = torch.cat((intimaup, intimadown, media), 1)
        
        return feature, pred_offset

def create_normalizer(normalizer_type='imagenet'):
    if normalizer_type == 'imagenet':
        return transforms.Normalize(
            mean=[0.193, 0.193, 0.193],
            std=[0.224, 0.224, 0.224]
        )
    elif normalizer_type == 'stat':
        return transforms.Normalize(
            mean=[0.18136882, 0.18137674, 0.18136712],
            std=[0.1563932, 0.1563886, 0.15638869]
        )
    elif normalizer_type == 'original_imagenet':
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        raise NotImplementedError

def load_model(arch, pretrained_path):
    model = Three_model(arch)
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    model_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_state_dict}
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    return model

def load_image(image_path):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def preprocess_image(image, input_size=256, normalizer_type='imagenet'):
    transform_fns = [
        transforms.Resize((input_size, input_size), 
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        create_normalizer(normalizer_type),
    ]
    transform = transforms.Compose(transform_fns)
    return transform(image)

def read_json(json_path):
    with open(json_path, 'r') as file:
        return json.load(file)

def predict(image_path, json_path, model, device='cuda'):
    img = load_image(image_path)
    image_tensor = preprocess_image(img).unsqueeze(0).to(device)
    
    data = read_json(json_path)
    image_height = data["imageHeight"]
    image_width = data["imageWidth"]
    
    model.eval()
    with torch.no_grad():
        features, pred_offsets = model(image_tensor)
    
    x_coords = torch.zeros(num_points).unsqueeze(0).to(device)
    x_coords = x_coords.expand(features.shape[0], -1).detach()
    pred_offsets = torch.stack([x_coords, pred_offsets], dim=2).reshape(features.shape[0], -1)
    pred_points = pred_offsets + average_position.unsqueeze(0).reshape(-1, num_points * 2).to(device)
    
    pred_points = pred_points.reshape(-1, num_points, 2)
    pred_points[:, :, 0] *= image_width
    pred_points[:, :, 1] *= image_height
    pred_intimaup = pred_points[:, :5, :]
    pred_intimadown = pred_points[:, 5:10, :]
    pred_media = pred_points[:, 10:15, :]
    
    return pred_intimaup, pred_intimadown, pred_media, image_width, image_height, data

def visualize_and_save(image_path, json_path, pred_intimaup, pred_intimadown, pred_media, 
                       image_width, image_height, data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    dir_name = os.path.basename(os.path.dirname(json_path))
    base_name = os.path.basename(json_path).replace('.json', '')
    output_img_path = os.path.join(output_dir, f"{dir_name}_{base_name}_prediction.jpg")
    output_txt_path = os.path.join(output_dir, f"{dir_name}_{base_name}_data.txt")
    
    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0) 

    cpu_pred_intimaup = pred_intimaup.cpu().numpy()[0]
    cpu_pred_intimadown = pred_intimadown.cpu().numpy()[0]
    cpu_pred_media = pred_media.cpu().numpy()[0]
    
    # intimaup
    for i in range(1, len(cpu_pred_intimaup)):
        line = [cpu_pred_intimaup[i-1, :], cpu_pred_intimaup[i, :]]
        ax.plot(*zip(*line), color='green', linewidth=2.5)
    ax.scatter(cpu_pred_intimaup[:, 0], cpu_pred_intimaup[:, 1], s=40, color='blue')
    
    # intimadown
    for i in range(1, len(cpu_pred_intimadown)):
        line = [cpu_pred_intimadown[i-1, :], cpu_pred_intimadown[i, :]]
        ax.plot(*zip(*line), color='orange', linewidth=2.5)
    ax.scatter(cpu_pred_intimadown[:, 0], cpu_pred_intimadown[:, 1], s=40, color='red')
    
    # media
    for i in range(1, len(cpu_pred_media)):
        line = [cpu_pred_media[i-1, :], cpu_pred_media[i, :]]
        ax.plot(*zip(*line), color='purple', linewidth=2.5)
    ax.scatter(cpu_pred_media[:, 0], cpu_pred_media[:, 1], s=40, color='green')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    with open(output_txt_path, "w") as f:
        f.write("pred intimaup \n")
        for point in cpu_pred_intimaup:
            f.write(f"{point[0]}, {point[1]}\n")
            
        f.write("pred intimadown \n")
        for point in cpu_pred_intimadown:
            f.write(f"{point[0]}, {point[1]}\n")
            
        f.write("pred media \n")
        for point in cpu_pred_media:
            f.write(f"{point[0]}, {point[1]}\n")
            
def main():
    parser = ArgumentParser(description='Biometric Measurement Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--json', type=str, required=True, help='Path to annotation JSON')
    parser.add_argument('--arch', default='resnet50', help='Model architecture')
    parser.add_argument('--pretrain', required=True, type=str, help='Path to pretrained model')
    parser.add_argument('--input-size', default=256, type=int, help='Input image size')
    parser.add_argument('--normalizer', default='imagenet', type=str, help='Normalization method')
    parser.add_argument('--output-dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    
    model = load_model(args.arch, args.pretrain).to(device)
    
    preds = predict(args.image, args.json, model, device)

    visualize_and_save(args.image, args.json, *preds, args.output_dir)

if __name__ == '__main__':
    main()
