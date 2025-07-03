import argparse
import torch
from PIL import Image
import torchvision
from torchvision import transforms as T
from pathlib import Path
from torch import nn
from tqdm import tqdm
import numpy as np

action_keys = ['u', 'i', 'o', 'j', 'k', 'l', '7', '8', '9', '4', '5', '6', 'x']
left, right, bottom, top = 520, 1144, 729, 163
action_info = ['backward', 'up', 'forward', 'left', 'down', 'right',
                       'roll-anti', 'pitch-up', 'roll', 'yaw-right', 
                       'pitch-down', 'yaw-left', 'stop']
action_key2action_idx = dict(zip(action_keys, range(len(action_keys))))
reverse_dict = {v:k for k,v in action_key2action_idx.items()}
action_dict = dict(zip(action_keys, action_info))


def get_corrects(y_true, y_pred, wrongs, path):
    corrects, total = 0, 0
    for y_t, y_p in zip(y_true, y_pred):
        if len(y_t) == 0:
            continue
        if y_p[0] in y_t:
            corrects += 1
        total += 1
    for y_t, y_p in zip(y_true, y_pred):
        if y_p[0] not in y_t:
            wrongs.append((path, y_p))
            break
    return corrects, total


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="artery_scan/test_set/stage1")
    parser.add_argument('--model_path', type=str, default="ckpts/stage1_allfold_checkpoint-10.pth.tar")
    parser.add_argument('--last_n', type=int, default=1, help='Last n samples to consider.')
    parser.add_argument('--offset', type=int, default=1, help='Last n samples to consider.')
    return parser.parse_args()


class DiscreteActor(nn.Module):
    def __init__(self, in_features, num_actions, dropout) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_actions),
        )
    def forward(self, x):
        return self.trunk(x)


def load_model(model_path):
    arch = 'resnet50'
    model = torch.load(model_path)
    model = torchvision.models.__dict__[arch]()
    in_features = model.fc.in_features
    actors = DiscreteActor(in_features, 13, 0)
    distancers = DiscreteActor(in_features, 6, 0)
    model.add_module("distancers", distancers)
    model.add_module("actors", actors)
    model.fc = nn.Identity()
    chkpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(chkpt['state_dict'], strict=True)
    print('model loaded')
    model.eval()
    return model

def get_labels(label_keys):
    labels = []
    for label_key in label_keys:
        if label_key in action_keys:
            labels.append(action_keys.index(label_key))
    return labels

def get_preds(batch, model, last_n, offset):
    normalize = T.Normalize(mean=[0.193, 0.193, 0.193], std =[0.224, 0.224, 0.224])
    trans = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
            normalize,
        ])
    image_tensors = []
    for image_path in batch:
        image = Image.open(image_path).convert("RGB")  
        image = image.crop((left, top, right, bottom)) 
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        tensor = trans(image)  
        image_tensors.append(tensor)
    samples = torch.stack(image_tensors)
    with torch.no_grad():
        all_logits = model.actors(model(samples))
    preds = []
    for i in range(offset, len(batch)):
        preds.append([torch.mode(all_logits[i-last_n:i].argmax(-1))[0].item()])
    labels = [get_labels(batch[0].split('/')[-1].split('-')[0])] * len(preds)
    print(preds)
    print(labels)
    return preds, labels

if __name__ == "__main__":
    args = get_args()
    model = load_model(args.model_path)
    image_paths = sorted([str(p) for p in Path(args.data_path).rglob('*.jpg')])
    batch_names = set()
    all_batches = []
    for path in image_paths:
        batch_name = '-'.join(path.split('/')[-1].split('.jpg')[0].split('-')[:-1]) + '@' +  path.split('/')[-1].split('.jpg')[1]
        if batch_name in batch_names:
            continue
        batch_names.add(batch_name)
        batch_paths = sorted(
            [str(p) for p in Path(args.data_path).rglob(f'{batch_name.split("@")[0]}*{batch_name.split("@")[1]}.jpg')],
            key = lambda x: int(x.split('/')[-1].split('.jpg')[0].split('-')[-1].split('_')[-1])
        )
        all_batches.append(batch_paths)
    corrects, total = 0, 0
    wrongs = []
    for batch in tqdm(all_batches):
        preds, labels = get_preds(batch, model, args.last_n, args.offset)
        print(batch[0])
        c, t = get_corrects(labels, preds, wrongs, batch[0])
        corrects += c
        total += t
    print(corrects / total)
    print(corrects)
    print(total)
