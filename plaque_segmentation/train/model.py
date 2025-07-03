import torch
import torch.nn.functional as F
from torch import nn
from resnext import ResNeXt101

class DFMS(nn.Module):  
    def __init__(self):
        super(DFMS, self).__init__()
        self.resnext = ResNeXt101()
        
        # Down-sampling layers for different stages
        self.down = nn.ModuleDict({
            'down1': self._down_layer(256),
            'down2': self._down_layer(512),
            'down3': self._down_layer(1024),
            'down4': self._down_layer(2048),
        })
        
        # Fusion layers
        self.fuse1 = self._fusion_layer(256)
        self.fuse2 = self._fusion_layer(128, final=True)
        self.fuse3 = self._fusion_layer(128)
        
        # Pre-convolution for input
        self.preconv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        
        # Spatial Feature Transform Layer
        self.SFT = SFTLayer(64)
        
        # Final prediction layer
        self.predict = nn.Conv2d(64, 1, kernel_size=1)

    def _down_layer(self, in_channels):
        """Creates a down-sampling block with convolution, batch norm, and PReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
    
    def _fusion_layer(self, in_channels, final=False):
        """Creates a fusion block."""
        layers = [
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            #nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        ]
        if final:
            layers[-1] = nn.Softmax2d()  # Apply softmax for final layer if required
        return nn.Sequential(*layers)

    def forward(self, x, is_train):
        x, bw = x  # Split input into image and background weight

        # Extract features from ResNeXt
        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)
        layer4 = self.resnext.layer4(layer3)

        # Down-sample features to match layer1 size
        down_features = {key: F.upsample(self.down[key](layer), size=layer1.size()[2:], mode='bilinear') 
                         for key, layer in zip(['down1', 'down2', 'down3', 'down4'], [layer1, layer2, layer3, layer4])}

        # Fuse the down-sampled features
        fuse1 = self.fuse1(torch.cat(tuple(down_features.values()), dim=1))

        # Create multiple branches using the fused features
        branches = {f'branch{i+1}': self.fuse2(torch.cat((fuse1, down_features[f'down{i+1}']), 1)) 
                    for i in range(4)}

        # Apply Spatial Feature Transform (SFT) to background weights
        bw = self.preconv(bw)
        sft_branches = {f'branch{i+1}': self.SFT([down_features[f'down{i+1}'], bw]) for i in range(4)}

        # Refine branches with attention
        refined_branches = {f'refine{i+1}': self.fuse3(torch.cat((branches[f'branch{i+1}'], sft_branches[f'branch{i+1}']), 1))
                            for i in range(4)}

        # Get predictions for each refined branch
        predictions = {f'predict{i+1}': self.predict(refined_branches[f'refine{i+1}']) for i in range(4)}

        # Upsample predictions to match input size
        predictions = {key: F.upsample(pred, size=x.size()[2:], mode='bilinear') for key, pred in predictions.items()}

        # Return the output during training or the averaged prediction during evaluation
        if is_train:
            return [pred for pred in predictions.values()]
        else:
            return torch.sigmoid(sum(predictions.values()) / 4)


class SFTLayer(nn.Module):
    def __init__(self, nfeat):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(nfeat, nfeat, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nfeat, nfeat, 1)
        self.SFT_shift_conv0 = nn.Conv2d(nfeat, nfeat, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nfeat, nfeat, 1)

    def forward(self, x):
        # Compute scale and shift for Spatial Feature Transform
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift
