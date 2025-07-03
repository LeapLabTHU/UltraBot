import os
import json
import argparse
import logging
import datetime
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image, ImageDraw
from torchmetrics.detection import MeanAveragePrecision
from sklearn.metrics import precision_recall_curve
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Carotid Plaque Detection Training')
    parser.add_argument('--epoch_num', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--test-batch-size', type=int, default=1, help='Validation batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--lr-step-size', type=int, default=3, help='LR scheduler step size')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='LR decay factor')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device_id', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--detetion_data_root', type=str, required=True, help='Path to detetion data')
    parser.add_argument('--output_dir', type=str, default='detection_output', help='Output directory')
    parser.add_argument('--save-freq', type=int, default=5, help='Validation/save frequency in epochs')
    return parser.parse_args()

def setup_logger(output_dir):
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"training_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return log_path

class CarotidDataset(Dataset):
    """Dataset class for carotid plaque detection"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Load annotations
        json_path = os.path.join(self.root_dir, img_name.replace('.jpg', '.json'))
        with open(json_path, 'r') as f:
            annotations = json.load(f)

        # Parse bounding boxes
        boxes = []
        for shape in annotations['shapes']:
            points = shape['points']
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            boxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, {'boxes': boxes, 'labels': labels}

def create_model(num_classes=2):
    """Create Faster R-CNN model with ResNet-50 backbone"""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_epoch(model, optimizer, loader, device, epoch):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        
        if batch_idx % 10 == 0:
            logging.info(f'Epoch [{epoch}] Batch [{batch_idx}/{len(loader)}] Loss: {losses.item():.4f}')

    return total_loss / len(loader)

def validate(model, dataset, device, epoch, output_dir):
    """Validate model performance"""
    model.eval()
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type="bbox",
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
        max_detection_thresholds=[1, 10, 100],
        class_metrics=True
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        for idx in range(len(dataset)):
            # Load and process image
            img_name = dataset.image_files[idx]
            raw_image = Image.open(os.path.join(dataset.root_dir, img_name)).convert("RGB")
            image_tensor = transform(raw_image).unsqueeze(0).to(device)
            
            # Generate predictions
            predictions = model(image_tensor)[0]
            
            # Load ground truth
            with open(os.path.join(dataset.root_dir, img_name.replace('.jpg', '.json')), 'r') as f:
                annotations = json.load(f)
            
            # Process annotations
            true_boxes = []
            for shape in annotations['shapes']:
                points = shape['points']
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                true_boxes.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])

            # Update metrics
            metric.update(
                preds=[{
                    'boxes': predictions['boxes'].cpu(),
                    'scores': predictions['scores'].cpu(),
                    'labels': predictions['labels'].cpu()
                }],
                target=[{
                    'boxes': torch.tensor(true_boxes, dtype=torch.float32),
                    'labels': torch.ones((len(true_boxes),), dtype=torch.int)
                }]
            )

    results = metric.compute()
    logging.info(f"\nValidation Metrics @ Epoch {epoch}:")
    logging.info(f"mAP@[0.5:0.95]: {results['map'].item():.4f}")
    logging.info(f"mAP@50: {results['map_50'].item():.4f}")
    logging.info(f"mAP@75: {results['map_75'].item():.4f}")
    logging.info(f"Average Recall: {results['mar_100'].item():.4f}")
    logging.info(f"Class Precision: {results['map_per_class'].item():.4f}")
    metric.reset()
    return results

def main():
    args = parse_args()
    
    # Set device
    torch.cuda.set_device(args.device_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = setup_logger(args.output_dir)
    logging.info(f"Training parameters: {vars(args)}")
    
    # Create datasets and loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    training_root = os.path.join(args.detetion_data_root, 'train', 'images')
    testing_root = os.path.join(args.detetion_data_root, 'test', 'images')
    train_dataset = CarotidDataset(training_root, transform)
    test_dataset = CarotidDataset(testing_root, transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # Initialize model
    model = create_model().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.lr_step_size, 
        gamma=args.lr_gamma
    )

    # Training loop
    for epoch in range(1, args.epoch_num+1):
        # Train
        avg_loss = train_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()
        logging.info(f'Epoch [{epoch}] Average Loss: {avg_loss:.4f}')
        
        # Validate and save
        if epoch % args.save_freq == 0:
            validate(model, test_dataset, device, epoch, args.output_dir)
            torch.save(model.state_dict(), 
                      os.path.join(args.output_dir, f'model_epoch_{epoch}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_final.pth'))
    logging.info(f"Training completed. Final model saved to {args.output_dir}")

if __name__ == "__main__":
    main()