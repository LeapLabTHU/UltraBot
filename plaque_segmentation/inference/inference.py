import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import DFMS
import cv2
from argparse import ArgumentParser

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFMS().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def preprocess_input(input_path):
    input = Image.open(input_path).convert('RGB').resize((256,256))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    input_tensor = transform(input).unsqueeze(0)  
    return input_tensor

def visualize(image_path, mask_path, vis_path, overlay_color=(255, 191, 0), alpha=0.4):
    kernel = np.ones((5, 5), np.uint8)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Error loading {image_path} or {mask_path}, skipping visualization.")
        return

    # Morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    if np.max(mask) == 0:
        print(f"Mask is completely black, skipping visualization for {image_path}.")
        return

    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = overlay_color

    mask_bin = mask > 0
    image[mask_bin] = cv2.addWeighted(image[mask_bin], 1 - alpha, colored_mask[mask_bin], alpha, 0)

    cv2.imwrite(vis_path, image)
    print(f"Saved visualization: {vis_path}")

def infer(model, device, image_path, det_path, save_mask_path, save_vis_path):
    img_tensor = preprocess_input(image_path).to(device)
    det_tensor = preprocess_input(det_path).to(device)

    with torch.no_grad():
        prediction = model([img_tensor, det_tensor], is_train=False)

    prob_map = prediction.data.squeeze(0).cpu().numpy()  
    if prob_map.ndim == 3:
        prob_map = prob_map[0]  
    binary_mask = (prob_map > 0.6).astype(np.uint8) * 255
    Image.fromarray(binary_mask).save(save_mask_path)
    print(f"Saved mask: {save_mask_path}")

    visualize(image_path, save_mask_path, save_vis_path)


def main():
    parser = ArgumentParser(description='Plaque Segmentation Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--det_path', type=str, required=True, help='Path to detection local input')
    parser.add_argument('--save_mask_path', default='./output', type=str, help='save mask path')
    parser.add_argument('--save_vis_path', default='./output', type=str, help='save vis path')
    args = parser.parse_args()

    model, device = load_model(args.model_path)
    infer(model, device, args.image_path, args.det_path, args.save_mask_path, args.save_vis_path)
    

if __name__ == "__main__":
    main()
