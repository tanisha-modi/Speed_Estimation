import os.path as osp
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

def load_model(model_path, device):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model

def process_image(img_path, model_path='models/RRDB_ESRGAN_x4.pth', device=torch.device('cpu')):
    # Load model
    model = load_model(model_path, device)

    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    return output

# Example usage (if you want to test this file directly)
if __name__ == "__main__":
    result = process_image('path_to_your_image.jpg')
    cv2.imwrite('results/result.png', result)
