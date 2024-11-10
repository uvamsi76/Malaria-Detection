import cv2
import torch
from PIL import Image
from malaria_detection.utils import transform,LeNet,round
import argparse
import warnings


def predict(image_path):
    pytorch_model = torch.load('malaria_detection/models/LenetMalariaDetection.pth')
    pytorch_model.eval()
    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = cv2.imread(image_path)
    image_pil = Image.fromarray(image)
    image = transform(image_pil)
    image=image.detach().numpy()
    image = image[None, :, :, :]
    image=torch.tensor(image)
    image = image.to(device)
    output=pytorch_model(image)
    output=round(output)
    print(output)
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("path", type=str, help="path for input image")
    args = parser.parse_args()
    path='malaria_detection/data/single_prediction/Parasitised.png'
    if(args.path):
        path=args.path
    else:
        print('taking default path for image')
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    predict(path)