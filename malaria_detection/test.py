import torch
import argparse

from torch.utils.data import DataLoader
from malaria_detection.data_preprocess import data_preprocessing
from malaria_detection.train_util import get_accuracy
from malaria_detection.utils import Malaria,transform,LeNet

def test(test_parasitized_paths,test_uninfected_paths):
    device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_paths=data_preprocessing(train_parasitized_paths=test_parasitized_paths,train_uninfected_paths=test_uninfected_paths)
    test_dataset = Malaria(test_paths, transform)
    test_loader = DataLoader(test_dataset, batch_size = 32)
    model=torch.load('malaria_detection/models/LenetMalariaDetection.pth')
    accuracy=get_accuracy(model,test_loader,test_dataset,device)
    print(accuracy)
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("--parasitized", type=str, help="path for parasitised training images")
    parser.add_argument("--uninfected", type=str, help="path for uninfected training images")
    args = parser.parse_args()
    test_parasitized_paths="malaria_detection/data/testing_set/Parasitized"
    test_uninfected_paths="malaria_detection/data/testing_set/Uninfected"
    if(args.parasitized):
        test_parasitized_paths=args.parasitized
    if(args.uninfected):
        test_uninfected_paths=args.uninfected
    accuracy=test(test_parasitized_paths,test_uninfected_paths)
