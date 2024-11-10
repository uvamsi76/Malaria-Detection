from malaria_detection.data_preprocess import data_preprocessing
from malaria_detection.utils import transform,Malaria,LeNet
from malaria_detection.train_util import train
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="Process some inputs.")
parser.add_argument("epochs", type=int, help="number of epochs")
args = parser.parse_args()

epochs=10
if(args.epochs):
    epochs=args.epochs


train_paths,val_paths=data_preprocessing()

train_dataset = Malaria(train_paths, transform)
val_dataset = Malaria(val_paths, transform)
model=LeNet()
model= train(train_dataset,val_dataset,model,epochs,model_version="1.0.0",lr=1e-3)
print(model.eval())