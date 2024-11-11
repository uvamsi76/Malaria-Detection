from malaria_detection.data_preprocess import data_preprocessing
from malaria_detection.utils import transform,Malaria,LeNet
from malaria_detection.train_util import train
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="Process some inputs.")
parser.add_argument("--epochs", type=int, help="number of epochs")
parser.add_argument("--parasitized", type=str, help="path for parasitised training images")
parser.add_argument("--uninfected", type=str, help="path for uninfected training images")
args = parser.parse_args()

epochs=10
parasitized_paths=None
uninfected_paths=None
if(args.epochs):
    epochs=args.epochs
if(args.parasitized):
    parasitized_paths=args.parasitized
if(args.uninfected):
    uninfected_paths=args.uninfected

if(parasitized_paths!=None and uninfected_paths!=None):
    train_paths,val_paths=data_preprocessing(train_parasitized_paths=parasitized_paths,train_uninfected_paths=uninfected_paths)
else:
    train_paths,val_paths=data_preprocessing()
train_dataset = Malaria(train_paths, transform)
val_dataset = Malaria(val_paths, transform)
model=LeNet()
model= train(train_dataset,val_dataset,model,epochs,model_version="1.0.0",lr=1e-3)
print(model.eval())