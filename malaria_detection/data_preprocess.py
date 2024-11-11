import os
import cv2
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def data_preprocessing(train_parasitized_paths='malaria_detection/data/training_set/Parasitized',train_uninfected_paths='malaria_detection/data/training_set/Uninfected'):
    train_paths_parasitized = []
    train_paths_uninfected = []

    train_paths_parasitized += os.listdir(train_parasitized_paths)
    train_paths_parasitized = ['malaria_detection/data/training_set/Parasitized/' + i for i in train_paths_parasitized]

    train_paths_uninfected += os.listdir(train_uninfected_paths)
    train_paths_uninfected = ['malaria_detection/data/training_set/Uninfected/' + i for i in train_paths_uninfected]

    paths = train_paths_parasitized + train_paths_uninfected

    c=0
    l=[]
    for i in range(len(paths)):
        # print(i)
        image = cv2.imread(paths[i])
        if(image is None):
            l.append(i)
    for i in l:
        paths.pop(i-c)
        c+=1

    random.shuffle(paths)
    return paths