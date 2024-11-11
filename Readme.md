### Building a malaria detection model using Lenet architecture

- Create a `venv` and install all required packages from `requirements.txt`
run  
```
python -m venv venv

.\venv\Scripts\activate 
or
venv/bin/activate

pip install -r ./requirements.txt
``` 
- To run malaria_detection model to see if the blood sample you have is parasitised or uninfected run

```
python -m malaria_detection.predict --path single_prediction\Parasitised.png
```

- we can give the image we have by passing the path to image after --path 


- To train model from scratch we can run
```
python -m malaria_detection.train --epochs 2 --parasitized malaria_detection/data/testing_set/Parasitized --uninfected malaria_detection/data/testing_set/Uninfected
```

- we can train on custome data by passing parasitized training images and uninfected training images path seperately after --parasitized  --uninfected