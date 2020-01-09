# Intel Image Classification
## Training
1. Change hyperparameters in CONST.py
2. Download intel-image-classification image to the directory
```
  python3 model.py
```
* train the model and save as model.h5

## Predicting an image
```
  python3 predict.py {h5 model path(model.h5)} {image path}
```
* predict an image

### Example
```
    python3 predict.py model.h5 demo/building.jpg
```

<img src="https://raw.githubusercontent.com/jsong336/README/master/Intel-Image-Classification/building.jpg"/>
<br/>
building

### Config
<i>Current configuration on CONST.py: epoch 1, number of image: 2000. Please change configuration that fits your hardware. Current config trains the model up to around 82% validation accuracy which leaves limitation of the model</i>
```
    python3 predict.py model.h5 demo/glacier-looking-sea.jpg
```

<img src="https://raw.githubusercontent.com/jsong336/README/master/Intel-Image-Classification/glacier-like-sea.jpg"/>
<br/>
glacier



## Class 
```
  python3 predict.py -l
```
<ul>
  <li>Building</li>
  <li>Forest</li>
  <li>Glacier</li>
  <li>Mountain</li>
  <li>Sea</li>
  <li>Street</li>
</ul>

## Requirement
<ul>
  <li>python 3.7</li>
  <li>tensorflow 2.0</li>
  <li>numpy</li>
</ul>

## Data source
https://www.kaggle.com/puneet6060/intel-image-classification
