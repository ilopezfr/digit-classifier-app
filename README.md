#  Digit Classifier App


## Instructions
1. Develop a NN training pipeline for MNIST handwritten digit classification. 
2. Implement an inference server with the trained model. The server should accept raw images as input and preduct the number with a probability. 


## Requirements

- Python 3
- Tensorflow 
- Keras
- Flask 

## Instructions

This repository contains the source code for a Python Flask Server application that allows one to upload an image of a handwritten digit and it returns the predicted digit. 


Clone the repo, install the dependencies, and download the model weights. 

```bash
git clone https://github.com/ilopezfr/digit-classifier-app.git
```

Create virtual environment and activate it:
```bash
python3 -m venv venv
. venv/bin/activate
```
Install dependencies:
```bash
cd digit-classifier-app
pip install -r requirements.txt
```

Download the model.h5 and save it into model folder
```bash
wget "https://drive.google.com/open?id=15ij4G9nYEb74CqhqooXRIyWjfJrqDmey" -P /model
```


## Overview of the Code
The code consists of X Python scripts and the file config.json that contains various parameter settings. 

```bash
├── model
|    ├── predict.py
|    ├── train.py
|    ├── model.h5
├── templates
|    ├── error.html
|    ├── index.html
|    ├── predict.html
├── app.py
```

## Train the model
A trained model file on MNIST dataset is already offered and can be directly downloaded from [this link](https://drive.google.com/open?id=15ij4G9nYEb74CqhqooXRIyWjfJrqDmey). However, should you want to rebuild the model using the same architecture, you can run the following script:

```bash
python model/train.py
```



## Start the server application and predict on a new image.
Flask allows you to serve an image of a handwritten digit with the server and get a prediction. It currently accepts images with the extensions 'png', 'jpg' and 'jpeg'.

Before starting the server, make sure have a `model.h5` file saved in the `/model` folder.
```bash
$ python app.py
* Serving Flask app "app" (lazy loading)
...
* Running on http://127.0.0.1:5002/
```
You can now access the REST API via http://127.0.0.1:5002.


