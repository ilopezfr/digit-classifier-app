#  Digit Classifier App


## Instructions
1. Develop a NN training pipeline for MNIST handwritten digit classification. 
2. Implement an inference server with the trained model. The server should accept raw images as input and preduct the number with a probability. 


## Requirements


## Instructions

This repository contains the source code for a Flask application that allows one to upload an image of a handwritten digit and it returns the predicted digit. 


Clone the repo, create a virtual environment, then install the dependencies. 

```bash
git clone https://github.com/ilopezfr/digit-classifier-app
cd digit-classifier-app
pip install -r requirements.txt

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



## Run the Flask app and predict on a new image.
Flask allows you to serve an image of a handwritten digit with the server and get a prediction.



