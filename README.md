#  Digit Classifier App


## Challenge Request:
1. Develop a NN training pipeline for MNIST handwritten digit classification. 
2. Implement an inference server with the trained model. The server should accept raw images as input and preduct the number with a probability. 


## Software Requirements

This application is written using Python 3.6.

Main Modules:
- Tensorflow (1.12.0)
- Keras
- Numpy (1.16.0)
- Flask (1.0.2)
- Gunicorn (19.9.0)

## Instructions

This repository contains the source code for a Python Flask Server application that has a Keras ML model deployed. REST API is used to communicate with the deployed model. The application allows one to upload an image of a handwritten digit and it returns the predicted digit. 


To start, first clone the repo, then install the dependencies, and finally download the model weights. 

```bash
git clone https://github.com/ilopezfr/digit-classifier-app.git
cd digit-classifier-app
```

Create virtual environment and activate it:
```bash
virtualenv -p /usr/local/bin/python3.6 venv
source venv/bin/activate
```
Install dependencies:
```bash
pip install -r requirements.txt
```
<!--
Download the model.h5 and save it into model folder
```bash
wget "https://drive.google.com/open?id=15ij4G9nYEb74CqhqooXRIyWjfJrqDmey" -P /model
```
-->


## Overview of the Code
This repo is structured as follows:

```bash
├── model
|    ├── predict.py
|    ├── train.py
|    ├── model-weights.h5
|    ├── model.json
|    ├── model-weights-lite.h5
|    ├── model.json
├── templates
|    ├── error.html
|    ├── index.html
|    ├── predict.html
├── tests
|    ├── conftest.py
|    ├── test_app.py
├── app.py
├── Procfile
```

## Train the model (Optional)
Two trained models on MNIST dataset are already offered and stored under the file `/model`. 
- `model.h5` was trained on 70,000 images, using the architecture described in the section below. It achieves 98% test accuracy after 20 epochs. 
- `model-lite.h5` was trained on the same data set, using the same architecture as `model.h5` except that it has an additional Max_pooling layer and a Batch Normalization layer in between the Convolution block and the Fully Connected layers. This model is lighter than `model.h5` and has only 20% of its parameters. However it's test accuracy is only 85%.  

In any case, should you want to rebuild the model using the same architecture that's described below, you can run the following script:

```bash
python model/train.py --e=20 --l=0.001
```
The `train.py` script automatically downloads MNIST dataset with 60,000 images, pre-process the images, performs data augmentation (+10000), and trains and evaluates the model on a test set. 

It saves the model files `model.h5`, `model-weights.h5`, `model.json` in the folder `\model`. 

With the model built, we are ready to serve it via Flask. 


## Start the server application and predict on a new image.
Flask allows you to serve an image of a handwritten digit with the server and get a prediction. 

In the code below, I provided a REST endpoint that supports GET and POST requests:

```bash
$ python app.py
* Serving Flask app "app" (lazy loading)
...
* Running on http://127.0.0.1:5002/
```
You can now access the REST API via http://127.0.0.1:5002.

In the browser, upload an image from your local and run the prediction. It currently accepts files with the extensions 'png', 'jpg' and 'jpeg'. 

## App Deployment options

While lightweight, Flask's built-in server is *not suitable for production* as it doesn't scale well. For this requirement, I've provided two options:

### Hosted Option:
App deployment in **Heroku**:

The app is live and can be visited throught this url:
https://quiet-journey-42975.herokuapp.com/

### Self-hosted Option:
I've also added a **self-hosted** solution using **`gunicorn`**, a WSGI HTTP Server for UNIX. To run the Flask application in this server, simply use:
```bash
gunicorn "app:create_app()" -b 0.0.0.0:5002 --workers 8 --timeout 600 --log-level critical
```

## Test coverage
To quickly run inference with one image file, once the server is running, simply open a new terminal and run the following line of code which uses a POST method to send an image to the server:
```bash
python tests/test_app.py
```

## Model Architecture

<img src="images_repo/mnist-model-architecture.png" />

The neural network contains 3 Convolutional layers, followed by 2 Fully Connected layers and 1 output layer that provides the estimations for each one of the 10 digit categories. This is a summary of `model.h5`: 

<img src="images_repo/mnist-model-summary.png"  />
