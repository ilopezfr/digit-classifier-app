import tensorflow as tf
from keras.models import model_from_json
from PIL import Image
import numpy as np
import os
import argparse
#from test_resize import resize_image

ap = argparse.ArgumentParser()
# Basic usage: python predict.py /path/to/image
ap.add_argument(dest='image_path', action='store',
                    default = './test_images/six-1.png', 
                    help='Path to image, e.g., "./test_images/six-1.png"')

args = ap.parse_args()
image_path = args.image_path

# load model
filepath = "./model"
#model = tf.keras.models.load_model(os.path.join(filepath,'model.h5'))


# load json and create model
json_file = open(os.path.join(filepath,'model-lite.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(os.path.join(filepath,'model-weights-lite.h5'))
print("Loaded model from disk")


graph = tf.get_default_graph()
#graph = tf.compat.v1.get_default_graph()  # tf 1.14.0

def resize_image(image):
    image_size = (28, 28)
    img = Image.open(image)
    img.thumbnail(image_size, Image.ANTIALIAS) # (28, 28) or (28, 27) # same as img = img.resize()
    img = img.convert('L')

    # for the (28, 27) case, make it square
    if img.size[0] != img.size[1]:
        new_im = Image.new('L', image_size)
        new_im.paste(img, ((image_size[0]-img.size[0])//2, (image_size[1]-img.size[1])//2))
        img = new_im

    image_data = np.asarray(img, dtype=np.float32) # (784)
    # print(image_data.size)
    image_data = image_data / 255
    image_data_test = image_data.reshape((1, image_size[0], image_size[1], 1))
    return image_data_test


def predict_image(test_image):
    # load test image and preprocess it
    image_data_test = resize_image(test_image)

    global graph
    with graph.as_default():
        classes = model.predict(image_data_test)
        image_pred = str(np.argmax(classes))
        confidence = round(classes[0][np.argmax(classes)]*100, 2)
        print(image_pred, confidence)

        return image_pred, confidence


# Optional: test model with sample image
if __name__ == '__main__':
    predict_image(image_path)