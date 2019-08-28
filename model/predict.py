import tensorflow as tf
from PIL import Image
import numpy as np
#from test_resize import resize_image

# load model
model = tf.keras.models.load_model('./model/model.h5')

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
    predict_image('./test_images/six-1.png')