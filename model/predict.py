import tensorflow as tf
from PIL import Image
import numpy as np
#from test_resize import resize_image

# load model
model = tf.keras.models.load_model('./model/model.h5')

graph = tf.get_default_graph()

def resize_image(image):
    # desired_size = 28

    # im = Image.open(image)
    # im.show()
    # print(im.size)
    # old_size = im.size  # old_size[0] is in (width, height) format

    # ratio = float(desired_size)/max(old_size)
    # new_size = tuple([int(x*ratio) for x in old_size])
    # print(new_size)
    # # use thumbnail() or resize() method to resize the input image

    # # thumbnail is a in-place operation

    # # im.thumbnail(new_size, Image.ANTIALIAS)

    # im = im.resize(new_size, Image.ANTIALIAS)
    # print(im.size)
    # im = im.convert('L')
    # print(im.size)

    # # create a new image and paste the resized on it

    # new_im = Image.new("L", (desired_size, desired_size))
    # print(new_im.size)

    # new_im.paste(im, ((desired_size-new_size[0])//2,
    #                     (desired_size-new_size[1])//2))

    # new_im.show()
    # print(new_im.size)
    ####
    image_size = (28, 28)
    img = Image.open(image)
    #old_size = img.size  # old_size[0] is in (width, height) format
    print(img.size)
    img.thumbnail(image_size, Image.ANTIALIAS) # (28, 28) or (28, 27) # same as img = img.resize()
    print(img.size)
    img = img.convert('L')

    # for the (28, 27) case, make it square

    if img.size[0] != img.size[1]:
        # ratio = float(image_size[0])/max(old_size)
        # new_size = tuple([int(x*ratio) for x in old_size])
        # print('new_size: ', new_size)

        new_im = Image.new('L', image_size)
        print('new_img.size: ', new_im.size)
        new_im.paste(img, ((image_size[0]-img.size[0])//2, (image_size[1]-img.size[1])//2))
        print(new_im.size)
        img = new_im
    img.show()

    image_data = np.asarray(img, dtype=np.float32) # (784)
    # print(image_data.size)
    image_data = image_data / 255
    image_data_test = image_data.reshape((1, image_size[0], image_size[1], 1))
    return image_data_test


def predict_image(test_image):
    # load test image and preprocess it
    image_data_test = resize_image(test_image)

    # image_size = (28, 28)
    # img = Image.open(test_image)
    # #old_size = img.size  # old_size[0] is in (width, height) format
    # #print(img.size)
    # img.thumbnail(image_size, Image.ANTIALIAS) # (28, 28) or (28, 27) # same as img = img.resize()
    # #print(img.size)
    # img = img.convert('L')

    # # if img not squared, make it square
    # if img.size[0] != img.size[1]:
    #     # ratio = float(image_size[0])/max(old_size)
    #     # new_size = tuple([int(x*ratio) for x in old_size])
    #     # print('new_size: ', new_size)

    #     new_im = Image.new('L', image_size)
    #     #print('new_img.size: ', new_im.size)
    #     new_im.paste(img, ((image_size[0]-img.size[0])//2, (image_size[1]-img.size[1])//2))
    #     #print(new_im.size)
    #     img = new_im
    # #img.show()

    # image_data = np.asarray(img, dtype=np.float32) # (784)
    # #print(image_data.size)
    # image_data = image_data / 255
    # image_data_test = image_data.reshape((1, image_size[0], image_size[1], 1))

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
