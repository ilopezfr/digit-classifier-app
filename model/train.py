import time
import os
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten
from keras.layers import Concatenate, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard

batch_size = 128
num_classes = 10
num_epochs = 2
input_shape = (28, 28, 1)


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255


    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def init_model():
    start_time = time.time()
    print('Compiling Model...')

    input_layer = Input(shape=input_shape)
    conv1 = Conv2D(32, (1, 1), kernel_initializer=keras.initializers.glorot_normal(), activation='relu')(input_layer)
    pool1 = MaxPooling2D(2, 2)(conv1)
    conv2_1 = Conv2D(64, (1, 1), activation='relu', padding='same')(pool1)
    pool2_1 = MaxPooling2D(2, 2)(conv2_1)
    drop2_1 = Dropout(0.5)(pool2_1)
    conv2_2 = Conv2D(64, (1, 1), activation='relu', padding='same')(pool1)
    pool2_2 = MaxPooling2D(2, 2)(conv2_2)
    drop2_2 = Dropout(0.5)(pool2_2)
    conv3_1 = Conv2D(256, (1, 1), activation='relu', padding='same')(drop2_1)
    conv3_2 = Conv2D(256, (1, 1), activation='relu', padding='same')(drop2_2)
    merged = Concatenate(axis=-1)([conv3_1, conv3_2])
    merged = Dropout(0.5)(merged)
    merged = Flatten()(merged)
    fc1 = Dense(1000, activation='relu')(merged)
    fc2 = Dense(500, activation='relu')(fc1)
    out = Dense(10, activation="softmax")(fc2)

    model = Model(input_layer, out)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,
                loss=categorical_crossentropy,
                metrics=['accuracy'])


    print('Model compiled in {0} seconds'.format(time.time() - start_time))

    return model  
  
  
def run_network(data=None, model=None, epochs=20, batch=128):
    try:
        start_time = time.time()
        if data is None: 
            (x_train, y_train), (x_test, y_test) = get_data()
        else:
            (x_train, y_train), (x_test, y_test) = data


        if model is None:
            model = init_model()
            
        print('Training model...')
        
        # define the checkpoint

        filepath = "./model"  #mnist_model.h5"
        # checkpoints = []
        if not os.path.exists('./model/'):
             os.makedirs('./model/')

        model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1, 
                    validation_data=(x_test, y_test), shuffle=True #,
                    #callbacks=checkpoints
                    )

        print("Training duration : {0}".format(time.time() - start_time))

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

        # save the model
        model.save(os.path.join(filepath,'model.h5'))
        # serialize model to JSON
        # model_json = model.to_json()
        # with open(os.path.join(filepath,'model.json'), 'w') as json_file:
        #     json_file.write(model_json)
        # serialize weights to HDF5
        # model.save_weights(os.path.join(filepath,'model-weights.h5'))
        print('Saved model to disk')


        return model

    except KeyboardInterrupt:
        print(' KeyboardInterrupt')
        return model
      
      
if __name__ == '__main__':
    data = get_data()   # this way we can explicitly use other data
    model = init_model()
    run_network(data, model)