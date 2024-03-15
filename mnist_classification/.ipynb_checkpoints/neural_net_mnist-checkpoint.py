import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


"""
2. Construct simple neural network for MNIST dataset with three
hidden layers. Derive the accuracy and loss curves.
"""


def plot_curves(history):
    """
    Plots the accuracy and loss curves
    after training
    Arguments:
        history    -- training history
    Result: Accuracy and loss curves
    """

    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    plt.figure(figsize=[8,6])
    plt.plot(history.history['accuracy'], 'r',linewidth=3.0)
    plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()


def create_model():
    """
    Creates a simple neural network with
    for MNIST classification
    Arguments: None
    Result: Trained model
    """
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print('Training data shape : ', train_images.shape, train_labels.shape)
    print('Testing data shape : ', test_images.shape, test_labels.shape)
    classes = np.unique(train_labels)
    classes_num = len(classes)
    print('Total number of outputs : ', classes_num)
    print('Output classes : ', classes)

    dim_data = np.prod(train_images.shape[1:])
    train_data = train_images.reshape(train_images.shape[0], dim_data)
    test_data = test_images.reshape(test_images.shape[0], dim_data)

    # Convert to [0 -1] range
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255

    train_labels_one_hot = to_categorical(train_labels)
    test_labels_one_hot = to_categorical(test_labels)

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(dim_data,)))   # hidden layer 1
    model.add(Dense(512, activation='relu'))                            # hidden layer 2
    model.add(Dense(512, activation='relu'))                            # hidden layer 3
    model.add(Dense(classes_num, activation='softmax'))                 # output layer
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(train_data, train_labels_one_hot,
                        batch_size=64,epochs=20,
                        verbose=1,validation_data=(test_data,test_labels_one_hot))
    [test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
    print("Evaluation result on Test Data : Loss = {}, accuracy ={}".format(test_loss, test_acc))
    plot_curves(history)


if __name__ == "__main__":
    create_model()
