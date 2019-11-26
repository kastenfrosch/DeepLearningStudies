from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt
from model import model_fn


def create_dataset(verify):

    # Download cifar10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Verify and show the imported data if needed
    if verify:
        verify_data(train_images, train_labels)

    return train_images, train_labels, test_images, test_labels


def verify_data(train_images, train_labels):

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


def start(epochs):

    # Importing dataset
    train_images, train_labels, test_images, test_labels = create_dataset(verify=False)

    # Create model
    model = model_fn()

    # Training the model
    history = model.fit(x=train_images, y=train_labels,
                        batch_size=16,
                        epochs=epochs,
                        validation_data=(test_images, test_labels))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print(test_acc)

    # TODO: train with estimator um fortschritt zum speichern, checkpoints etc
    # https://www.tensorflow.org/tutorials/estimator/premade?hl=de
    # https://developers.googleblog.com/2017/12/creating-custom-estimators-in-tensorflow.html
    # https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428

    # cifar >90%!
    # learning rate decay!
    # data augmentation: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/

    # nlp, ocr oder sowas

    # TODO: MAZE SOLVER?
    # https://www.youtube.com/watch?v=rop0W4QDOUI

    # DRL?


if __name__ == '__main__':
    start(epochs=25)


