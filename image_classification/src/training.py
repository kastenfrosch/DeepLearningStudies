from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, regularizers

import matplotlib.pyplot as plt
from model import model_fn
from config import config
import dataset


def plot_history(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(config.epochs)

    plt.figure(figsize=(16, 8))
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


def input_fn(inputs):

    images, labels = inputs
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.batch(config.batch_size).repeat()

    return ds


def input_fn_2(train: bool):

    if train:
        split = tfds.Split.TRAIN
    else:
        split = tfds.Split.TEST

    ds = tfds.load('cifar10', split=split, as_supervised=True)

    ds = ds.map(lambda features, labels: ({'conv2d_input': features/255}, labels))
    ds = ds.batch(config.batch_size).repeat()

    return ds


def start():

    # Importing dataset
    train, test = dataset.create_dataset(verify=False)

    # Creating the model
    model = model_fn()

    # Create estimator from model
    cifar_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model,
        model_dir=config.model_dir
    )

    for x in range(config.no_runs):
        for y in range(config.epochs):
            # Train and evaluate the estimator
            print("\nStarting training run ({} of {}).".format(y+1, config.epochs))
            cifar_estimator.train(input_fn=lambda: input_fn_2(True), steps=int(50000/config.batch_size))

        print("\nStarting evaluation ({}/{})".format(x+1, config.no_runs))
        eval_result = cifar_estimator.evaluate(input_fn=lambda: input_fn_2(False), steps=int(10000/config.batch_size))
        print("\nEval result after {}: {}".format(x+1, eval_result))

    # Training the model
    # history = model.fit(x=train_images, y=train_labels,
    #                     batch_size=config.batch_size,
    #                     epochs=config.epochs,
    #                     validation_data=(test_images, test_labels))
    #
    # plot_history(history=history)
    #
    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    # print("\nTest accuracy after {} epochs: {:.2%}".format(config.epochs, test_acc))

    # TODO: train with estimator um fortschritt zum speichern, checkpoints etc

    # cifar >90%!
    # learning rate decay!
    # data augmentation: https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/

    # nlp, ocr oder sowas

    # TODO: MAZE SOLVER?
    # https://www.youtube.com/watch?v=rop0W4QDOUI

    # DRL?


if __name__ == '__main__':
    start()


