from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.keras import datasets, layers, models, regularizers


def model_fn():

    # Create CNN model
    model = models.Sequential()

    weight_decay = 1e-4

    # Convolutional Layers
    # First conv layer requires an input shape
    model.add(layers.Conv2D(filters=32,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay),
                            input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=32,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=64,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=128,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))

    # Adding dense layers with softmax activation
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Output model summary
    model.summary()

    return model

