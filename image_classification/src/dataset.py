import tensorflow as tf
import matplotlib.pyplot as plt


def create_dataset(verify):

    # Download cifar10 dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Verify and show the imported data if needed
    if verify:
        verify_data(train_images, train_labels)

    # save dataset into model_files?

    # augmentation?

    return (train_images, train_labels), (test_images, test_labels)


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


