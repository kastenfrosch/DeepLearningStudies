import tensorflow as tf

import logging
import random as rd
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import time

from IPython import display

import gan_config as config
from gan_dataset import create_dataset
from gan_model import make_generator_model, make_discriminator_model


def start():
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    rd.seed()
    start_training()


def start_training():
    # Set memory growth for gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Create config:
    cfg = config.Config()

    # Load and prepare the dataset
    train_dataset = create_dataset(train=True)

    # Creating the models:
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Loading checkpoints:
    # latest = tf.train.latest_checkpoint(os.path.join(cfg.model_dir, 'checkpoints'))
    # if latest:
    #     generator.load_weights(latest)
    #     discriminator.load_weights(latest)
    #     print("Loaded weights from {}".format(latest))

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_prefix = os.path.join(cfg.model_dir, 'checkpoints', "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([cfg.num_examples_to_generate, cfg.noise_dim])

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        noise = tf.random.normal([cfg.batch_size, cfg.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    def train(dataset, epochs):
        print("Starting training...")
        for epoch in range(epochs):
            start = time.time()
            for image_batch in dataset:
                train_step(image_batch)

            # Produce images for the GIF as we go
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)

            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epochs,
                                 seed)

    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        print("Generating images...")
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(os.path.join(cfg.model_dir, 'images', 'image_at_epoch_{:04d}.png'.format(epoch)))
        plt.show()

    train(train_dataset, cfg.epochs)

    checkpoint.restore(tf.train.latest_checkpoint(os.path.join(cfg.model_dir, 'checkpoints')))

    # Display a single image using the epoch number
    def display_image(epoch_no):
        return Image.open(os.path.join(cfg.model_dir, 'images', 'image_at_epoch_{:04d}.png'.format(epoch_no)))

    display_image(cfg.epochs)

    anim_file = os.path.join(cfg.model_dir, 'images', 'dcgan.gif')

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(cfg.model_dir, 'images', 'image*.png'))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    import IPython
    if IPython.version_info > (6, 2, 0, ''):
        display.Image(filename=anim_file)


if __name__ == '__main__':
    start()
