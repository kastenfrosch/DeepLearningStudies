import tensorflow as tf
import csv
import random as rd
import date_model
import os
import shutil


# ALTER THESE ACCORDING TO YOUR NEEDS:
main_directory = r'E:\dev\j.morzeck\238\DateModel'
date_window_model = main_directory + r'\model_date_window'
date_window_train = main_directory + r'\100k_train.csv'
date_window_test = main_directory + r'\100k_eval.csv'

model_dir = date_window_model

batch_size = 16
epochs = 2
number_of_runs = 8


# DO NOT EDIT THIS:
num_timesteps = 12
blank_label = "%"
blank_class = 10


########################################################################################################################
def start():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    rd.seed()
    start_training(model_dir)


#######################################################################################################################
def prepare_images(path, label):

    image_as_string = tf.io.read_file(path)
    image_decoded = tf.image.decode_jpeg(image_as_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)

    label = label

    return {"x": image}, label


########################################################################################################################
def input_fn(path, mode):

    paths = []
    labels = []

    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=',')

        for line in reader:

            paths.append(line[0])

            tup = list()
            for char in line[1]:
                if char == blank_label:
                    tup.append(blank_class)
                else:
                    tup.append(int(char))

            labels.append(tup)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if mode == tf.estimator.ModeKeys.TRAIN:
        ds = ds.shuffle(batch_size * 10)

    ds = ds.map(prepare_images, num_parallel_calls=4)
    ds = ds.batch(batch_size, drop_remainder=True)

    if mode == tf.estimator.ModeKeys.TRAIN:
        ds = ds.prefetch(batch_size * 10)
        ds = ds.repeat(epochs)

    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)

    batch_images, batch_labels = iterator.get_next()

    return batch_images, batch_labels


########################################################################################################################
def get_checkpoint_files():
    checkpoint_files = []
    files = os.scandir(model_dir)
    for file in files:
        if file.is_file() and not file.name.startswith("events"):
            checkpoint_files.append(file)
    return checkpoint_files


########################################################################################################################
def copy_checkpoint(files, accuracy_value):
    percentage = int(accuracy_value * 10000)
    dest_dir = model_dir + "_" + str(percentage)
    os.makedirs(dest_dir, exist_ok=True)
    for file in files:
        shutil.copy(file.path, dest_dir)


########################################################################################################################
def train(estimator, train_hooks):
    # Start training
    tf.compat.v1.logging.info("Start Training via Estimator")
    print("Start Training via Estimator")
    eval_accuracy = 0

    for run in range(number_of_runs):
        estimator.train(
            input_fn=lambda: input_fn(date_window_train, tf.estimator.ModeKeys.TRAIN),
            steps=None,
            hooks=train_hooks)

        # Start evaluating
        tf.compat.v1.logging.info("Start Evaluation")
        print("Start Evaluation")
        eval_results = estimator.evaluate(
            input_fn=lambda: input_fn(date_window_test, tf.estimator.ModeKeys.EVAL))
        print(eval_results)

        tmp = eval_results['accuracy_eval']
        if tmp >= eval_accuracy:
            eval_accuracy = tmp
            files = get_checkpoint_files()
            copy_checkpoint(files, eval_accuracy)


########################################################################################################################
def start_training(model_dir):

    tf.compat.v1.logging.info("Create Datasets")

    tf.config.experimental.list_physical_devices('GPU')

    config_proto = tf.compat.v1.ConfigProto()
    config_proto.gpu_options.visible_device_list = '0'
    config_proto.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(save_summary_steps=100, session_config=config_proto, keep_checkpoint_max=10)

    # Create the Estimator
    date_estimator = tf.estimator.Estimator(
        model_fn=date_model.model_fn,
        model_dir=model_dir,
        config=config)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "probabilities_tensor",
                      "labels": "encoded_labels",
                      "accuracy": "accuracy",
                      "predicted": "predicted_labels"}
    logging_hook = tf.estimator.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)
    train_hooks = [logging_hook]

    # Train the model
    for x in range(number_of_runs):
        train(date_estimator, train_hooks)


########################################################################################################################
if __name__ == "__main__":
    start()
