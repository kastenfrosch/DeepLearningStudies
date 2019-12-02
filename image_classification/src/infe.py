import tensorflow as tf
import csv
from src import date_model as model
import os
from src.training import input_fn

checkpoint_number = '879375'                                    # Alter the number of the checkpoint file accordingly
main_directory = r'E:\dev\j.morzeck\238\DateModel'              # Your path here
date_window_predict = main_directory + r'\100k_predict.csv'     # Path to test set file here
results_file = main_directory + r'\results_file.txt'            # Path to save the results file into


def extract_labels(path):
    labels = []

    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter=',')

        for line in reader:
            tup = list()
            for char in line[1]:
                if char == "%":
                    tup.append(10)
                else:
                    tup.append(int(char))
            labels.append(tup)

    return labels


def start_inference(load_checkpoint):

    tf.compat.v1.logging.info("Creating Estimator")

    config_proto = tf.compat.v1.ConfigProto()
    config_proto.gpu_options.visible_device_list = '0'
    config_proto.gpu_options.allow_growth = True
    config = tf.estimator.RunConfig(save_summary_steps=100, session_config=config_proto, keep_checkpoint_max=10)

    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=os.path.split(load_checkpoint)[0],
        config=config)

    tf.compat.v1.logging.info("Starting Prediction")
    predictions = estimator.predict(input_fn=lambda: input_fn(date_window_predict, tf.estimator.ModeKeys.PREDICT),
                                    checkpoint_path=load_checkpoint)

    acc = 0
    num_test_samples = 0
    test_labels = extract_labels(date_window_predict)

    with open("results_1.txt", 'w+') as f:
        f.write("GROUNDTRUTH, PREDICTED, PROBABILITY, CORRECT \n")
        pred_list = list(predictions)
        print("Number of Predictions: ", len(pred_list))

        for i in range(0, len(pred_list)):
            decoded_pred = model.decode(pred_list[i]['date'])
            print(test_labels[i])
            decoded_gt = model.decode(test_labels[i])
            print("No: " + str(i), "Groundtruth: " + str(decoded_gt), "Predicted: " + str(decoded_pred), "Accuracy: " + str(pred_list[i][model.prediction_probabilities]) + "%")
            comparison = str(decoded_gt == decoded_pred)

            f.write(decoded_gt + " " + decoded_pred + " " + str(pred_list[i][model.prediction_probabilities]) + " " +
                    comparison + "\n")
            if decoded_gt == decoded_pred:
                acc += 1
            num_test_samples = i + 1

        print("Num test Samples: ", num_test_samples)
        print("Accuracy for test set: ", acc / num_test_samples)
        f.write("Accuracy: " + str(acc/num_test_samples))


if __name__ == "__main__":
    start_inference(load_checkpoint=r'..\model_files\model.ckpt-' + checkpoint_number)
