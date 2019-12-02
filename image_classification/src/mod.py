import tensorflow as tf
import numpy as np
from resnet_model import Model as resnet
import model_architectures as ma
import cv2
from training import date_window_model
import os


# Definiere Hyperparameter für das Modell hier:
prediction_probabilities = "probabilities"
prediction_date = "date"
feature_name = "x"
image_width = 180
image_height = 60
resnet_size = 18
num_filters = 32
filter_size = 256
batch_size = 16
num_timesteps = 12
input_len_batch = batch_size * [num_timesteps]     # len(TT.MM.JJJJ)*2
num_classes = 13                                   # digits "0" to "9" + "%" + ctc_blank???
eos_token = "X"
eos_class = 11
blank_label = "%"
blank_class = 10
label_max_length = 6
num_cells = 128                                # no. of cells for rnn
model_dir = date_window_model
# keep_probability = 0.75
learning_rate = 0.00001


########################################################################################################################
def build_model(features, labels, mode):

    tf.compat.v1.logging.info("Model creation")
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    if is_training:
        keep_probability = 0.65
    else:
        keep_probability = 1.0

    features = tf.reshape(features[feature_name], [batch_size, image_height, image_width, 1])

    if mode is not tf.estimator.ModeKeys.PREDICT:
        labels = tf.reshape(labels, [batch_size, label_max_length], name='labels')

    cnn_output = build_resnet(resnet_size, features, filter_size, is_training, num_filters)
    tf.compat.v1.logging.info("cnn_output shape")
    tf.compat.v1.logging.info(cnn_output)

    rnn_output = build_rnna(cnn_output, input_len_batch, filter_size, keep_probability, "RNN", batch_size)
    tf.compat.v1.logging.info("rnn output")
    tf.compat.v1.logging.info(rnn_output)

    return rnn_output, labels


########################################################################################################################
def model_fn(features, labels, mode):

    logits_all, labels = build_model(features, labels, mode)

    # Softmax applied on the last dimenstion (i.e. num_classes)
    # of the logits tensor = tf.nn.softmax(logits_all, name="softmax_op")
    soft = tf.nn.softmax(logits_all, name="softmax_op")
    logits = tf.identity(logits_all, name="out_logits")

    # Find the largest probability for each time step, i.e. the probability with which the letter has been detected
    # Shape of probs is [num_time_steps, batch_size], corresponding to the shape of softmax output
    probabilities = tf.reduce_max(soft, axis=2, name="probabilities")
    probabilities = tf.transpose(probabilities, name="probabilities_tensor")  # TODO warum das raus?
    probabilities = tf.reduce_mean(probabilities, axis=1, name="mean_prob_tensor")

    # Decode
    # Predicted is a list with len = # top_paths. We need the top path in this case,
    # which is the first element of the list.
    predicted, log_prob = tf.nn.ctc_greedy_decoder(logits, input_len_batch, merge_repeated=False)
    predicted = tf.cast(predicted[0], tf.int32)  # sparse
    predicted_dense = tf.sparse.to_dense(predicted, default_value=eos_class, name='pred_dense')

    predicted_dense = tf.identity(predicted_dense, name="predicted_labels")  # TODO warum das raus?

    if mode is tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={prediction_probabilities: probabilities,
                                                                  prediction_date: predicted_dense})

    labels = tf.identity(labels, name='encoded_labels')

    # Cast labels of shape (batch_size, 10) to int32 and a sparse tensor
    labels_sparse = tf.contrib.layers.dense_to_sparse(labels, eos_token=eos_class)

    predicted_dense_padded = pad_and_slice(predicted_dense)
    labels_padded = pad_and_slice(labels)
    correct_prediction = tf.equal(predicted_dense_padded, labels_padded, name='correct_pred')
    accuracy_boolean = tf.reduce_all(correct_prediction, 1, name='accuracy')
    accuracy_as_num = tf.cast(accuracy_boolean, tf.float32)
    accuracy = tf.reduce_mean(accuracy_as_num)

    # Calculate the normalized CER metric
    edit_dist = tf.edit_distance(hypothesis=predicted, truth=labels_sparse)
    cer = tf.reduce_mean(edit_dist, name="cer_tensor_per_field")

    loss_per_image = tf.compat.v1.nn.ctc_loss(labels_sparse, logits_all, input_len_batch, ctc_merge_repeated=False)
    loss_per_batch = tf.reduce_mean(loss_per_image)

    # Summarize CTC Loss (for both TRAIN and EVAL modes)
    tf.compat.v1.summary.scalar('CER', cer)
    tf.compat.v1.summary.scalar('accuracy', accuracy)

    # Configure the Training Op (for TRAIN mode)
    images = tf_put_text(features["x"], predicted_dense_padded, 0, 12)
    images = tf_put_text(images, labels_padded, 0, 58)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.compat.v1.summary.image('train', images)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss_per_batch,
                                      global_step=tf.compat.v1.train.get_global_step())
        train_op = tf.group([train_op, update_ops])

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss_per_batch,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    accuracy_eval = tf.compat.v1.metrics.mean(accuracy)
    tf.compat.v1.summary.image('val', images)

    eval_summary_hook = tf.estimator.SummarySaverHook(
        save_steps=100,
        output_dir=os.path.join(model_dir, "eval_saved_hook"),
        summary_op=tf.compat.v1.summary.merge_all())

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss_per_batch,
                                      eval_metric_ops={'accuracy_eval': accuracy_eval}) #,
                                      #evaluation_hooks=[eval_summary_hook])


########################################################################################################################
def build_resnet(resnet_size, features, final_filter_size, is_training, num_filters):
    # Die Filterzahl mal die Anzahl der Gruppen muss der final_filter_size entsprechen
    resnet_model = resnet(resnet_size, False, num_filters, 7, 2, 2, 2,
                          [3, 4, 6, 3], [1, 2, 2, 2], final_filter_size,
                          data_format='channels_last')
    resnet_features = resnet_model(features, training=is_training)

    return resnet_features


########################################################################################################################
def build_rnna(cnn_features, sequence, depth, keep_prob, rnn_type, batch_size):
    size_height, size_width = cnn_features.get_shape()[1:3]
    cnn_features = tf.reshape(cnn_features, [-1, size_width * size_height * depth])
    with tf.compat.v1.variable_scope('attention'):
        if rnn_type == "RNN":
            rnn_outputs = ma.RNN_Attention(cnn_features, size_height, size_width, sequence, depth,
                                           num_cells, keep_prob)
        elif rnn_type == "BLSTM":
            rnn_outputs = ma.BLSTM_Attention(cnn_features, size_height, size_width, sequence, depth,
                                             num_cells, keep_prob)
    with tf.compat.v1.variable_scope('logits'):
        stacked_rnn_output = tf.reshape(rnn_outputs, [-1, num_cells])
        w_fc1 = ma.weight_variable([num_cells, num_classes], 'wf1')
        b_fc1 = ma.bias_variable([num_classes], 'bf1')

        logits = tf.matmul(stacked_rnn_output, w_fc1) + b_fc1
        logits = tf.reshape(logits, [-1, sequence[0], num_classes])
        logits = tf.transpose(logits, perm=[1, 0, 2])
    return logits


########################################################################################################################
def pad_and_slice(tensor):
    padding = tf.constant([[0, 0], [0, num_timesteps]])
    padded_tensor = tf.pad(tensor, padding, "CONSTANT", constant_values=eos_class)
    sliced_tensor = tf.slice(padded_tensor, [0, 0], [-1, num_timesteps])
    return sliced_tensor


########################################################################################################################
def put_text(imgs, text, x, y):
    result = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        encoded_text = text[i]
        decoded_text = decode(encoded_text)
        # You may need to adjust text size and position and size.
        # If your images are in [0, 255] range replace (0, 0, 1) with (0, 0, 255)
        result[i, :, :, :] = cv2.putText(imgs[i, :, :, :], str(decoded_text), (x, y),
                                         cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 1), 2)
    return result


########################################################################################################################
def decode(array):
    encoded_text = ""
    for number in array:
        if number == blank_class:
            encoded_text += blank_label
        else:
            encoded_text += str(number)
    return encoded_text


########################################################################################################################
def tf_put_text(imgs, text, x, y):
    return tf.numpy_function(put_text, [imgs, text, x, y], Tout=imgs.dtype)

