
"""Code for turning the tfrecord file into other format readable for pytorch."""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


# tf record data location:
DATA_DIR = 'data/raw/push/push_testseen'

OUT_DIR = 'data/processed/push/push_testseen'

SEQUENCE_LENGTH = 10

ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
COLOR_CHAN = 3

# Dimension of the state and action.
STATE_DIM = 5
ACTION_DIM = 5


def convert():
    with tf.Session() as sess:
        files = gfile.Glob(os.path.join(DATA_DIR, '*'))
        queue = tf.train.string_input_producer(files, shuffle=False)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)
        image_seq, state_seq, action_seq = [], [], []

        for i in range(SEQUENCE_LENGTH):
            image_name = 'move/' + str(i) + '/image/encoded'
            action_name = 'move/' + str(i) + '/commanded_pose/vec_pitch_yaw'
            state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'

            features = {
                image_name: tf.FixedLenFeature([1], tf.string),
                action_name: tf.FixedLenFeature([STATE_DIM], tf.float32),
                state_name: tf.FixedLenFeature([ACTION_DIM], tf.float32)
            }

            features = tf.parse_single_example(serialized_example, features=features)
            image_buffer = tf.reshape(features[image_name], shape=[])
            image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
            image.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])

            crop_size = min(ORIGINAL_WIDTH, ORIGINAL_HEIGHT)
            image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
            image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
            image_seq.append(image)

            state = tf.reshape(features[state_name], shape=[1, STATE_DIM])
            state_seq.append(state)
            action = tf.reshape(features[action_name], shape=[1, ACTION_DIM])
            action_seq.append(action)

        image_seq = tf.concat(axis=0, values=image_seq)
        state_seq = tf.concat(axis=0, values=state_seq)
        action_seq = tf.concat(axis=0, values=action_seq)

        [image_batch, action_batch, state_batch] = tf.train.batch(
            [image_seq, action_seq, state_seq],
            64,
            num_threads=8,
            capacity=100 * 64,
            allow_smaller_final_batch=True)

        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)

        for j in range(len(files)):
            data_length = sum(1 for _ in tf.python_io.tf_record_iterator(files[j])) // 64 + 1
            image_bolb, action_bolb, state_bolb = [], [], []
            for _ in range(data_length):
                imgs, acts, stas = sess.run([image_batch, action_batch, state_batch])
                image_bolb.append(imgs)
                action_bolb.append(acts)
                state_bolb.append(stas)
            image_bolb = np.concatenate(image_bolb, axis=0)
            action_bolb = np.concatenate(action_bolb, axis=0)
            state_bolb = np.concatenate(state_bolb, axis=0)

            np.save(OUT_DIR + '/image_batch_' + str(j), image_bolb)
            np.save(OUT_DIR+ '/action_batch_' + str(j), action_bolb)
            np.save(OUT_DIR + '/state_batch_' + str(j), state_bolb)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    convert()
