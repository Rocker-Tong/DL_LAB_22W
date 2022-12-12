import os
import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

import input_pipeline.TFRecord as tfr
# import input—_pipeline.preprocessing
from input_pipeline.preprocessing import preprocess, augment
from input_pipeline.TFRecord import histogram


@gin.configurable
def load(name, data_dir):
    if name == "IDRID":
        logging.info(f"Preparing dataset {name}...")
        if os.path.exists(data_dir + '/train.tfrecords')\
            and os.path.exists(data_dir + '/test.tfrecords')\
                and os.path.exists(data_dir + '/val.tfrecords'):
            train_file = data_dir + '/train.tfrecords'
            test_file = data_dir + '/test.tfrecords'
            val_file = data_dir + '/val.tfrecords'
            logging.info('The TFRecord files are existed')
        else:
            logging.info('The TFRecord files are not existed, creating new files...')
            train_file, test_file, val_file = tfr.creating_action(data_dir)

        ds_info = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_height': tf.io.FixedLenFeature([], tf.int64),
            'image_width': tf.io.FixedLenFeature([], tf.int64),
            'image_depth': tf.io.FixedLenFeature([], tf.int64),
        }

        # Parse the dataset
        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, ds_info)
            feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=3)
            feature_dict['image'] = preprocess(feature_dict['image']) / 255.0
            return feature_dict['image'], feature_dict['label']

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # Import the train set from TFRecord file
        ds_train = tf.data.TFRecordDataset(train_file)
        ds_train = ds_train.map(_parse_example, num_parallel_calls=AUTOTUNE)

        # Import the train set from TFRecord file
        ds_val = tf.data.TFRecordDataset(val_file)
        ds_val = ds_val.map(_parse_example, num_parallel_calls=AUTOTUNE)

        # Import the train set from TFRecord file
        ds_test = tf.data.TFRecordDataset(test_file)
        ds_test = ds_test.map(_parse_example, num_parallel_calls=AUTOTUNE)

        # Check the class distribution of the labels
        # histogram(labels_num, data_dir)

        # Preprocessing and augmentation
        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            images = []
            labels = []
            csv = pd.read_csv(r"/Users/yinzheming/Desktop/Deep_Learning/Lab/IDRID_dataset/labels/train.csv", header=0)
            for index in csv.index:
                image = str(csv.loc[index].values[0])
                label = int(csv.loc[index].value[1])
                images.append(image)
                labels.append(label)
            return img_label_dict[images], img_label_dict[labels]

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, buffer_size=256, batch_size=16, caching=False):
    logging.info('Preparing the datasets...')
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Prepare training dataset
    # ds_train = preprocessing.augment(ds_train)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size)
    ds_train = ds_train.batch(batch_size)
    # ds_train = ds_train.repeat(10)
    ds_train = ds_train.prefetch(AUTOTUNE)
    logging.info('Train set is prepared')

    # Prepare validation dataset
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(AUTOTUNE)
    logging.info('Validation set is prepared')

    # Prepare test dataset
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(AUTOTUNE)
    logging.info('Test set is prepared')

    return ds_train, ds_val, ds_test, ds_info


