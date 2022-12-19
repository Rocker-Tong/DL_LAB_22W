import logging
import gin
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
from sklearn import model_selection
from input_pipeline.preprocessing import preprocess, augment, to_image, to_csv
import cv2


# Read labels
@gin.configurable
def read_labels(data_dir, dataset, classification):
    if dataset == "train":
        files = pd.read_csv(data_dir + "/labels/train.csv")
        files.dropna(inplace=True, axis='columns')
        train_files_with_labels = files['Retinopathy grade'].values.tolist()  # .drop_duplicates()

        """Following is just for the test of VGG"""
        '''for i in range(len(train_files_with_labels)):
            if train_files_with_labels[i] <= 1:
                train_files_with_labels[i] = 0
            else:
                train_files_with_labels[i] = 1'''
        """The VGG test stops here"""

        return train_files_with_labels

    elif dataset == "test":
        files = pd.read_csv(data_dir + "/labels/test.csv")
        files.dropna(inplace=True, axis='columns')
        test_files_with_labels = files['Retinopathy grade'].values.tolist()
        if classification == 'binary':
            for i in range(len(test_files_with_labels)):
                if test_files_with_labels[i] <= 1:
                    test_files_with_labels[i] = 0
                else:
                    test_files_with_labels[i] = 1
        return test_files_with_labels

    else:
        print("Please choose a train or test set.")

@gin.configurable
def resampling(data_dir, dataset, classification):
    train_dir = data_dir + "/images/train/"
    filenames = [train_dir + filename for filename in os.listdir(train_dir)]
    filenames.sort(key=lambda x: x[-7:-4])
    if len(filenames) == 684:
        filenames.pop()
    train_files_with_labels = read_labels(data_dir, dataset, classification)
    label_3_num = 1
    label_4_num = 1
    i = 0
    fold_new = "tfrecord/train_resampling/"
    # fold_new = "~/train_resampling/"
    if os.path.exists(fold_new):
        shutil.rmtree(fold_new)
    os.makedirs(fold_new)
    if classification == 'multiple' or classification == 'regression':
        for filename in filenames:
            if train_files_with_labels[i] == 3 and label_3_num <= 10:
                to_image(filename=filename, path=fold_new)
                to_csv(list=train_files_with_labels, label=3, i=i)
                i += 7
                label_3_num += 1

            elif train_files_with_labels[i] == 4 and label_4_num <= 15:
                to_image(filename=filename, path=fold_new)
                to_csv(list=train_files_with_labels, label=4, i=i)
                i += 7
                label_4_num += 1

            elif train_files_with_labels[i] == 1:
                to_image(filename=filename, path=fold_new)
                to_csv(list=train_files_with_labels, label=1, i=i)
                i += 7

            elif train_files_with_labels[i] == 0 or train_files_with_labels[i] == 2:
                file_name_with_jpg = filename.split("train/", 1)[1]
                file_name_without_jpg = file_name_with_jpg.split(".jpg", 1)[0]
                filename_new = fold_new + file_name_without_jpg + "0.jpg"
                shutil.copy(filename, filename_new)
                i += 1

            else:
                file_name_with_jpg = filename.split("train/", 1)[1]
                file_name_without_jpg = file_name_with_jpg.split(".jpg", 1)[0]
                filename_new = fold_new + file_name_without_jpg + "0.jpg"
                shutil.copy(filename, filename_new)
                i += 1



    elif classification == 'binary':
        for filename in filenames:
            if train_files_with_labels[i] == 1:
                to_image(filename=filename, path=fold_new)
                to_csv(list=train_files_with_labels, label=1, i=i)
                i += 7
            else:
                file_name_with_jpg = filename.split("train/", 1)[1]
                file_name_without_jpg = file_name_with_jpg.split(".jpg", 1)[0]
                filename_new = fold_new + file_name_without_jpg + "0.jpg"
                shutil.copy(filename, filename_new)
                i += 1
        print(train_files_with_labels)
        for i in range(len(train_files_with_labels)):
            if train_files_with_labels[i] <= 1:
                train_files_with_labels[i] = 0
            else:
                train_files_with_labels[i] = 1

    logging.info('The training dataset is resampled.')

    return train_files_with_labels


@gin.configurable
# Read images and define a path for tfrecord file
def prepare_images(data_dir, dataset):
    if dataset == "train":
        train_dir = "tfrecord/train_resampling/"

        # train_dir = "~/train_resampling/"

        # train_dir = data_dir + "/images/train/" # without resampling
        path = "tfrecord/train.tfrecords"

        # path = "~/train.tfrecords"

        filenames = [train_dir + filename for filename in os.listdir(train_dir)]
        # filenames.sort(key=lambda x: int(x.split('.')[0][-3:])) # without resampling
        filenames.sort(key=lambda x: x[-8:-4])
        return path, filenames

    elif dataset == "test":
        test_dir = data_dir + "/images/test/"
        path = "tfrecord/test.tfrecords"

        # path = "~/test.tfrecords"

        filenames = [test_dir + filename for filename in os.listdir(test_dir)]
        filenames.sort(key=lambda x: int(x[-7:-4]))
        return path, filenames

    else:
        return ValueError


def histogram(data, data_dir):
    # data.head(413)
    x = data.loc[:, ['Retinopathy grade']]
    n, bins, patches = plt.hist(x=x, bins=30, range=(0, 4), )
    for i in range(len(n)):
        plt.text(bins[i], n[i] * 1.02, int(n[i]), fontsize=12, horizontalalignment="center")
    plt.ylim(0, 150)
    plt.xlabel('Retinopathy grade')
    plt.ylabel('Number')
    # fig = plt.gcf()
    # fig.set_size_inches(5, 10)
    plt.savefig(data_dir + "/labels/histogram.png")
    plt.close()
    return


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

@gin.configurable
# Create TFRecord file from training or testing set
def write_tfrecord(filenames, labels, path):
    logging.info(f"Creating TFRecord file to {path} now...")
    with tf.io.TFRecordWriter(path) as writer:
        for filename, label in zip(filenames, labels):
            image_string = open(filename, 'rb').read()  # read image to RAM in binary mode
            image_shape = tf.io.decode_jpeg(image_string).shape
            feature = {  # build Feature dictionary
                'image': _bytes_feature(image_string),
                'label': _int64_feature(label),
                'image_height': _int64_feature(image_shape[0]),
                'image_width': _int64_feature(image_shape[1]),
                'image_depth': _int64_feature(image_shape[2]),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))  # build Example
            writer.write(example.SerializeToString())
    logging.info("A new TFRecord file is created.")
    return


@gin.configurable
def creating_action(data_dir, classification):
    # Read labels
    train_labels = resampling(data_dir, "train", classification)
    # train_labels = read_labels(data_dir, "train") # without resampling
    print(len(train_labels))
    test_labels = read_labels(data_dir, "test", classification)

    # Read images
    train_tfrecord_file, train_filenames = prepare_images(data_dir, "train")
    print(len(train_filenames))
    test_tfrecord_file, test_filenames = prepare_images(data_dir, "test")
    val_tfrecord_file = "tfrecord/val.tfrecords"

    # val_tfrecord_file = "~/val.tfrecords"

    # Split validation set
    train_filenames, val_filenames, train_labels, val_labels \
        = model_selection.train_test_split(train_filenames, train_labels, test_size=0.2, shuffle=False)

    # Create TFRecord files
    write_tfrecord(train_filenames, train_labels, train_tfrecord_file)
    write_tfrecord(test_filenames, test_labels, test_tfrecord_file)
    write_tfrecord(val_filenames, val_labels, val_tfrecord_file)
    return train_tfrecord_file, test_tfrecord_file, val_tfrecord_file


"""batch_size = 1

# read
train_dataset = tf.data.TFRecordDataset(train_tfrecord_file)
# decode
train_dataset = train_dataset.map(parse_example)
train_dataset = train_dataset.shuffle(buffer_size=256)
train_dataset = train_dataset.batch(batch_size=32)
train_dataset = train_dataset.prefetch(buffer_size=256)

# read
test_dataset = tf.data.TFRecordDataset(test_tfrecord_file)
# decode
test_dataset = test_dataset.map(parse_example)
test_dataset = test_dataset.shuffle(buffer_size=256)
test_dataset = test_dataset.batch(batch_size=32)
test_dataset = test_dataset.prefetch(buffer_size=256)
# for image,label in test_dataset: # .take(1):
#     print(label)
#     plt.imshow(image)
#     plt.show()
# (256, 256, 3) tf.Tensor(0, shape=(), dtype=int64)



epochs = 10

model = models.Sequential()
model.add(layers.Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
opt = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()
history = model.fit(train_dataset, epochs=epochs, batch_size=8)

plt.plot(history.history["loss"])
plt.legend(["loss"])
plt.xticks(range(epochs))
plt.xlabel("epochs")
plt.title("Training process")
plt.show()

print(model.predict(test_dataset))
"""
