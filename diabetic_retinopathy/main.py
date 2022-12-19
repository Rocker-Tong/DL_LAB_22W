import sys

import gin
import logging

import keras.callbacks
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.Resnet import ResNet101
from models.VGG import VGG16
from models.CNN import CNN
import tensorflow as tf
from models.transfer_learning import inception_resnet_v2, mobilenet


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')


@gin.configurable
def main(argv, classification='regression'):
# choose classification, binary or multiple or regression

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    if FLAGS.train:
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)
    else:
        utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)


    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(classification=classification)

    # Choose model
    # model = ResNet101(bottleneck_list=[3, 4, 23, 3], neurons=64, classification)
    # model = VGG16(classification)
    model = CNN(classification)
    # model = inception_resnet_v2(classification)
    # model = mobilenet(classification)

    model.build((16, 256, 256, 3))

    # Show the model structure
    model.get_layer('sequential').summary()

    # Show the model structure while using transfer learning
    # model.summary()
    logging.info('The model is loaded successfully.')

    if FLAGS.train:
        logging.info('Start the training process...')
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths=run_paths, classification=classification)

        '''Method of compiled model.fit()'''
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=run_paths['path_model_Tensorboard'],
        #                                                    profile_batch='500, 520')
        # opt = tf.keras.optimizers.Adam(lr=5e-5)
        # model.compile(loss="BinaryCrossentropy", optimizer=opt, metrics=["accuracy"])
        # model.fit(ds_train,
        #           epochs=5,
        #           batch_size=16,
        #           validation_data=ds_val,
        #           callbacks=[tensorboard_callback])

        '''Method of low-level control of training'''
        trainer.train(10)

    else:
        evaluate(model, ds_test, classification=classification)

if __name__ == "__main__":
    app.run(main)