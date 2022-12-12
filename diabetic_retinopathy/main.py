import sys

import gin
import logging
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


FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')



def main(argv):

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
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # model
    # model = ResNet101(bottleneck_list=[3, 4, 23, 3], neurons=64)
    # model = VGG16()
    model = CNN()
    model.build((16, 256, 256, 3))
    model.get_layer('sequential').summary()
    logging.info('The model is loaded successfully.')

    if FLAGS.train:
        logging.info('Start the training process...')
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths=run_paths)
        # opt = tf.keras.optimizers.Adam(lr=0.0005 / 10)
        # model.compile(loss="BinaryCrossentropy", optimizer=opt, metrics=["accuracy"])
        # model.fit(ds_train, epochs=10, batch_size=8, validation_data=ds_val)
        # trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue

    else:
        evaluate(model,
                 ds_test,
                 ds_info,
                 run_paths)

if __name__ == "__main__":
    app.run(main)