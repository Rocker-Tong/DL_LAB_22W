import gin
import tensorflow as tf
import logging
import wandb
import time


@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval, classification):
    # def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval):

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.classification = classification

        # Summary Writer
        self.summary_writer = tf.summary.create_file_writer(self.run_paths['path_model_Tensorboard'])

        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=model, optimizer=tf.keras.optimizers.Adam())
        self.manager = tf.train.CheckpointManager(self.ckpt,
                                                  directory=self.run_paths["path_ckpts_train"],
                                                  max_to_keep=10)

        # Loss objective
        if self.classification == 'binary':
            self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        elif self.classification == 'multiple':
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

        # accuracy objective
        if self.classification == 'binary':
            self.accuracy_objective = tf.keras.metrics.BinaryAccuracy()
        elif self.classification == 'multiple':
            self.accuracy_objective = tf.keras.metrics.SparseCategoricalAccuracy()
        elif self.classification == 'regression':
            self.accuracy_objective = tf.keras.metrics.Accuracy()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean()
        self.val_loss = tf.keras.metrics.Mean()
        self.train_accuracy = tf.keras.metrics.Mean()
        self.val_accuracy = tf.keras.metrics.Mean()

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            if self.classification == 'multiple' or self.classification == 'binary':
                train_loss = self.loss_object(labels, predictions)
            elif self.classification == 'regression':
                train_loss = tf.keras.losses.MSE(labels, predictions)
        gradients = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        train_accuracy = self.accuracy_objective(labels, predictions)
        return train_loss, train_accuracy

    @tf.function
    def val_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=True)
        if self.classification == 'multiple' or self.classification == 'binary':
            val_loss = self.loss_object(labels, predictions)
        elif self.classification == 'regression':
            val_loss = tf.keras.losses.MSE(labels, predictions)
        val_accuracy = self.accuracy_objective(labels, predictions)
        return val_loss, val_accuracy

    # @tf.function
    def train(self, epochs, batch_size=32):
        logging.info(f'{self.ds_train}')
        logging.info('Starting')
        self.ckpt.restore(self.manager.latest_checkpoint)

        # If training was interrupted unexpectedly, resume the training process
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        # epochs = 10
        for epoch in range(epochs):
            logging.info("\nStart of epoch %d" % (epoch+1,))
            start_time = time.time()

            for batch_idx, (images, labels) in enumerate(self.ds_train):
                labels = tf.reshape(labels, (-1, 1))
                train_loss_list = []
                train_accuracy_list = []
                loss_value, accuracy_value = self.train_step(images, labels)
                train_loss_list.append(loss_value)
                train_accuracy_list.append(accuracy_value)

            for val_images, val_labels in self.ds_val:
                val_labels = tf.reshape(val_labels, (-1, 1))
                val_loss_list = []
                val_accuracy_list = []
                loss_value, accuracy_value = self.val_step(val_images, val_labels)
                val_loss_list.append(loss_value)
                val_accuracy_list.append(accuracy_value)

            if epoch % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.manager.save()

            if epoch % epochs == 0:
                logging.info(f'Finished training after epochs')
                # Save final checkpoint
                self.manager.save()

            self.train_loss.update_state(train_loss_list)
            train_loss = self.train_loss.result()
            self.val_loss.update_state(val_loss_list)
            val_loss = self.val_loss.result()
            self.train_accuracy.update_state(train_accuracy_list)
            train_accuracy = self.train_accuracy.result()
            self.val_accuracy.update_state(val_accuracy_list)
            val_accuracy = self.val_accuracy.result()

            template = 'epoch {}, Loss: {}, Accuracy: {}%, ' \
                       'Validation Loss: {}, Validation Accuracy: {}%, Time taken: {}'
            logging.info(template.format(epoch + 1,
                                         train_loss,
                                         train_accuracy * 100,
                                         val_loss,
                                         val_accuracy * 100,
                                         time.time() - start_time
                                         )
                         )

            # Write summary to tensorboard
            tf.summary.trace_on(graph=True, profiler=False)
            with self.summary_writer.as_default():
                tf.summary.scalar("train_loss", train_loss, epoch+1)
                tf.summary.scalar("train_accuracy", self.train_accuracy.result() * 100, epoch+1)
                tf.summary.scalar("val_loss", self.val_loss.result(), epoch+1)
                tf.summary.scalar("val_accuracy", self.val_accuracy.result() * 100, epoch+1)
                tf.summary.trace_export(name="Default", step=0,
                                        profiler_outdir=self.run_paths['path_model_Tensorboard'])

            # Reset train metrics
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
