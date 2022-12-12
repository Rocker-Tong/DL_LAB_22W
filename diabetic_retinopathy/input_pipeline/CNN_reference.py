class CNNModel(k.Model):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(16, 3, strides=2, padding="same", activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation=tf.nn.relu)
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation=tf.nn.relu)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.d1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.d2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.d3 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)


    def call(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.maxpool1(out)
        out = self.conv3(out)
        out = self.flatten1(out)
        out = self.dropout1(out)
        out = self.d1(out)
        out = self.d2(out)
        out = self.dropout2(out)
        out = self.d3(out)
        return out


learning_rate = 0.001
model = CNNModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate)

# 损失与评估
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    # 计算梯度
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)  # update

EPOCHS = 10

for epoch in range(EPOCHS):
    # 重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()

    for images, labels in train_dataset:
        images =np.expand_dims(images, axis=-1)
        # images = tf.reshape(images, (256, 256, 3))
        # images = tf.reshape([images.shape[0], [1]] + list(images.shape[1:]))
        train_step(images, labels)

    template = 'Epoch {}, Loss : {}, Accuracy : {}'

    # 打印
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          ))
