import tensorflow as tf
from IPython.core.display import clear_output


class MOFCnn(tf.keras.Model):
    def __init__(self):
        super(MOFCnn, self).__init__()
        # CNN layers
        self.cnn1 = tf.keras.layers.Conv2D(32, 3, (2, 2), padding='same', input_shape=(28, 28, 1), activation='relu')
        self.cnn2 = tf.keras.layers.Conv2D(64, 3, (2, 2), padding='same', activation='relu')

        # maxpool layers:
        self.maxpool = tf.layers.MaxPooling2D((2, 2), (2, 2))

        # flatten layer:
        self.flatten = tf.layers.Flatten()

        # fully connected layers
        self.dense1 = tf.layers.Dense(100, activation='relu')
        self.denseOutput = tf.layers.Dense(2, activation='softmax')

        # dropout
        self.dropoutFull = tf.layers.Dropout(0.5)
        self.dropoutHalf = tf.layers.Dropout(0.25)

    def call(self, input):
        result = self.cnn1(input)
        result = self.maxpool(result)
        result = self.cnn2(result)
        result = self.maxpool(result)
        result = self.flatten(result)
        result = self.dense1(result)
        result = self.dropoutFull(result)
        result = self.denseOutput(result)
        return result

    @staticmethod
    def loss(model, x, y):
        prediction = model(x)
        return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)

    @staticmethod
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = MOFCnn.loss(model, inputs, targets)
        return tape.gradient(loss_value, model.variables)

    @staticmethod
    def accuracy(logits, labels):
        predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
        labels = tf.cast(labels, tf.int64)
        batch_size = int(logits.shape[0])
        predictions = tf.reshape(predictions, (predictions.shape[0], 1))
        return tf.reduce_sum(
            tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size

    @staticmethod
    def train(model, x, y, batch_size, number_of_epochs):
        optimizer = tf.train.MomentumOptimizer(0.001, 0.9, use_nesterov=True)
        x = tf.data.Dataset.from_tensor_slices(x)
        y = tf.data.Dataset.from_tensor_slices(y)
        data = tf.data.Dataset.zip((x, y)).batch(batch_size)
        global_step = tf.train.get_or_create_global_step()
        for _ in range(number_of_epochs):
            for xs, ys in data:
                clear_output(True)
                with tf.contrib.summary.record_summaries_every_n_global_steps(10, global_step):
                    grads = MOFCnn.grad(model, xs, ys)
                    optimizer.apply_gradients(zip(grads, model.variables), global_step)
                    loss = MOFCnn.loss(model, xs, ys)
                    predictions = model(xs)
                    accuracy = MOFCnn.accuracy(predictions, ys)
                    print("accuracy: {:.3f}".format(accuracy))
                    print("loss: {:.3f}".format(loss))

    @staticmethod
    def test(model, x, y):
        predictions = model(x)
        acc = MOFCnn.accuracy(predictions, y)
        print("test accuracy: {:.3f}".format(acc))
