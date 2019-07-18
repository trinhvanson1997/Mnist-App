import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class ConvNet:
    def __init__(self, is_training=True):
        if is_training:
            self.mnist = input_data.read_data_sets('mnist/', one_hot=True)
        self.learning_rate = 0.01
        self.n_epochs = 10
        self.batch_size = 128

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Create placeholder
            X = tf.placeholder(tf.float32, [None, 784], 'X')
            self.X = tf.reshape(X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Build model CNN
            conv1 = tf.layers.conv2d(inputs=self.X,
                                     filters=32,
                                     kernel_size=[5, 5],
                                     padding='SAME',
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                            pool_size=[2, 2],
                                            strides=2
                                            )
            conv2 = tf.layers.conv2d(inputs=pool1,
                                     filters=64,
                                     kernel_size=[5, 5],
                                     padding='SAME',
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                            pool_size=[2, 2],
                                            strides=2)

            pool2 = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])

            fc = tf.layers.dense(inputs=pool2,
                                 units=1024,
                                 activation=tf.nn.relu)

            dropout = tf.layers.dropout(inputs=fc,
                                        rate=0.75)

            self.logits = tf.layers.dense(inputs=dropout,
                                          units=10)
            # Create optimize function
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
            self.loss = tf.reduce_mean(entropy)
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

            # Predict accuracy
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
