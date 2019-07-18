import tensorflow as tf
import numpy as np
from models import ConvNet

def train():
    g = ConvNet(is_training=True)

    # Start train
    with tf.Session(graph=g.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        if tf.train.latest_checkpoint('checkpoint'):
            saver.restore(sess, tf.train.latest_checkpoint('checkpoint'))
            print("Loaded parameter from {}".format(tf.train.latest_checkpoint('checkpoint')))

        n_batches = g.mnist.train.num_examples // g.batch_size
        print("Start to train")
        for i in range(10):
            total_loss = 0

            for j in range(n_batches):
                X_batch, Y_batch = g.mnist.train.next_batch(g.batch_size)

                X_batch = np.reshape(X_batch, [-1, 28, 28, 1])

                _, loss_batch = sess.run([g.optimizer, g.loss], {g.X: X_batch, g.Y: Y_batch})
                total_loss += loss_batch
            print('Epoch {}: {}'.format(i + 1, total_loss / n_batches))

        X_test = np.reshape(g.mnist.test.images, [-1, 28, 28, 1])
        print('Accuracy:', sess.run(g.accuracy, feed_dict={g.X: X_test, g.Y: g.mnist.test.labels}))
        saver.save(sess, 'checkpoint/model')

if __name__ == '__main__':
    train()