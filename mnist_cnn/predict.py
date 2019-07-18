import tensorflow as tf
import numpy as np
from mnist_app.mnist_cnn.models import ConvNet
import cv2


def predict(img_path=None, checkpoint_path=None):
    # Convert image to numpy array size [1, 28, 28, 1]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = np.asarray(img, dtype=np.int32)
    img = np.reshape(img, [1, 28, 28, 1])

    # Load model trained and make prediction
    g = ConvNet(is_training=False)

    with tf.Session(graph=g.graph) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        result = sess.run(
            tf.argmax(g.logits, 1), feed_dict={g.X: img})
    return int(result[0])

if __name__ == '__main__':
    predict()