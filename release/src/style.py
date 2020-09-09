import tensorflow as tf
from tensorflow.python.platform import gfile
import transform
import utils
import numpy as np

class StyleTransfer:

    def __init__(self, shape=None):
        # image transform network
        self.transform = transform.Transform()
        # open session
        soft_config = tf.ConfigProto(allow_soft_placement=True)
        soft_config.gpu_options.allow_growth = True # to deal with large image
        self.sess = tf.Session(config=soft_config)
        self.shape = shape
        self._build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_graph(self):

        # graph input
        self.x = tf.placeholder(tf.float32, shape=self.shape, name='input')
        self.xi = tf.expand_dims(self.x, 0) # add one dim for batch

        # result image from transform-net
        self.y_hat = self.transform.net(self.xi/255.0)
        self.y_hat = tf.squeeze(self.y_hat) # remove one dim for batch
        self.y_hat = tf.clip_by_value(self.y_hat, 0., 255.)

    def load(self, model_path=None):
        # load pre-trained model
        self.saver.restore(self.sess, model_path)

    def run(self, img):
        # load content image
        self.x0 = img
        image = self.sess.run(self.y_hat, feed_dict={self.x: self.x0})
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        # Convert to bytes.
        image = image.astype(np.uint8)
        return image
