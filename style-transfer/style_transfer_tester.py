import tensorflow as tf
from tensorflow.python.platform import gfile
import transform
import time

class StyleTransferTester:

    def __init__(self, session, content_image, model_path):
        # session
        self.sess = session

        # input images
        self.x0 = content_image

        # input model
        self.model_path = model_path

        # image transform network
        self.transform = transform.Transform()

        # build graph for style transfer
        self._build_graph()

    def _build_graph(self):

        # graph input
        self.x = tf.placeholder(tf.float32, shape=self.x0.shape, name='input')
        self.xi = tf.expand_dims(self.x, 0) # add one dim for batch

        # result image from transform-net
        self.y_hat = self.transform.net(self.xi/255.0)
        self.y_hat = tf.squeeze(self.y_hat) # remove one dim for batch
        self.y_hat = tf.clip_by_value(self.y_hat, 0., 255.)

    def test(self):

        # initialize parameters
        self.sess.run(tf.global_variables_initializer())

        if '.ckpt' in self.model_path:
            # load pre-trained model
            saver = tf.train.Saver()
            saver.restore(self.sess, self.model_path)
        # get transformed image
        start = time.time()
        output = self.sess.run(self.y_hat, feed_dict={self.x: self.x0})
        print('Inter time : %f msec' % (1000.*float(time.time() - start)))
        return output
