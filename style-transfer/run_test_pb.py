import tensorflow as tf
import os
import utils
import argparse
import time
from tensorflow.python.platform import gfile
import transform
import numpy as np
"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--style_model', type=str, default='models/wave.pb', help='location for model file (*.pb)',
                        required=True)

    parser.add_argument('--content', type=str, default='content/female_knight.jpg',
                        help='File path of content image (notation in the paper : x)', required=True)

    parser.add_argument('--output', type=str, default='result.jpg',
                        help='File path of output image (notation in the paper : y_c)', required=True)

    parser.add_argument('--max_size', type=int, default=None, help='The maximum width or height of input images')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --style_model
    if '.ckpt' in args.style_model:
        try:
            #Tensorflow r0.12 requires 3 files related to *.ckpt
            assert os.path.exists(args.style_model + '.index') and os.path.exists(args.style_model + '.meta') and os.path.exists(
                args.style_model + '.data-00000-of-00001')
        except:
            print('There is no %s'%args.style_model)
            print('Tensorflow r0.12 requires 3 files related to *.ckpt')
            print('If you want to restore any models generated from old tensorflow versions, this assert might be ignored')
            return None
    elif '.pb' in args.style_model:
        try:
            assert os.path.exists(args.style_model)
        except:
            print('There is no %s'%args.style_model)
            return None
    # --content
    try:
        assert os.path.exists(args.content)
    except:
        print('There is no %s' % args.content)
        return None

    # --max_size
    try:
        if args.max_size is not None:
            assert args.max_size > 0
    except:
        print('The maximum width or height of input image must be positive')
        return None

    # --output
    dirname = os.path.dirname(args.output)
    try:
        if len(dirname) > 0:
            os.stat(dirname)
    except:
        os.mkdir(dirname)

    return args

"""main"""
def main():

    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    # open session
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True # to deal with large image
    soft_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=soft_config)
    # load content image
    content_image = utils.load_image(args.content, shape=(256, 256), max_size=args.max_size)
    #content_image = utils.load_image(args.content, max_size=args.max_size)
    shape = content_image.shape #(batch, width, height, channel)
    
    sess.run(tf.global_variables_initializer())
    with gfile.FastGFile(args.style_model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    #for op in sess.graph.get_operations():
    #    print(op.name)
        
    x = tf.get_default_graph().get_tensor_by_name("content:0")
    y_hat = tf.get_default_graph().get_tensor_by_name("output:0")
    start_time = time.time()
    output_image = sess.run(y_hat, feed_dict={x: [content_image]})
    end_time = time.time()
    
    output_image = np.squeeze(output_image[0])  # remove one dim for batch
    output_image = np.clip(output_image, 0., 255.)

    # save result
    utils.save_image(output_image, args.output)

    # report execution time
    print('Execution time for a %d x %d image : %f msec' % (shape[0], shape[1], 1000.*float(end_time - start_time)))

if __name__ == '__main__':
    main()