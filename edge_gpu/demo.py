import tensorflow as tf
import os
import argparse
import time
from tensorflow.python.platform import gfile
import numpy as np
import cv2

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'Perceptual Losses for Real-Time Style Transfer and Super-Resolution'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--style_model', type=str, default='models/wave.pb', help='location for model file (*.pb)',
                        required=True)

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

    return args

def switchChannel(raw_img, i, j):
    red = raw_img[:,:,i].copy()
    blue = raw_img[:,:,j].copy()
    raw_img[:,:,i] = blue
    raw_img[:,:,j] = red

def cropImg(img, rationW, rationH):

    width = img.shape[1]
    height = img.shape[0]
    unit = height / rationH
    newWidth = unit * rationW

    # print ("new width %d " % (newWidth))
    if newWidth > width:
        print("Error!!! can not crop width")

    diff = (width - newWidth) / 2
    left = diff
    right = width - diff

    # (left, 0) -> (right, height)
    return img[0:int(height), int(left):int(right)]

def showImg(img):
    img = np.clip(img, 0.0, 255.0)
    img = img.astype(np.uint8)

    switchChannel(img, 0, 2)

    cropResult = cropImg(img, 9, 16)
    
    cv2.imshow('result', cropResult)

"""main"""
def main():

    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    # open session
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True # to deal with large image
    sess = tf.Session(config=soft_config)
    # load content image
    #content_image = utils.load_image(args.content, shape=(256, 256), max_size=args.max_size)
    #content_image = utils.load_image(args.content, max_size=args.max_size)
    #shape = content_image.shape #(batch, width, height, channel)
    
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
    _, h, w, c = x.shape
    cap = cv2.VideoCapture(0)

    idx = 0
    while(True):
        all_start_time = time.time()
        ret, frame = cap.read()
        if(not ret):
            print('cap img error')
            continue
        #img = adjustImg(frame)
        cv2.imshow("raw", cv2.resize(frame, (455, 256))
        img = cv2.resize(frame, (w, h))
        infer_start_time = time.time()
        output_image = sess.run(y_hat, feed_dict={x: [img]})
        infer_end_time = time.time()
        print('Infer time for a %d x %d image : %f msec' % (img.shape[0], img.shape[1], 1000.*float(infer_end_time - infer_start_time)))
        
        output_image = np.squeeze(output_image[0])  # remove one dim for batch
        image = np.clip(output_image, 0., 255.)

        # Convert to bytes.
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (455, 256))
        cv2.imshow('result', image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release() 
    cv2.destroyAllWindows()  

if __name__ == '__main__':
    main()
