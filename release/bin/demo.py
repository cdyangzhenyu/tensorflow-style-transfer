import tensorflow as tf
import os
import utils
import style
import cv2
import numpy as np
import time
from time import gmtime, strftime
import shutil  

def switchChannel(raw_img, i, j):
    red = raw_img[:,:,i].copy()
    blue = raw_img[:,:,j].copy()
    raw_img[:,:,i] = blue
    raw_img[:,:,j] = red

def adjustImg(img):
    temp = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    #temp = cv2.resize(temp, (480, 640))
    # temp = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imshow("raw", temp)
    switchChannel(temp, 0, 2)
    return np.float32(temp)

def loadImg(name):
    raw_img = cv2.imread(name, 1)
    switchChannel(raw_img, 0, 2)
    
    small_img = cv2.resize(raw_img, (640, 480))
    content_image = np.float32(small_img)  

    return content_image

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
    cropResult = cv2.resize(cropResult, (1080, 1920))
    cv2.imshow('result', cropResult)
    adjustImg# cv2.waitKey(-1)

def overlay(lower, upper):
    lowerW = lower.shape[1]
    lowerH = lower.shape[0]

    upperW = upper.shape[1]
    upperH = upper.shape[0]

    upperRatio = upperW / upperH
    newUpperH = lowerW

def testImg():
    imgChicago = loadImg("content/chicago.jpg")
    imgKnight = loadImg("content/female_knight.jpg")
    model = "models/udnie.ckpt"

    transfer = style.StyleTransfer(shape=imgKnight.shape)
    transfer.load(model_path=model)

    img1 = transfer.run(imgKnight)

    startTime = time.time()
    img0 = transfer.run(imgChicago)
    img1 = transfer.run(imgKnight)
    stopTime = time.time()

    print("time is %f" % (stopTime - startTime))

    showImg(img1)

    utils.save_image(img0, "img0.jpg")
    utils.save_image(img1, "img1.jpg")

def saveImg(img, name):

    switchChannel(img, 0, 2)
    wantedRatio = 1.48
    border = 50

    bannerName = "../banner/" + name + ".jpg"
    banner = cv2.imread(bannerName, 1)

    print("banner %s is %d x %d\n" % (bannerName, banner.shape[1], banner.shape[0]))
    
    newWidth = banner.shape[1]
    newHeight = int((float(newWidth) / img.shape[1]) * img.shape[0])
    resized = cv2.resize(img, (newWidth, newHeight))
    
    print("resize img to %d x %d\n" % (resized.shape[1], resized.shape[0]))

    totalHeight = newHeight + banner.shape[0]
    wantedHeight = int(newWidth * wantedRatio)

    cropHeight = (newHeight - (totalHeight - wantedHeight))

    croped = resized[0: cropHeight, 0:newWidth]
    print("crop img to %d x %d\n" % (croped.shape[1], croped.shape[0]))

    concated = np.concatenate((croped, banner), axis=0)
    
    result = cv2.copyMakeBorder(concated, border, border, border, border, cv2.BORDER_CONSTANT, None, [255, 255, 255])

    fname = "../save/" + strftime("save_%Y-%m-%d-%H-%M-%S.jpg", gmtime())

    if not os.path.exists('../save'):
        os.mkdir('../save')
    print("save to %s, %d x %d\n" % (fname, result.shape[1], result.shape[0]))
    
    cv2.imwrite(fname, result)
    cv2.namedWindow("photo",0);
    cv2.resizeWindow("photo", 320, 480);
    cv2.moveWindow("photo", 910, 600)
    cv2.imshow('photo', result)
    # cv2.imwrite("kkk.jpg", result)

def camera():
    cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("result",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("raw",0);
    cv2.resizeWindow("raw", 320, 480);
    cv2.moveWindow("raw", 910, 0)
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 10)

    firstRun = False
    transfer =  None

    models = [
        "udnie",
        "la_muse",
        "rain_princess",
        "shipwreck",
        "wave",
        "the_scream",
        "tangyan",
        "shuimo",
    ]

    idx = 0
    index = 0
    while(True):
        index += 1
        all_start_time = time.time()
        ret, frame = cap.read()
        if(not ret):
            print('cap img error')
            continue
        img = adjustImg(frame)

        if(not firstRun):
            firstRun = True
            transfer = transfer = style.StyleTransfer(shape=img.shape)
            transfer.load( model_path = "../models/" + models[idx] + ".ckpt")
            result = transfer.run(img)
        
        infer_start_time = time.time()
        result = transfer.run(img)
        infer_end_time = time.time()
        print('Infer time for a %d x %d image : %f msec' % (img.shape[0], img.shape[1], 1000.*float(infer_end_time - infer_start_time)))
        showImg(result)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        if key & 0xFF == ord('n') or key & 0xFF == ord(' '):            
            idx = (idx + 1) % len(models)
            transfer.load( model_path = "../models/" + models[idx] + ".ckpt")
            print("switching to " + models[idx])
        if key & 0xFF == ord('m') or key & 0xFF == ord(' '):
            idx = (idx - 1) % len(models)
            transfer.load( model_path = "../models/" + models[idx] + ".ckpt")
            print("switching to " + models[idx])
        if key & 0xFF == ord('s'):
            # utils.save_image(result, "save.jpg")
            saveImg(result, models[idx])
        if key & 0xFF == ord('c'):
            # clean all files
            try:
                shutil.rmtree('../save')
            except:
                print('rm save error')
        all_end_time = time.time()
        print('All time for a %d x %d image : %f msec, index: %s' % (img.shape[0], img.shape[1], 1000.*float(all_end_time - all_start_time), index))

    cap.release() 
    cv2.destroyAllWindows()    
"""main"""
def main():
    #testImg()
    camera()

if __name__ == '__main__':
    main()
    

