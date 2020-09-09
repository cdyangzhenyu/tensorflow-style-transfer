import sys, os, cv2, time, heapq, argparse
from PIL import Image, ImageFont, ImageDraw
import numpy as np, math
try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin
import multiprocessing as mp
from time import sleep, gmtime, strftime
import threading
import json
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("/var/log/async_multi_vpu.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

processes = []

fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0

cur_path = sys.path[0]

#model_xml = cur_path + "/lrmodels/tangyan/tangyan.xml"
#model_bin = cur_path + "/lrmodels/tangyan/tangyan.bin"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
parser.add_argument('-numncs','--numberofncs',dest='number_of_ncs',type=int,default=1,help='Number of NCS. (Default=1)')
args = parser.parse_args()

model_xml = args.model
model_bin = os.path.splitext(model_xml)[0] + ".bin"

def switchChannel(raw_img, i, j):
    red = raw_img[:,:,i].copy()
    blue = raw_img[:,:,j].copy()
    raw_img[:,:,i] = blue
    raw_img[:,:,j] = red

def saveImg(img, name):
    img *= 255
    #switchChannel(img, 0, 2)
    wantedRatio = 1.48
    border = 50

    bannerName = "banner/" + name + ".jpg"
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

    fname = "save/" + strftime("save_%Y-%m-%d-%H-%M-%S.jpg", gmtime())

    if not os.path.exists('save'):
        os.mkdir('save')
    print("save to %s, %d x %d\n" % (fname, result.shape[1], result.shape[0]))
    
    cv2.imwrite(fname, result)
    cv2.namedWindow("photo",0);
    cv2.resizeWindow("photo", 320, 480);
    #cv2.moveWindow("photo", 910, 480)
    cv2.imshow('photo', result/255.)
    # cv2.imwrite("kkk.jpg", result)


def camThread(results, frameBuffer, camera_width, camera_height, vidfps):
    global fps
    global detectfps
    global framecount
    global detectframecount
    global time1
    global time2
    global send_time
    global cam
    global window_name
    global output_image

    cam = cv2.VideoCapture(0)
    if cam.isOpened() != True:
        logger.info("USB Camera Open Error!!!")
        print("USB Camera Open Error!!!")
        sys.exit(0)
    cam.set(cv2.CAP_PROP_FPS, vidfps)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    
    window_name = "USB Camera"
    wait_key_time = 1

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    #cv2.moveWindow(window_name, 910, 0)
    #cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("result",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    output_image = []
    while True:
        t1 = time.perf_counter()

        # USB Camera Stream Read
        s, color_image = cam.read()
        if not s:
            continue
        if frameBuffer.full():
            frameBuffer.get()

        height = color_image.shape[0]
        width = color_image.shape[1]
        frameBuffer.put(color_image.copy())
        
        if not results.empty():
            detectframecount += 1
            output_image = results.get(False)
            #output_image = np.rot90(output_image, 3)
            #output_image = cv2.resize(output_image, (1080, 1920))
            #cv2.imshow('result', output_image)
            #print(output_image)
            cv2.imshow('result', cv2.resize(output_image, (width, height)))

        cv2.putText(color_image, fps,       (width-140,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(color_image, detectfps, (width-140,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (38,0,255), 1, cv2.LINE_AA)
        
        #color_image = np.rot90(color_image, 3)
        cv2.imshow(window_name, cv2.resize(color_image, (width, height)))
        
        key = cv2.waitKey(wait_key_time)
        if key & 0xFF == ord('q'):
            sys.exit(0)
        
        if key & 0xFF == ord('s'):
            print(output_image)
            saveImg(output_image, 'test')
       
        ## Print FPS
        framecount += 1
        if framecount >= 15:
            fps       = "(Capture) {:.1f} FPS".format(time1/15)
            detectfps = "(Calculate) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime

# l = Search list
# x = Search target value
def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue


def async_infer(ncsworker):

    ncsworker.skip_frame_measurement()

    while True:
        ncsworker.predict_async()


class NcsWorker(object):

    def __init__(self, devid, frameBuffer, results, camera_width, camera_height, number_of_ncs, vidfps):
        self.devid = devid
        self.frameBuffer = frameBuffer
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.m_input_size = 256
        self.threshould = 0.7
        self.num_requests = 4
        self.inferred_request = [0] * self.num_requests
        self.heap_request = []
        self.inferred_cnt = 0
        self.plugin = IEPlugin(device="MYRIAD")
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        self.exec_net = self.plugin.load(network=self.net, num_requests=self.num_requests)
        self.results = results
        self.number_of_ncs = number_of_ncs
        self.predict_async_time = 800
        self.skip_frame = 0
        self.roop_frame = 0
        self.vidfps = vidfps
        self.new_w = int(camera_width * self.m_input_size/camera_width)
        self.new_h = int(camera_height * self.m_input_size/camera_height)

    def image_preprocessing(self, color_image):
        resized_image = cv2.resize(color_image, (self.new_w, self.new_h), interpolation = cv2.INTER_CUBIC)
        canvas = np.full((self.m_input_size, self.m_input_size, 3), 128)
        canvas[(self.m_input_size-self.new_h)//2:(self.m_input_size-self.new_h)//2 + self.new_h,(self.m_input_size-self.new_w)//2:(self.m_input_size-self.new_w)//2 + self.new_w,  :] = resized_image
        prepimg = canvas
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        return prepimg


    def skip_frame_measurement(self):
            surplustime_per_second = (1000 - self.predict_async_time)
            if surplustime_per_second > 0.0:
                frame_per_millisecond = (1000 / self.vidfps)
                total_skip_frame = surplustime_per_second / frame_per_millisecond
                self.skip_frame = int(total_skip_frame / self.num_requests)
            else:
                self.skip_frame = 0


    def predict_async(self):
        try:

            if self.frameBuffer.empty():
                return

            self.roop_frame += 1
            if self.roop_frame <= self.skip_frame:
               self.frameBuffer.get()
               return
            self.roop_frame = 0

            prepimg = self.image_preprocessing(self.frameBuffer.get())
            reqnum = searchlist(self.inferred_request, 0)

            if reqnum > -1:
                self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg})
                self.inferred_request[reqnum] = 1
                self.inferred_cnt += 1
                if self.inferred_cnt == sys.maxsize:
                    self.inferred_request = [0] * self.num_requests
                    self.heap_request = []
                    self.inferred_cnt = 0
                heapq.heappush(self.heap_request, (self.inferred_cnt, reqnum))

            cnt, dev = heapq.heappop(self.heap_request)

            if self.exec_net.requests[dev].wait(0) == 0:
                self.exec_net.requests[dev].wait(-1)
                res = self.exec_net.requests[dev].outputs[self.out_blob][0]
                res = np.swapaxes(res, 0, 2)
                res = np.swapaxes(res, 0, 1)
                res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
                res[res < 0] = 0
                res[res > 255] = 255
                res /= 255
                self.results.put(res)
                self.inferred_request[dev] = 0
            else:
                heapq.heappush(self.heap_request, (cnt, dev))
        except:
            import traceback
            traceback.print_exc(traceback.print_exc())


def inferencer(results, frameBuffer, number_of_ncs, camera_width, camera_height, vidfps):

    # Init infer threads
    threads = []
    for devid in range(number_of_ncs):
        thworker = threading.Thread(target=async_infer, args=(NcsWorker(devid, frameBuffer, results, camera_width, camera_height, number_of_ncs, vidfps),))
        thworker.start()
        threads.append(thworker)

    for th in threads:
        th.join()


if __name__ == '__main__':
    number_of_ncs = args.number_of_ncs
    camera_width = 320
    camera_height = 240
    vidfps = 10

    try:

        mp.set_start_method('forkserver')
        frameBuffer = mp.Queue(10)
        results = mp.Queue()

        # Start detection MultiStick
        # Activation of inferencer
        p = mp.Process(target=inferencer, args=(results, frameBuffer, number_of_ncs, camera_width, camera_height, vidfps), daemon=True)
        p.start()
        processes.append(p)

        sleep(number_of_ncs * 7)

        # Start streaming
        p = mp.Process(target=camThread, args=(results, frameBuffer, camera_width, camera_height, vidfps), daemon=True)
        p.start()
        processes.append(p)

        while True:
            sleep(1)

    except:
        import traceback
        traceback.print_exc()
    finally:
        for p in range(len(processes)):
            processes[p].terminate()

        print("\n\nFinished\n\n")