from detect import detect
from time import time

def time_detect(image_size):
    t1 = time()
    detect( "coors_boa_dataset/custom_yolov3.cfg",
            "coors_boa_dataset/dataset.data",
            "weights/best.pt", 
            "W:/", 
            "W:/inference_out" + str(image_size), 
            img_size=image_size,
            conf_thres=0.4,
            nms_thres=0.4,)
    t2 = time()
    return t2-t1

t3 = time_detect(416)

print("Time for image size {}: {}".format(416, t3))
