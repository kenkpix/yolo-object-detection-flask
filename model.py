# -*- coding: utf-8 -*-

import argparse
import time

from tool.darknet2pytorch import Darknet
from tool.torch_utils import *
from tool.utils import *

"""hyper parameters"""
use_cuda = True


def detect_cv2(cfgfile, weightfile, imgfile, savename):
    import cv2

    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = "data/voc.names"
    elif num_classes == 80:
        namesfile = "data/coco.names"
    else:
        namesfile = "data/x.names"
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print("%s: Predicted in %f seconds." % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename=savename, class_names=class_names)


def detect_cv2_camera(cfgfile, weightfile, videofile, savename):
    import cv2

    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)

    if use_cuda:
        m.cuda()

    cap = cv2.VideoCapture(videofile)
    cap.set(3, 1280)
    cap.set(4, 720)

    size = (int(cap.get(3)), int(cap.get(4)))
    out = cv2.VideoWriter(
        savename, cv2.VideoWriter_fourcc(*'VP90'), 30, size
    )

    # out = cv2.VideoWriter(
    #     savename, -1, 30, size
    # )
    print("Starting the YOLO loop...")

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = "data/voc.names"
    elif num_classes == 80:
        namesfile = "data/coco.names"
    else:
        namesfile = "data/x.names"
    class_names = load_class_names(namesfile)

    while cap.isOpened():
        ret, img = cap.read()
        if ret == True:
            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            start = time.time()
            # default 0.4, 0.6
            boxes = do_detect(m, sized, 0.35, 0.5, use_cuda)
            finish = time.time()
            print("Predicted in %f seconds." % (finish - start))

            result_img = plot_boxes_cv2(
                img, boxes[0], savename=None, class_names=class_names
            )

            out.write(result_img)
            cv2.imshow("Yolo demo", result_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()


def detect_skimage(m, imgfile):
    from skimage import io
    from skimage.transform import resize

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = "data/voc.names"
    elif num_classes == 80:
        namesfile = "data/coco.names"
    else:
        namesfile = "data/x.names"
    class_names = load_class_names(namesfile)

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print("%s: Predicted in %f seconds." % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes, savename="predictions.jpg", class_names=class_names)


# detect_cv2('cfg/yolov4-tiny.cfg', 'weights/yolov4-tiny.weights', 'test.jpg')
# detect_cv2_camera('cfg/yolov4-tiny.cfg', 'weights/yolov4-tiny.weights', 'test_videos/obolon_test.mp4')
