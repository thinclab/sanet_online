#!/usr/bin/env python3
# coding: utf8
'''
This file was modified from the original in order to get it working
with ROS by Prasanth Suresh(ps32611@uga.edu).
Please make sure you provide credit if you are using this code.
'''

import torch.backends.cudnn as cudnn

import time

import rospkg
import argparse
import torch.backends.cudnn as cudnn
from utils import google_utils
from utils.datasets import *
from utils.utils import *


class YOLO():
    def __init__(self, weightsfile = 'best_realkinect.pt', conf_thres = 0.8):
        
        rospack = rospkg.RosPack()  # get an instance of RosPack with the default search paths
        path = rospack.get_path('sanet_online/other_scripts')   # get the file path for sanet_online
        self.weights = path + '/weights/'+ weightsfile
        self.source = path + '/inference/images'
        self.output = path + '/inference/output'
        self.img_size = 640
        self.conf_thres = conf_thres
        self.iou_thres = 0.3
        self.fourcc = 'mp4v'
        self.device = ''
        self.view_img = False
        self.save_txt = False
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.bounding_boxes = []

    def detect(self, Image = None):
        out, source, weights, view_img, save_txt, imgsz = \
            self.output, self.source, self.weights, self.view_img, self.save_txt, self.img_size
        webcam = source == '0' or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = torch_utils.select_device(self.device)
        # if os.path.exists(out):
        #     shutil.rmtree(out)  # delete output folder
        # os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        google_utils.attempt_download(weights)
        model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
        # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
        # model.fuse()
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(
                name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)]
                for _ in range(len(names))]

        # Run inference
        t0 = time.time()

        # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

        # # run once
        # _ = model(img.half() if half else img) if device.type != 'cpu' else None

        im0s = Image

        img = letterbox(im0s, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # # for path, img, im0s, vid_cap in dataset:
        vid_cap = None
        path = 'img.jpg'
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
            
        # print('**** pred: ',pred)
        ''' Sorting the bounding boxes according to descending x values '''
        if pred[0] != None:
            pred[0] = pred[0].cpu().numpy()
            pred[0] = pred[0][-pred[0][:,0].argsort()]
            pred[0] = torch.from_numpy(pred[0])
            # print('**** Sorted pred: \n',pred)
        # Process detections
        self.bounding_boxes = []
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            
            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            #  normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0s size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                minx = 0
                miny = 0
                maxx = 0
                maxy = 0

                for *xyxy, conf, cls in det:

                    tlx, tly, brx, bry = int(xyxy[0]), int(
                        xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    # print(f"tlx: {tlx}, tly: {tly}, brx: {brx}, bry: {bry}")

                    if tlx >= 840:   # NOTE: WITHIN THE SORTING REGION, OTHERWISE ALREADY SORTED
                        # print("Found onions in sorting area: ")
                        self.bounding_boxes.append([tlx, tly, brx, bry, conf, cls])

                    if tlx < minx:
                        minx = tlx
                    if tly < miny:
                        miny = tly
                    if bry > maxy:
                        maxy = bry
                    if brx > maxx:
                        maxx = brx  # crop_img = img[y:y+h, x:x+w]

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                        ) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') %
                                    (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                    color=colors[int(cls)], line_thickness=3)

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*self.fourcc), fps, (w, h))
                    vid_writer.write(im0)

        if save_txt or save_img:
            # print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)

        if not len(self.bounding_boxes) > 0:
            self.bounding_boxes.append(torch.tensor([-100000]))
        return self.bounding_boxes
