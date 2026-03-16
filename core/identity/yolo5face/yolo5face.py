from copy import deepcopy
import os
import cv2
import torch
import torch.nn as nn
from core.identity.yolo5face.model.yolo import Model
from core.identity.yolo5face.utils.data_ops import letterbox, scale_coords_landmarks, show_results, check_img_size, non_max_suppression_face, scale_coords


class YOLO5Face(object):
    def __init__(self, model_path, config_path=None, device=None) -> None:
        self.conf_thres = 0.6
        self.iou_thres = 0.5
        self.imgsz = (640, 640)
        self.img_size = 640

        if device != None:
            if device == "cpu":
                self.device = torch.device("cpu")
            elif isinstance(device, str):
                if device == "cuda" and torch.cuda.is_available():
                     self.device = torch.device("cuda")
                elif "cuda:" in device and torch.cuda.is_available():
                     self.device = torch.device(device)
                else:
                    self.device = torch.device("cpu")
            elif torch.cuda.is_available():
                 # Fallback/default handling matching original logic slightly better
                dev_list = [int(d) for d in str(device).replace("cuda:", "").split(",")] if "cuda" in str(device) else [0]
                self.device = torch.device(dev_list[0])
            else:
                self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        if config_path is None:
            package_dir = os.path.dirname(os.path.abspath(__file__))
            model_config_path = os.path.join(package_dir, "model", "yolo5face.yaml")
        else:
            model_config_path = config_path

        # Always RGB for PURE/Box evaluation
        self.model = Model(cfg=model_config_path, ch=3, nc=None).to(self.device)
        ckpt = model_path

        checkpoint = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint)
        self.model.float().eval()


    def detect_face(self, frame):

        img0 = deepcopy(frame)
        h0, w0 = img0.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size
        img = letterbox(img0, new_shape=imgsz)[0]

        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            # Inference
            pred = self.model(img)[0]
            
            # Apply NMS
            pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)

            if len(pred[0]) > 0:
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if i==0 and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class

                        det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], frame.shape).round()

                        # only process first face.
                        # for j in range(det.size()[0]):
                        xyxy = det[0, :4].view(-1).tolist()

                        x1 = int(xyxy[0])
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])
                        res = [x1, y1, x2, y2]
            else:
                res = None
            
            return res