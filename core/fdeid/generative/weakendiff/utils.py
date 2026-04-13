# =============================================================================
# Import required libraries
# =============================================================================
import os
import time
import cv2


# print("DEBUG: Importing utils.py...")
import torch
from core.identity.retinaface import FaceDetector

from assets.models import irse, ir152, facenet
# print("DEBUG: Finished imports in utils.py")



class MyTimer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.


_detector = None


def get_detector():
    global _detector
    if _detector is None:
        default_path = "./weight/retinaface_pre_trained/Resnet50_Final.pth"
        model_path = os.environ.get("DETECTOR_MODEL", default_path)
        if not os.path.exists(model_path):
            print(f"Warning: RetinaFace model not found at {model_path}")
        _detector = FaceDetector(model_path=model_path, device='cuda')
    return _detector


def alignment(image):
    detector = get_detector()
    results = detector.detect(image)
    if results:
        return results[0].bbox
    raise ValueError("No face detected by RetinaFace")


def load_FR_models(args, model_names):
    FR_models = {}
    models_dir = os.path.join(args.assets_path, 'models')
    if not os.path.exists(models_dir):
        models_dir = os.path.join(args.assets_path, 'face_recognition_models')

    for model_name in model_names:
        if model_name == 'ir152':
            FR_models[model_name] = []
            FR_models[model_name].append((112, 112))
            fr_model = ir152.IR_152((112, 112))
            fr_model.load_state_dict(torch.load(
                os.path.join(models_dir, 'ir152.pth'), weights_only=True))
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)
        if model_name == 'irse50':
            FR_models[model_name] = []
            FR_models[model_name].append((112, 112))
            fr_model = irse.Backbone(50, 0.6, 'ir_se')
            fr_model.load_state_dict(torch.load(
                os.path.join(models_dir, 'irse50.pth'), weights_only=True))
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)
        if model_name == 'facenet':
            FR_models[model_name] = []
            FR_models[model_name].append((160, 160))
            fr_model = facenet.InceptionResnetV1(
                num_classes=8631, device=args.device)
            fr_model.load_state_dict(torch.load(
                os.path.join(models_dir, 'facenet.pth'), weights_only=True))
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)
        if model_name == 'mobile_face':
            FR_models[model_name] = []
            FR_models[model_name].append((112, 112))
            fr_model = irse.MobileFaceNet(512)
            fr_model.load_state_dict(torch.load(
                os.path.join(models_dir, 'mobile_face.pth'), weights_only=True))
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)
        
        # New models for evaluation
        if model_name == 'arcface_r100':
            FR_models[model_name] = []
            FR_models[model_name].append((112, 112))
            # ArcFace R100 is typically irse.Backbone(100, 0.6, 'ir_se') or based on implementation
            # Checking commonly used MS1MV3 ArcFace R100
            fr_model = irse.Backbone(100, 0.5, 'ir_se') 
            fr_model.load_state_dict(torch.load(args.arcface_model_path, weights_only=True))
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)

        if model_name == 'cosface_r50':
            FR_models[model_name] = []
            FR_models[model_name].append((112, 112))
            # CosFace R50 usually IR50
            fr_model = irse.Backbone(50, 0.5, 'ir_se')
            fr_model.load_state_dict(torch.load(args.cosface_model_path, weights_only=True))
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)

        if model_name == 'adaface_ir50':
            FR_models[model_name] = []
            FR_models[model_name].append((112, 112))
            # AdaFace IR50
            fr_model = irse.Backbone(50, 0.6, 'ir_se')
            # AdaFace might need state_dict['state_dict'] handling if it's a Lightning checkpoint
            ckpt = torch.load(args.adaface_model_path, weights_only=True)
            if 'state_dict' in ckpt:
                state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
                fr_model.load_state_dict(state_dict)
            else:
                fr_model.load_state_dict(ckpt)
            fr_model.to(args.device)
            fr_model.eval()
            FR_models[model_name].append(fr_model)
            
    return FR_models


def preprocess(im, mean, std, device):
    if len(im.size()) == 3:
        im = im.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    elif len(im.size()) == 4:
        im = im.transpose(1, 3).transpose(2, 3)
    mean = torch.tensor(mean).to(device)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).to(device)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im = (im - mean) / std
    return im


def read_img(data_dir, mean, std, device):
    img = cv2.imread(data_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
    img = torch.from_numpy(img).to(torch.float32).to(device)
    img = preprocess(img, mean, std, device)
    return img


def get_target_test_images(target_choice, device, MTCNN_cropping, assets_path):
    target_dir = os.path.join(assets_path, 'datasets', 'target')
    test_dir = os.path.join(assets_path, 'datasets', 'test')
    # Fallback if datasets subdir doesn't exist (original structure)
    if not os.path.exists(target_dir):
        target_dir = os.path.join(assets_path, 'target_images')
        test_dir = os.path.join(assets_path, 'test_images')
    
    obfs_dir = os.path.join(assets_path, 'obfs_target_images')

    # Impersonation
    if target_choice == '1':
        target_image = read_img(
            os.path.join(target_dir, '005869.jpg'), 0.5, 0.5, device)
        test_image = read_img(
            os.path.join(test_dir, '008793.jpg'), 0.5, 0.5, device)
        if MTCNN_cropping:
            target_image = target_image[:, :, 168:912, 205:765]
            test_image = test_image[:, :, 145:920, 202:775]
    elif target_choice == '2':
        target_image = read_img(
            os.path.join(target_dir, '085807.jpg'), 0.5, 0.5, device)
        test_image = read_img(
            os.path.join(test_dir, '047073.jpg'), 0.5, 0.5, device)
        if MTCNN_cropping:
            target_image = target_image[:, :, 187:891, 244:764]
            test_image = test_image[:, :, 234:905, 266:791]
    elif target_choice == '3':
        target_image = read_img(
            os.path.join(target_dir, '116481.jpg'), 0.5, 0.5, device)
        test_image = read_img(
            os.path.join(test_dir, '055622.jpg'), 0.5, 0.5, device)
        if MTCNN_cropping:
            target_image = target_image[:, :, 214:955, 188:773]
            test_image = test_image[:, :, 185:931, 198:780]
    elif target_choice == '4':
        target_image = read_img(
            os.path.join(target_dir, '169284.jpg'), 0.5, 0.5, device)
        test_image = read_img(
            os.path.join(test_dir, '166607.jpg'), 0.5, 0.5, device)
        if MTCNN_cropping:
            target_image = target_image[:, :, 173:925, 233:792]
            test_image = test_image[:, :, 172:917, 219:779]
    # Obfuscation
    elif target_choice == '5':
        target_image = read_img(
            os.path.join(obfs_dir, '0808002.png'), 0.5, 0.5, device)
        test_image = None
    else:
        raise ValueError(
            "Invalid target choice!")
    return target_image, test_image
