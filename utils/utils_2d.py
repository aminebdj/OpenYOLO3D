# Copyright (c) Tencent Inc. All rights reserved.
import os
import cv2
import os.path as osp
from torchvision.ops import nms
import torch
from mmengine.runner.amp import autocast
from tqdm import tqdm
import yaml
from PIL import Image
from mmengine.dataset import Compose
from mmyolo.registry import RUNNERS
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
import supervision as sv

def load_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def get_image_resolution(image_path):
    """
    Get the resolution of an image.

    :param image_path: Path to the image file
    :return: A tuple containing the width and height of the image
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

class Network_2D():
    def __init__(self, config):
        self.texts = [[t] for t in config["network2d"]["text_prompts"]] + [[' ']]
        self.topk = config["network2d"]["topk"]
        self.th = config["network2d"]["th"]
        self.nms = config["network2d"]["nms"]
        self.use_amp = config["network2d"]["use_amp"]
        self.resolution = None  
        self.frequency = config["openyolo3d"]["frequency"]
        cfg = Config.fromfile(os.path.join(os.getcwd(), config["network2d"]["config_path"]))
        cfg.work_dir = osp.join(f'{os.getcwd()}/models/YOLO-World/yolo_world/work_dirs',
                                osp.splitext(config["network2d"]["config_path"])[0].split("/")[-1])
        cfg.load_from = os.path.join(os.getcwd(), config["network2d"]["pretrained_path"])
        if 'runner_type' not in cfg:
            self.runner = Runner.from_cfg(cfg)
        else:
            self.runner = RUNNERS.build(cfg)   
        
        self.runner.call_hook('before_run')
        self.runner.load_or_resume()
        pipeline = cfg.test_dataloader.dataset.pipeline
        self.runner.pipeline = Compose(pipeline)
        self.runner.model.eval() 

    def get_bounding_boxes(self, path_2_images, text=None): 
        self.texts = [[t] for t in text] + [[' ']] if text is not None else self.texts 
        print(f"Infering from {len(path_2_images)} images")
        
        scene_preds = {}
        for image_path in tqdm(path_2_images):
            frame_prediction = self.inference_detector([image_path]) 
            scene_preds.update(frame_prediction)
        return scene_preds

    def inference_detector(self, images_batch):
        
        if self.resolution is None:
            self.resolution = get_image_resolution(images_batch[0])
        inputs = []
        data_samples = []
        for img_id, image_path in enumerate(images_batch):
            data_info = dict(img_id=img_id, img_path=image_path, texts=self.texts)
            data_info = self.runner.pipeline(data_info)
            inputs.append(data_info['inputs'])
            data_samples.append(data_info['data_samples'])
        
        
        data_batch = dict(inputs=torch.stack(inputs),
                        data_samples=data_samples)
        
        with autocast(enabled=self.use_amp), torch.no_grad():
            output = self.runner.model.test_step(data_batch)
        frame_prediction = {}

        for img_id, image_path in enumerate(images_batch):
            with autocast(enabled=self.use_amp), torch.no_grad():
                pred_instances = output[img_id].pred_instances
            keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=self.nms)
            pred_instances = pred_instances[keep]
            pred_instances = pred_instances[pred_instances.scores.float() > self.th]
            
            if len(pred_instances.scores) > self.topk:
                indices = pred_instances.scores.float().topk(self.topk)[1]
                pred_instances = pred_instances[indices]
            mask = ~(((pred_instances['bboxes'][:,2]-pred_instances['bboxes'][:,0] > self.resolution[0]-50)*(pred_instances['bboxes'][:,3]-pred_instances['bboxes'][:,1] > self.resolution[1]-50)) == 1)
            bboxes_ = pred_instances['bboxes'][mask].cpu()
            labels_ = pred_instances['labels'][mask].cpu()
            scores_ = pred_instances['scores'][mask].cpu()
            frame_id = osp.basename(image_path).split(".")[0] 
            
            frame_prediction.update({frame_id:{"bbox":bboxes_, "labels":labels_, "scores":scores_}})
        
        return frame_prediction
        
            


