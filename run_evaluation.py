
import torch
from tqdm import tqdm
import argparse
from evaluate import SCENE_NAMES_REPLICA, SCENE_NAMES_SCANNET200, evaluate_scannet200, evaluate_replica
from utils import OpenYolo3D
import yaml
import os.path as osp

class InstSegEvaluator():
    def __init__(self, dataset_type):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type

    def evaluate_full(self, preds, scene_gt_dir, dataset, output_file='temp_output.txt', pretrained_on_scannet200=True):
        if dataset == "replica":
            inst_AP = evaluate_replica(preds, scene_gt_dir, output_file=output_file, dataset=dataset)
        elif dataset == "scannet200":
            inst_AP = evaluate_scannet200(preds, scene_gt_dir, output_file=output_file, dataset=dataset, pretrained_on_scannet200 = pretrained_on_scannet200)
        else:
            print("DATASET NOT SUPPORTED!")
            exit()
        return inst_AP

def test_pipeline_full(dataset_type, path_to_3d_masks, is_gt):
    config = load_yaml(osp.join(f'./pretrained/config_{dataset_type}.yaml'))
    path_2_dataset = osp.join('./data', dataset_type)
    gt_dir = osp.join('./data', dataset_type, 'ground_truth')
    depth_scale = config["openyolo3d"]["depth_scale"]
    
    if dataset_type == "replica":
        scene_names = SCENE_NAMES_REPLICA
        datatype="point cloud"
    elif dataset_type == "scannet200":
        scene_names = SCENE_NAMES_SCANNET200
        datatype="mesh"
        
    evaluator = InstSegEvaluator(dataset_type)
    openyolo3d = OpenYolo3D(f"./pretrained/config_{dataset_type}.yaml")
    predictions = {}
    for scene_name in tqdm(scene_names):
        scene_id = scene_name.replace("scene", "")
        processed_file = osp.join(path_2_dataset, scene_name, f"{scene_id}.npy") if dataset_type == "scannet200" else None
        prediction = openyolo3d.predict(path_2_scene_data = osp.join(path_2_dataset, scene_name), 
                                        depth_scale = depth_scale,
                                        datatype = datatype, 
                                        processed_scene = processed_file,
                                        path_to_3d_masks = path_to_3d_masks,
                                        is_gt = is_gt)
        predictions.update(prediction)
    
    preds = {}
    print("Evaluation ...")
    for scene_name in tqdm(scene_names):
        preds[scene_name] = {
            'pred_masks': predictions[scene_name][0].cpu().numpy(),
            'pred_scores': torch.ones_like(predictions[scene_name][2]).cpu().numpy(),
            'pred_classes': predictions[scene_name][1].cpu().numpy()}

    inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset=dataset_type)

def load_yaml(path):
    with open(path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='scannet200', type=str, help='Name of the dataset [replica, scannet200]')
    parser.add_argument('--path_to_3d_masks', default=None, type=str, help='Path to pre computed 3d masks')
    parser.add_argument('--is_gt', default=False, action=argparse.BooleanOptionalAction, help='If pre computed 3d masks are ground truth masks')
    opt = parser.parse_args() 
    test_pipeline_full(opt.dataset_name, opt.path_to_3d_masks, opt.is_gt)
       