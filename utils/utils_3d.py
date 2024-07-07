
import sys
sys.path.append("..")
from models.Mask3D.mask3d import get_model, load_mesh, prepare_data, map_output_to_pointcloud, save_colorized_mesh 
import torch

class Network_3D():
    def __init__(self, config):
        self.model = get_model(config["network3d"]["pretrained_path"])
        self.model.eval()
        self.device = torch.device("cuda:0")
        self.model.to(self.device)
    
    def get_class_agnostic_masks(self, pointcloud_file, datatype="point cloud", point2segment=None):
        data, points, colors, features, unique_map, inverse_map, point2segment, point2segment_full = prepare_data(pointcloud_file, datatype, self.device)
        with torch.no_grad():
            outputs = self.model(data, raw_coordinates=features, point2segment=[point2segment] if point2segment is not None else None)
        return map_output_to_pointcloud(outputs, inverse_map, point2segment, point2segment_full)
        