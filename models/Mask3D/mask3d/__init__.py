import hydra
import torch
from torch_scatter import scatter_mean

from mask3d.models.mask3d import Mask3D
from mask3d.utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)


    def forward(self, x, raw_coordinates=None, point2segment=None):
        return self.model(x, raw_coordinates=raw_coordinates, point2segment=point2segment)
    

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose

# imports for input loading
import albumentations as A
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d

# imports for output
from mask3d.datasets.scannet200.scannet200_constants import (VALID_CLASS_IDS_20, VALID_CLASS_IDS_200, SCANNET_COLOR_MAP_20, SCANNET_COLOR_MAP_200)

def get_model(checkpoint_path=None, dataset_name = "scannet200"):


    # Initialize the directory with config files
    with initialize(config_path="conf"):
        # Compose a configuration
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    cfg.general.checkpoint = checkpoint_path

    # would be nicd to avoid this hardcoding below
    # dataset_name = checkpoint_path.split('/')[-1].split('_')[0]
    if dataset_name == 'scannet200':
        cfg.general.num_targets = 201
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 300
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.001

        # # data
        cfg.data.num_labels = 200
        cfg.data.test_mode = "test"

        # # model
        cfg.model.num_queries = 150
        
    if dataset_name == 'scannet':
        cfg.general.num_targets = 19
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 300
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.001

        # # data
        cfg.data.num_labels = 20
        cfg.data.test_mode = "test"

        # # model
        cfg.model.num_queries = 150
        
        #TODO: this has to be fixed and discussed with Jonas
        # cfg.model.scene_min = -3.
        # cfg.model.scene_max = 3.

    # # Initialize the Hydra context
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize(config_path="conf")

    # Load the configuration
    # cfg = hydra.compose(config_name="config_base_instance_segmentation.yaml")
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


def load_mesh(pcl_file):
    
    # load point cloud
    input_mesh_path = pcl_file
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    return mesh

def load_ply(path_2_mesh):
    pcd = o3d.io.read_point_cloud(path_2_mesh)
    return pcd

def load_mesh_or_pc(pointcloud_file, datatype):
    
    if pointcloud_file.split('.')[-1] == 'ply':
        if datatype == "mesh":
            data = load_mesh(pointcloud_file)
        elif datatype == "point cloud":
            data = load_ply(pointcloud_file)
            
        if datatype is None:
            print("DATA TYPE IS NOT SUPPORTED!")
            exit()
    return data

def prepare_data(pointcloud_file, datatype, device):
    # normalization for point cloud features
    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)
    
    if pointcloud_file.split('.')[-1] == 'ply':
        if datatype == "mesh":
            mesh = load_mesh(pointcloud_file)
            points = np.asarray(mesh.vertices)
            colors = np.asarray(mesh.vertex_colors)
            colors = colors * 255.
        elif datatype == "point cloud":
            pcd = load_ply(pointcloud_file)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
        if datatype is None:
            print("DATA TYPE IS NOT SUPPORTED!")
            exit()
        segments = None
    elif pointcloud_file.split('.')[-1] == 'npy':
        points = np.load(pointcloud_file)
        points, colors, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )
        datatype = "mesh"
        
    else:
        print("FORMAT NOT SUPPORTED")
        exit()
    if datatype == "mesh":
        pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
        colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

    coords = np.floor(points / 0.02)
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coords,
        features=colors,
        return_index=True,
        return_inverse=True,
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]

    if segments is not None:
        point2segment_full = segments
        point2segment = segments[unique_map]
        point2segment = [torch.from_numpy(point2segment).long()]
        point2segment_full = [torch.from_numpy(point2segment_full).long()]

        # Concatenate all lists
        input_dict = {"coords": coordinates, "feats": features}
        if len(point2segment) > 0:
            input_dict["labels"] = point2segment
            coordinates, _, point2segment = ME.utils.sparse_collate(**input_dict)
            point2segment = point2segment.cuda()
        else:
            coordinates, _ = ME.utils.sparse_collate(**input_dict)
            point2segment = None
            point2segment_full = None
    else: 
        point2segment = None
        point2segment_full = None
        coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )
    return data, points, colors, features, unique_map, inverse_map, point2segment, point2segment_full


def map_output_to_pointcloud(outputs, 
                             inverse_map,
                             point2segment, 
                             point2segment_full):
    
    # parse predictions
    logits = outputs["pred_logits"]
    logits = torch.functional.F.softmax(logits, dim=-1)[..., :-1]
    masks = outputs["pred_masks"]
    # reformat predictions
    logits = logits[0]
    masks = masks[0] if point2segment is None else masks[0][point2segment]

    num_queries = len(logits)
    scores_per_query, topk_indices = logits.flatten(0, 1).topk(
        num_queries, sorted=True
    )

    topk_indices = topk_indices // 200
    masks = masks[:, topk_indices]

    result_pred_mask = (masks > 0).float()
    heatmap = masks.float().sigmoid()

    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
        result_pred_mask.sum(0) + 1e-6
    )
    score = scores_per_query * mask_scores_per_image
    result_pred_mask = get_full_res_mask(result_pred_mask, inverse_map, point2segment_full[0]) if point2segment_full is not None else result_pred_mask[inverse_map]
    return (result_pred_mask, score)

def get_full_res_mask(mask, inverse_map, point2segment_full):
    mask = mask.detach().cpu()[inverse_map]  # full res
    mask = scatter_mean(mask, point2segment_full, dim=0)  # full res segments
    mask = (mask > 0.5).float()
    mask = mask.detach().cpu()[point2segment_full.cpu()]  # full res points
    return mask

def save_colorized_mesh(mesh, labels_mapped, output_file, colormap='scannet'):
    
    # colorize mesh
    colors = np.zeros((len(mesh.vertices), 3))
    for li in np.unique(labels_mapped):
        if colormap == 'scannet':
            raise ValueError('Not implemented yet')
        elif colormap == 'scannet200':
            v_li = VALID_CLASS_IDS_200[int(li)]
            colors[(labels_mapped == li)[:, 0], :] = SCANNET_COLOR_MAP_200[v_li]
        else:
            raise ValueError('Unknown colormap - not supported')
    
    colors = colors / 255.
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(output_file, mesh)

if __name__ == '__main__':
    
    model = get_model('checkpoints/scannet200/scannet200_benchmark.ckpt')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load input data
    pointcloud_file = 'data/pcl.ply'
    mesh = load_mesh(pointcloud_file)
    
    # prepare data
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)
    
    # run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
        
    # map output to point cloud
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map)
    
    # save colorized mesh
    save_colorized_mesh(mesh, labels, 'data/pcl_labelled.ply', colormap='scannet200')
    