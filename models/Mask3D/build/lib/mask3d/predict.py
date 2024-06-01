import hydra
from omegaconf import DictConfig, OmegaConf
from models.mask3d import Mask3D
import os
import torch

import MinkowskiEngine as ME
import open3d as o3d
import numpy as np
import albumentations as A

from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

from datasets.scannet200.scannet200_constants import (
    SCANNET_COLOR_MAP_200,
    SCANNET_COLOR_MAP_20,
    VALID_CLASS_IDS_200,
    VALID_CLASS_IDS_20,
    CLASS_LABELS_200,
    CLASS_LABELS_20,
)

root_dir = '/home/weders/scratch/scratch/scannetter/arkit/raw/Validation'

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)


    def forward(self, x, raw_coordinates=None):
        return self.model(x, raw_coordinates=raw_coordinates)

@hydra.main(
    config_path="conf", config_name="config_base_instance_segmentation.yaml"
)
def main(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(hydra.utils.get_original_cwd())
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    model = model.to(device)
    # model.eval()

    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    # iterate over data
    for sc in os.listdir(root_dir):


        if not os.path.exists(os.path.join(root_dir, sc, 'mesh_tsdf.ply')):
            continue

        # save outputs
        output_dir = os.path.join(root_dir, sc, 'pred_mask3d_ours')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if sc != '42445991':
            continue
        
        # if os.path.exists(os.path.join(output_dir, 'mask3d_predictions.txt')):
        #     print('Skipping', sc)
        #     continue
        
        print('Processing', sc)

        mesh = o3d.io.read_triangle_mesh(os.path.join(root_dir, sc, 'mesh_tsdf.ply'))
        mesh.compute_vertex_normals()
    
        points = np.asarray(mesh.vertices)
        colors = np.asarray(mesh.vertex_colors)
        

        colors = colors * 255.
        pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
        colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

        # voxelize data
        coords = np.floor(points / 0.02)

         # maybe this change (_, _, ...) is not necessary and we can directly get out
        # the sample coordinates?
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(coordinates=coords, features=colors, return_index=True, return_inverse=True)

        sample_coordinates = coords[unique_map]
        coordinates = [torch.from_numpy(sample_coordinates).int()]
        sample_features = colors[unique_map]
        features = [torch.from_numpy(sample_features).float()]

        coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
        features = torch.cat(features, dim=0)
        data = ME.SparseTensor(
            coordinates=coordinates,
            features=features,
            device=device,
        )

        # run model
        with torch.no_grad():
            outputs = model(data, raw_coordinates=features)

        del data
        torch.cuda.empty_cache()

        # parse predictions
        logits = outputs["pred_logits"]
        masks = outputs["pred_masks"]


        # reformat predictions
        logits = logits[0].detach().cpu()
        masks = masks[0].detach().cpu()

        labels = []
        confidences = []
        masks_binary = []

        for i in range(len(logits)):
            p_labels = torch.softmax(logits[i], dim=-1)
            p_masks = torch.sigmoid(masks[:, i])
            l = torch.argmax(p_labels, dim=-1)
            c_label = torch.max(p_labels)            
            m = p_masks > 0.5
            c_m = p_masks[m].sum() / (m.sum() + 1e-8)
            c = c_label * c_m
            if l < 200 and c > 0.5:
                labels.append(l.item())
                confidences.append(c.item())
                masks_binary.append(m[inverse_map]) # mapping the mask back to the original point cloud

            
        # save labelled mesh
        mesh_labelled = o3d.geometry.TriangleMesh()
        mesh_labelled.vertices = mesh.vertices
        mesh_labelled.triangles = mesh.triangles

        labels_mapped = np.zeros((len(mesh.vertices), 1))
        colors_mapped = np.zeros((len(mesh.vertices), 3))

        confidences, labels, masks_binary = zip(*sorted(zip(confidences, labels, masks_binary), reverse=False))
        for i, (l, c, m) in enumerate(zip(labels, confidences, masks_binary)):
            labels_mapped[m == 1] = l
            if l == 0:
                l_ = -1 + 2 # label offset is 2 for scannet 200, 0 needs to be mapped to -1 before (see trainer.py in Mask3D)
            else:
                l_ = l + 2
            # print(VALID_CLASS_IDS_200[l_], SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l_]], l_, CLASS_LABELS_200[l_])
            colors_mapped[m == 1] = SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l_]]

            # colors_mapped[mask_mapped == 1] = SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[l]]
        

        

        mesh_labelled.vertex_colors = o3d.utility.Vector3dVector(colors_mapped.astype(np.float32) / 255.)
        o3d.io.write_triangle_mesh(f'{output_dir}/mesh_tsdf_labelled.ply', mesh_labelled)

        mask_path = os.path.join(output_dir, 'pred_mask')
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        # sorting by confidence
        with open(os.path.join(output_dir, 'mask3d_predictions.txt'), 'w') as f:
            for i, (l, c, m) in enumerate(zip(labels, confidences, masks_binary)):
                mask_file = f'pred_mask/{str(i).zfill(3)}.txt'
                f.write(f'{mask_file} {VALID_CLASS_IDS_200[l]} {c}\n')
                np.savetxt(os.path.join(output_dir, mask_file), m.numpy(), fmt='%d')


if __name__ == "__main__":
    main()