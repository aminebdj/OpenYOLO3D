from utils.utils_3d import Network_3D
from utils.utils_2d import Network_2D, load_yaml
import time
import torch
import os
import os.path as osp
import imageio
import glob
import open3d as o3d
import numpy as np
import math
from models.Mask3D.mask3d import load_mesh_or_pc
import colorsys
from tqdm import tqdm

def get_iou(masks):
    masks = masks.float()
    intersection = torch.einsum('ij,kj -> ik', masks, masks)
    num_masks = masks.shape[0]
    masks_batch_size = 2 # scannet 200: 20
    if masks_batch_size < num_masks:
        ratio = num_masks//masks_batch_size
        remaining = num_masks-ratio*masks_batch_size
        start_masks = list(range(0,ratio*masks_batch_size, masks_batch_size))
        if remaining == 0:
            end_masks = list(range(masks_batch_size,(ratio+1)*masks_batch_size,masks_batch_size))
        else:
            end_masks = list(range(masks_batch_size,(ratio+1)*masks_batch_size,masks_batch_size))
            end_masks[-1] = num_masks
    else:
        start_masks = [0]
        end_masks = [num_masks]
    union = torch.cat([((masks[st:ed, None, :]+masks[None, :, :]) >= 1).sum(-1) for st,ed in zip(start_masks, end_masks)])
    iou = torch.div(intersection,union)
    
    return iou

def apply_nms(masks, scores, nms_th):
    masks = masks.permute(1,0)
    scored_sorted, sorted_scores_indices = torch.sort(scores, descending=True)
    inv_sorted_scores_indices = {sorted_id.item(): id for id, sorted_id in enumerate(sorted_scores_indices)}
    maskes_sorted = masks[sorted_scores_indices]
    iou = get_iou(maskes_sorted)
    available_indices = torch.arange(len(scored_sorted))
    for indx in range(len(available_indices)):
        remove_indices = torch.where(iou[indx,indx+1:] > nms_th)[0]
        available_indices[indx+1:][remove_indices] = 0
    remaining = available_indices.unique()
    keep_indices = torch.tensor([inv_sorted_scores_indices[id.item()] for id in remaining])
    return keep_indices

def generate_vibrant_colors(num_colors):
    colors = []
    hue_increment = 1.0 / num_colors
    saturation = 1.0
    value = 1.0
    
    for i in range(num_colors):
        hue = i * hue_increment
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    
    return colors

def get_visibility_mat(pred_masks_3d, inside_mask, topk = 15):
    intersection = torch.einsum("ik, fk -> if", pred_masks_3d.float(), inside_mask.float())
    total_point_number = pred_masks_3d[:, None, :].float().sum(dim = -1)
    visibility_matrix = intersection/total_point_number
    
    if topk > visibility_matrix.shape[-1]:
        topk = visibility_matrix.shape[-1]
    
    max_visiblity_in_frame = torch.topk(visibility_matrix, topk, dim = -1).indices
    
    visibility_matrix_bool = torch.zeros_like(visibility_matrix).bool()
    visibility_matrix_bool[torch.tensor(range(len(visibility_matrix_bool)))[:, None],max_visiblity_in_frame] = True
    
    return visibility_matrix_bool

def compute_iou(box, boxes):
    assert box.shape == (4,), "Reference box must be of shape (4,)"
    assert boxes.shape[1] == 4, "Boxes must be of shape (N, 4)"
    
    x1_inter = torch.max(box[0], boxes[:, 0])
    y1_inter = torch.max(box[1], boxes[:, 1])
    x2_inter = torch.min(box[2], boxes[:, 2])
    y2_inter = torch.min(box[3], boxes[:, 3])
    inter_area = (x2_inter - x1_inter).clamp(0) * (y2_inter - y1_inter).clamp(0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    iou = inter_area / union_area
    
    return iou

class OpenYolo3D():
    def __init__(self, openyolo3d_config = ""):
        config = load_yaml(openyolo3d_config)
        self.network_3d = Network_3D(config)
        self.network_2d = Network_2D(config)
        self.openyolo3d_config = config
    
    def predict(self, path_2_scene_data, depth_scale, text = None, datatype="point cloud", processed_scene = None, path_to_3d_masks = None, is_gt=False):
        self.num_classes = len(text)+1 if text is not None else len(self.openyolo3d_config["network2d"]["text_prompts"])+1
        self.datatype = datatype
        self.world2cam = WORLD_2_CAM(path_2_scene_data, depth_scale, self.openyolo3d_config)
        self.mesh_projections = self.world2cam.get_mesh_projections()
        self.scaling_params = [self.world2cam.depth_resolution[0]/self.world2cam.image_resolution[0], self.world2cam.depth_resolution[1]/self.world2cam.image_resolution[1]]
        
        scene_name = path_2_scene_data.split("/")[-1]
        print("[🚀 ACTION] 3D mask proposals computation ...")
        start = time.time()
        
        if path_to_3d_masks is None:
            self.preds_3d = self.network_3d.get_class_agnostic_masks(self.world2cam.mesh, datatype) if processed_scene is None else self.network_3d.get_class_agnostic_masks(processed_scene, datatype)
            keep_score = self.preds_3d[1] >= self.openyolo3d_config["network3d"]["th"]
            keep_nms = apply_nms(self.preds_3d[0][:, keep_score].cuda(), self.preds_3d[1][keep_score].cuda(), self.openyolo3d_config["network3d"]["nms"])
            self.preds_3d = (self.preds_3d[0].cpu().permute(1,0)[keep_score][keep_nms].permute(1,0), self.preds_3d[1].cpu()[keep_score][keep_nms])
        else:
            self.preds_3d = torch.load(osp.join(path_to_3d_masks, f"{scene_name}.pt"))
            
        print(f"[🕒 INFO] Elapsed time {(time.time()-start)}")
        print(f"[✅ INFO] Proposals computed.")   

        print("[🚀 ACTION] 2D Bounding Boxes computation ...")
        start = time.time()
        self.preds_2d = self.network_2d.get_bounding_boxes(self.world2cam.color_paths, text)
        # self.preds_2d = torch.load(osp.join(f"/share/data/drive_3/OpenYolo3D/bboxes_2d", f"{scene_name}.pt"))
        print(f"[🕒 INFO] Elapsed time {(time.time()-start)}")
        print(f"[✅ INFO] Bounding boxes computed.")  
        
        print("[🚀 ACTION] Predicting ...")
        start = time.time()
        prediction = self.label_3d_masks_from_2d_bboxes(scene_name, is_gt)
        print(f"[🕒 INFO] Elapsed time {(time.time()-start)}")
        print(f"[✅ INFO] Prediction completed")    
            
        return prediction
    
    def label_3d_masks_from_2d_bboxes(self, scene_name, is_gt=False):
        projections_mesh_to_frame , keep_visible_points = self.mesh_projections
        predictions_2d_bboxes = self.preds_2d
        prediction_3d_masks, _ = self.preds_3d
        
        predicted_masks, predicated_classes, predicated_scores = self.label_3d_masks_from_label_maps(prediction_3d_masks.bool(), 
                                                                                                        predictions_2d_bboxes, 
                                                                                                        projections_mesh_to_frame,
                                                                                                        keep_visible_points, 
                                                                                                        is_gt)
        
        self.predicted_masks = predicted_masks
        self.predicated_scores = predicated_scores
        self.predicated_classes = predicated_classes
        
        return {scene_name : (predicted_masks, predicated_classes, predicated_scores)}
    
    
    def label_3d_masks_from_label_maps(self, 
                                        prediction_3d_masks, 
                                        predictions_2d_bboxes, 
                                        projections_mesh_to_frame, 
                                        keep_visible_points,
                                        is_gt):
        
        label_maps = self.construct_label_maps(predictions_2d_bboxes) #construct the label maps , start from the biggest bbox to small one

        visibility_matrix = get_visibility_mat(prediction_3d_masks.cuda().permute(1,0), keep_visible_points.cuda(), topk = 25 if is_gt else self.openyolo3d_config["openyolo3d"]["topk"])
        valid_frames = visibility_matrix.sum(dim=0) >= 1
        
        prediction_3d_masks = prediction_3d_masks.permute(1,0).cpu()
        prediction_3d_masks_np = prediction_3d_masks.numpy()
        projections_mesh_to_frame = projections_mesh_to_frame[valid_frames].cpu().numpy()
        visibility_matrix = visibility_matrix[:, valid_frames].cpu().numpy()
        keep_visible_points = keep_visible_points[valid_frames].cpu().numpy()
        distributions = []
        
        class_labels = []
        class_probs = []
        class_dists = []
        label_maps = label_maps[valid_frames].numpy()
        bounding_boxes = predictions_2d_bboxes.values()
        bounding_boxes_valid = [bbox for (bi, bbox) in enumerate(bounding_boxes) if valid_frames[bi]]
        for mask_id, mask in enumerate(prediction_3d_masks_np):
            prob_normalizer = 0

            representitive_frame_ids = np.where(visibility_matrix[mask_id])[0]
            labels_distribution = []
            iou_vals = []
            for representitive_frame_id in representitive_frame_ids:
                visible_points_mask = (keep_visible_points[representitive_frame_id].squeeze()*mask).astype(bool)
                prob_normalizer +=  visible_points_mask.sum()
                instance_x_y_coords = projections_mesh_to_frame[representitive_frame_id][np.where(visible_points_mask)].astype(np.int64)
                
                boxes = bounding_boxes_valid[representitive_frame_id]["bbox"].long()
                if len(boxes) > 0 and len(instance_x_y_coords) > 10:
                    x_l, x_r, y_t, y_b = instance_x_y_coords[:, 0].min(), instance_x_y_coords[:, 0].max()+1, instance_x_y_coords[:, 1].min(), instance_x_y_coords[:, 1].max()+1
                    box = torch.tensor([x_l/self.scaling_params[1], y_t/self.scaling_params[0], x_r/self.scaling_params[1], y_b/self.scaling_params[0]])
                
                    iou_values = compute_iou(box, boxes)
                    iou_vals.append(iou_values.max().item())
                selected_labels = label_maps[representitive_frame_id, instance_x_y_coords[:, 1], instance_x_y_coords[:, 0]]
                labels_distribution.append(selected_labels)
            
            labels_distribution = np.concatenate(labels_distribution) if len(labels_distribution) > 0 else np.array([-1])
            
            # class_dists.append(labels_distribution)
            distribution = torch.zeros(self.num_classes) if self.openyolo3d_config["openyolo3d"]["topk_per_image"] != -1 else None
            if (labels_distribution != -1).sum() != 0:
                
                if distribution is not None:
                    all_labels = torch.from_numpy(labels_distribution[labels_distribution != -1])
                    all_labels_unique = all_labels.unique()
                    for lb in all_labels_unique:
                        distribution[lb] = (all_labels == lb).sum()
                        
                    distribution = distribution/distribution.max()
                
                class_label = torch.mode(torch.from_numpy(labels_distribution[labels_distribution != -1])).values.item()
                class_prob = (labels_distribution == class_label).sum()/prob_normalizer
            else:
                if distribution is not None:
                    distribution[-1] = 1.0
                class_label = self.num_classes-1
                class_prob = 0.0

            iou_vals = torch.tensor(iou_vals)
            
            class_labels.append(class_label)
            if (iou_vals != 0).sum():
                iou_prob = iou_vals[iou_vals != 0].mean().item()
            else:
                iou_prob = 0.0
            
            class_probs.append(class_prob*iou_prob)
            if distribution is not None:
                distributions.append(distribution)
                
        pred_classes = torch.tensor(class_labels)
        pred_scores = torch.tensor(class_probs)
        if distribution is not None:
            distributions = torch.stack(distributions) if len(distributions) > 0 else torch.tensor((0, self.num_classes))
        
        if (self.openyolo3d_config["openyolo3d"]["topk_per_image"] != -1) and (not is_gt):
            # print("TOPK USED")
            n_instance = distributions.shape[0]
            distributions = distributions.reshape(-1)
            labels = (
            torch.arange(self.num_classes, device=distributions.device)
            .unsqueeze(0)
            .repeat(n_instance, 1)
            .flatten(0, 1)
            )

            cur_topk = self.openyolo3d_config["openyolo3d"]["topk_per_image"]
            _, idx = torch.topk(distributions, k=min(cur_topk, len(distributions)), largest=True)
            mask_idx = torch.div(idx, self.num_classes, rounding_mode="floor")

            pred_classes = labels[idx]
            pred_scores = distributions[idx].cuda()
            prediction_3d_masks = prediction_3d_masks[mask_idx]
        
        return prediction_3d_masks.permute(1,0), pred_classes, pred_scores
    
    def construct_label_maps(self, predictions_2d_bboxes, save_label_map=False):
        label_maps = (torch.ones((len(predictions_2d_bboxes), self.world2cam.height, self.world2cam.width))*-1).type(torch.int16)
        for frame_id, pred in enumerate(predictions_2d_bboxes.values()):
            bboxes = pred["bbox"].long()
            labels = pred["labels"].type(torch.int16)
        
            bboxes[:,0] = bboxes[:,0]*self.scaling_params[1]
            bboxes[:,2] = bboxes[:,2]*self.scaling_params[1]
            bboxes[:,1] = bboxes[:,1]*self.scaling_params[0]
            bboxes[:,3] = bboxes[:,3]*self.scaling_params[0]
            bboxes_weights = (bboxes[:,2]-bboxes[:,0])+(bboxes[:,3]-bboxes[:,1])
            sorted_indices = bboxes_weights.sort(descending=True).indices
            bboxes = bboxes[sorted_indices]
            labels = labels[sorted_indices]
            for id, bbox in enumerate(bboxes):
                label_maps[frame_id, bbox[1]:bbox[3],bbox[0]:bbox[2]] = labels[id]
                
        return label_maps
    
    def save_output_as_ply(self, save_path, th = 0.1):
        num_classes = len(self.predicated_classes.unique())
        data = load_mesh_or_pc(self.world2cam.mesh, self.datatype)
        if self.datatype == 'mesh':
            mesh = data
            vertex_colors = np.asarray(mesh.vertex_colors)
            vibrant_colors = generate_vibrant_colors(num_classes)

            for i, class_id in enumerate((self.predicated_classes).unique()):
                if class_id == self.num_classes-1:
                    continue
                class_id_mask = self.predicated_classes == class_id
                scores_per_class = self.predicated_scores[class_id_mask]
                class_id_max = torch.argmax(scores_per_class)
                mask = self.predicted_masks.permute(1,0)[class_id_mask][class_id_max]
                vertex_colors[mask] = np.array(vibrant_colors.pop())
                
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            o3d.io.write_triangle_mesh(save_path, mesh)
        elif self.datatype == 'point cloud':
            point_cloud = data
            point_colors = np.asarray(point_cloud.colors)
            vibrant_colors = generate_vibrant_colors(num_classes)
            for i, class_id in enumerate(self.predicated_classes.unique()):
                if class_id == self.num_classes-1:
                    continue
                class_id_mask = self.predicated_classes == class_id
                scores_per_class = self.predicated_scores[class_id_mask]
                class_id_max = torch.argmax(scores_per_class)
                if scores_per_class[class_id_max] < th:
                    continue
                mask = self.predicted_masks.permute(1,0)[class_id_mask][class_id_max]
                point_colors[mask] = np.array(vibrant_colors.pop())

            point_cloud.colors = o3d.utility.Vector3dVector(point_colors)
            o3d.io.write_point_cloud(save_path, point_cloud)
        else:
            print("[ERROR] output is not saved, please check input 3D scene folder.")
            exit()
    


class WORLD_2_CAM():
    def __init__(self, path_2_scene, depth_scale, openyolo3d_config = None):
        self.poses = {}
        self.intrinsics = {}
        self.meshes = {}
        self.depth_maps_paths = {}
        self.depth_color_paths = {}
        self.vis_depth_threshold =  openyolo3d_config["openyolo3d"]['vis_depth_threshold']
        
        frequency = openyolo3d_config["openyolo3d"]['frequency']
        
        path_2_poses = osp.join(path_2_scene,"poses")
        num_frames = len(os.listdir(path_2_poses))
        self.poses = [osp.join(path_2_poses, f"{i}.txt") for i in list(range(num_frames))[::frequency]]
        
        path_2_intrinsics = osp.join(path_2_scene,"intrinsics.txt")
        self.intrinsics = [path_2_intrinsics for i in list(range(num_frames))[::frequency]] 
        
        self.mesh = glob.glob(path_2_scene+"/*.ply")[0]
        
        path_2_depth = osp.join(path_2_scene,"depth")
        self.depth_maps_paths = [osp.join(path_2_depth, f"{i}.png") for i in list(range(num_frames))[::frequency]]
        
        path_2_color = osp.join(path_2_scene,"color")
        self.color_paths = [osp.join(path_2_color, f"{i}.jpg") for i in list(range(num_frames))[::frequency]]
        
            
        self.image_resolution = imageio.imread(list(self.color_paths)[0]).shape[:2]
        self.depth_resolution = imageio.imread(list(self.depth_maps_paths)[0]).shape
        self.height = self.depth_resolution[0]
        self.width = self.depth_resolution[1]
        
        self.depth_scale = depth_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @staticmethod
    def load_ply(path_2_mesh):
        pcd = o3d.io.read_point_cloud(path_2_mesh)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        # print(points.shape)
        coords = np.concatenate([points, np.ones((points.shape[0], 1))], axis = -1)
        return coords, colors
    
    def load_depth_maps(self):
        depth_maps = []
        paths_to_depth_maps_scene_i = self.depth_maps_paths
        for depth_map_path_i in paths_to_depth_maps_scene_i:
            depth_path = os.path.join(depth_map_path_i)
            depth_maps.append(torch.from_numpy(imageio.imread(depth_path) / self.depth_scale).to(self.device))
        return torch.stack(depth_maps)
    
    def adjust_intrinsic(self, intrinsic, original_resolution, new_resolution):
        if original_resolution == new_resolution:
            return intrinsic
        
        resize_width = int(math.floor(new_resolution[1] * float(
                        original_resolution[0]) / float(original_resolution[1])))
        
        adapted_intrinsic = intrinsic.copy()
        adapted_intrinsic[0, 0] *= float(resize_width) / float(original_resolution[0])
        adapted_intrinsic[1, 1] *= float(new_resolution[1]) / float(original_resolution[1])
        adapted_intrinsic[0, 2] *= float(new_resolution[0] - 1) / float(original_resolution[0] - 1)
        adapted_intrinsic[1, 2] *= float(new_resolution[1] - 1) / float(original_resolution[1] - 1)
        return adapted_intrinsic
    
    def get_mesh_projections(self):
        N_Large = 2000000*250
        
        points, colors = self.load_ply(self.mesh)
        points, colors = torch.from_numpy(points).cuda(), torch.from_numpy(colors).cuda()
        
        intrinsic = self.adjust_intrinsic(np.loadtxt(self.intrinsics[0]), self.image_resolution, self.depth_resolution)
        intrinsics = torch.from_numpy(np.stack([intrinsic for frame_id in range(len(self.poses))])).cuda()
        extrinsics = torch.linalg.inv(torch.from_numpy(np.stack([np.loadtxt(pose) for pose in self.poses])).cuda())
        
        if extrinsics.shape[0]*points.shape[0] < N_Large:
            word2cam_mat = torch.einsum('bij, jk -> bik',torch.einsum('bij,bjk -> bik', intrinsics,extrinsics), points.T).permute(0,2,1)
        else:
            B_size = 800000
            Num_Points = points.shape[0]
            Num_batches = Num_Points//B_size+1
            word2cam_mat = []
            for b_i in range(Num_batches):
                dim_start = b_i*B_size
                dim_last = (b_i+1)*B_size if b_i != Num_batches-1 else points.shape[0]
                word2cam_mat_i = torch.einsum('bij, jk -> bik',torch.einsum('bij,bjk -> bik', intrinsics,extrinsics), points[dim_start:dim_last].T).permute(0,2,1)
                word2cam_mat.append(word2cam_mat_i.cpu())
            word2cam_mat = torch.cat(word2cam_mat, dim = 1)
        del intrinsics
        del extrinsics
        del points
        del colors
        torch.cuda.empty_cache()
        
        point_depth = word2cam_mat[:, :, 2].cuda()
        if word2cam_mat.shape[1]*word2cam_mat.shape[0] < N_Large:
            size = (word2cam_mat.shape[0], word2cam_mat.shape[1])
            mask = (word2cam_mat[:, :, 2] != 0).reshape(size[0]*size[1])
            
            projected_points = torch.stack([(word2cam_mat[:, :, 0].reshape(size[0]*size[1])[mask]/word2cam_mat[:, :, 2].reshape(size[0]*size[1])[mask]).reshape(size), 
                        (word2cam_mat[:, :, 1].reshape(size[0]*size[1])[mask]/word2cam_mat[:, :, 2].reshape(size[0]*size[1])[mask]).reshape(size)]).permute(1,2,0).long()
            inside_mask = ((projected_points[:,:,0] < self.width)*(projected_points[:,:,0] > 0)*(projected_points[:,:,1] < self.height)*(projected_points[:,:,1] >0) == 1 )
        
        else:
            B_size = 200000
            Num_Points = word2cam_mat.shape[1]
            Num_batches = Num_Points//B_size+1
            projected_points = []

            for b_i in range(Num_batches):
                dim_start = b_i*B_size
                dim_last = (b_i+1)*B_size if b_i != Num_batches-1 else word2cam_mat.shape[1]
                batch_z = word2cam_mat[:, dim_start:dim_last, 2].cuda()
                batch_y = word2cam_mat[:, dim_start:dim_last, 1].cuda()
                batch_x = word2cam_mat[:, dim_start:dim_last, 0].cuda()
                
                size = (word2cam_mat.shape[0], dim_last-dim_start)
                mask = (batch_z != 0).reshape(size[0]*size[1])
                projected_points_i = torch.stack([(torch.div(batch_x.reshape(size[0]*size[1])[mask],batch_z.reshape(size[0]*size[1])[mask])).reshape(size), 
                            (torch.div(batch_y.reshape(size[0]*size[1])[mask],batch_z.reshape(size[0]*size[1])[mask])).reshape(size)]).permute(1,2,0).long()
                projected_points.append(projected_points_i.cpu())
                

           
            # merge parts
            projected_points = torch.cat(projected_points, dim = 1)
            inside_mask = ((projected_points[:,:,0] < self.width)*(projected_points[:,:,0] > 0)*(projected_points[:,:,1] < self.height)*(projected_points[:,:,1] >0) == 1 )
            
        
        # Get visible points with depth, width, and height
        depth_maps = self.load_depth_maps()
        num_frames = depth_maps.shape[0]
        # pixel_to_3d_point = []
        for frame_id in range(num_frames):
            points_in_frame_mask = inside_mask[frame_id].clone()
            points_in_frame = (projected_points[frame_id][points_in_frame_mask])
            depth_in_frame = point_depth[frame_id][points_in_frame_mask]
            visibility_mask = (torch.abs(depth_maps[frame_id][points_in_frame[:,1].long(), points_in_frame[:,0].long()]
                                        - depth_in_frame) <= \
                                        self.vis_depth_threshold)
            
            inside_mask[frame_id][points_in_frame_mask] = visibility_mask.to(inside_mask.device)
        
        return projected_points.type(torch.int16).cpu(), inside_mask.cpu()