from utils import OpenYolo3D
import os
import pyviz3d.visualizer as viz
from models.Mask3D.mask3d import load_mesh_or_pc
import numpy as np

openyolo3d = OpenYolo3D(f"{os.getcwd()}/pretrained/config.yaml")
prediction = openyolo3d.predict(path_2_scene_data=f"{os.getcwd()}/data/replica/office0", depth_scale=6553.5, text = ["chair"]) 
openyolo3d.save_output_as_ply(f"{os.getcwd()}/output.ply") 


pc = load_mesh_or_pc(f"{os.getcwd()}/output.ply", datatype="point cloud")
point_size = 35.0
v = viz.Visualizer(position=[5, 5, 1])
v.add_points('Prediction', np.asarray(pc.points), np.asarray(pc.colors)*255.0, point_size=point_size, visible=True)

blender_args = {'output_prefix': './',
                  'executable_path': '/Applications/Blender.app/Contents/MacOS/Blender'}
v.save('example_meshes', blender_args=blender_args)