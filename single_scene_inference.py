from utils import OpenYolo3D
import os
import pyviz3d.visualizer as viz

openyolo3d = OpenYolo3D(f"{os.getcwd()}/pretrained/config.yaml")
prediction = openyolo3d.predict(path_2_scene_data=f"{os.getcwd()}/data/replica/office0", depth_scale=6553.5, text = ["chair"]) 
openyolo3d.save_output_as_ply(f"{os.getcwd()}/output.ply", True) 

v = viz.Visualizer(position=[5, 5, 1])
v.add_mesh('Room', path=f"{os.getcwd()}/output.ply")
blender_args = {'output_prefix': './',
                  'executable_path': '/Applications/Blender.app/Contents/MacOS/Blender'}
v.save('example_meshes', blender_args=blender_args)