from utils import OpenYolo3D
import os

openyolo3d = OpenYolo3D(f"{os.getcwd()}/pretrained/config.yaml")
prediction = openyolo3d.predict(f"{os.getcwd()}/sample/scene_0011_00", 1000.0)
openyolo3d.save_output_as_ply(f"{os.getcwd()}/sample/output.ply", True)