from utils import OpenYolo3D

openyolo3d = OpenYolo3D("$(pwd)/pretrained/config.yaml")
prediction = openyolo3d.predict("$(pwd)/sample/scene_0011_00", 1000.0)
openyolo3d.save_output_as_ply("$(pwd)/sample/output.ply", True)