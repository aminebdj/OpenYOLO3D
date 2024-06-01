from utils import OpenYolo3D


openyolo3d = OpenYolo3D("/home/jean/Amine/OpenYolo3D/pretrained/config.yaml")

a = openyolo3d.predict("/home/jean/Amine/OpenYolo3D/sample/scene_0011_00", 1000.0, "/home/jean/Amine/OpenYolo3D/sample/scene_0011_00/0011_00.npy")

# openyolo3d.save_output_as_ply("/home/jean/Amine/OpenYolo3D/sample/output.ply", True)