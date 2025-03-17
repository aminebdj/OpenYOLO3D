## ScanNet200

### Download ScanNet200 preprocessed data
The link to the preprocessed scannet200 data is available in the following link <a href="https://docs.google.com/forms/d/e/1FAIpQLSemdl7wwy4DWn8Nh3_pzEXCx-bBKFjzdDL6PC6t6laVSIdCCg/viewform?usp=header">ScanNet200 preprocessed</a>. 

Important: please make sure to fill in the ScanNet terms of use, and send it to the scannet group email before downloading the pre-processed data.

### Or process the raw scannet200 data
Download <a href="https://kaldir.vc.in.tum.de/scannet_benchmark/documentation">ScaNnet200</a> and follow <a href="[https://kaldir.vc.in.tum.de/scannet_benchmark/documentation](https://github.com/JonasSchult/Mask3D)">Mask3D</a> to preprocess data. A preprocessed point cloud scene has the following format `id.npy`. Kindly move all preprocessed scenes to their corresponding folders e.g. `0011_00.npy` to `./data/scannet200/scene0011_00`.

Next, you need to download the RGB-D sequence, which is available in <a href="[https://kaldir.vc.in.tum.de/scannet_benchmark/documentation](https://github.com/pengsongyou/openscene)">OpenScene repository</a>, or from the official Scannet200 website <a href="https://kaldir.vc.in.tum.de/scannet_benchmark/documentation">ScaNnet200</a>. 

ScanNet200 data folder should be structured as follows
```
./data
  └── scannet200
    └── ground_truth
            ├── scene0011_00.txt
            └── ...
    └── scene0011_00
            ├── poses                            <- folder with camera poses
            │      ├── 0.txt 
            │      ├── 1.txt 
            │      └── ...  
            ├── color                           <- folder with RGB images
            │      ├── 0.jpg (or .png/.jpeg)
            │      ├── 1.jpg (or .png/.jpeg)
            │      └── ...  
            ├── depth                           <- folder with depth images
            │      ├── 0.png (or .jpg/.jpeg)
            │      ├── 1.png (or .jpg/.jpeg)
            │      └── ...  
            ├── intrinsics.txt                 <- camera intrinsics
            │ 
            └── 0011_00.npy                <- preprocessed point cloud of the scene
            │ 
            └── scene0011_00_vh_clean_2.ply      <- raw point cloud of the scene
    └── ... 
```

## Replica

Run the following command. 
```
sh scripts/get_replica_dataset.sh
```

A replica folder with the following format will be generated.
```
./data
  └── replica
    └── ground_truth
            ├── office0.txt
            └── ...
    └── office0
            ├── poses                            <- folder with camera poses
            │      ├── 0.txt 
            │      ├── 1.txt 
            │      └── ...  
            ├── color                           <- folder with RGB images
            │      ├── 0.jpg (or .png/.jpeg)
            │      ├── 1.jpg (or .png/.jpeg)
            │      └── ...  
            ├── depth                           <- folder with depth images
            │      ├── 0.png (or .jpg/.jpeg)
            │      ├── 1.png (or .jpg/.jpeg)
            │      └── ...  
            ├── intrinsics.txt                 <- camera intrinsics
            │ 
            └── office0_mesh.ply      <- raw point cloud of the scene
    └── ... 
```

