## Clone the repository
```
git clone https://github.com/aminebdj/OpenYOLO3D.git
cd OpenYOLO3D

```

## Conda Environment
```
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

conda env create -f environment.yml

conda activate openyolo3d

cd models/Mask3D/
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

cd third_party

git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas

cd ..
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make
cd ../../

cd pointnet2
python setup.py install

cd ../../
pip3 install pytorch-lightning==1.7.2

pip install black==21.4b2 cloudpickle==3.0.0 future hydra-core==1.0.5 pycocotools>=2.0.2 pydot iopath==0.1.7 loguru albumentations

pip install .

cd ../YOLO-World
pip install -e .
cd ../../
pip install mmyolo==0.6.0 mmdet==3.0.0 plyfile
pip install openmim
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
mim install mmcv==2.0.0
pip install open3d
pip install pillow==9.1.0

python -m pip install pyviz3d # optional for visualization

```
## Prepare the class agnostic masks
```
sh scripts/get_class_agn_masks.sh
```
## Prepare the checkpoints
```
sh scripts/get_checkpoints.sh
```
