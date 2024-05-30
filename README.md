 <div align="center">
 
## Open-YOLO 3D: Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation

</div>

<div align="center">
<a href="">Mohamed El Amine Boudjoghra</a><sup>1</sup>, <a href="">Angela Dai</a><sup>2</sup>, <a href=""> Jean Lahoud</a><sup>1</sup>, <a href="">Hisham Cholakkal</a><sup>1</sup>, <a href="">Rao Muhammad Anwer</a><sup>1,3</sup>,  <a href="">Salman Khan</a><sup>1,4</sup>, <a href="">Fahad Khan</a><sup>1,5</sup>

<sup>1</sup>Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI) <sup>2</sup>Technical University of Munich (TUM) <sup>3</sup>Aalto University <sup>4</sup>Australian National University <sup>5</sup>Linköping University
</div>


<div align="center">
 
<a href='' target="_blank">![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)</a> 


 </div>



### News

* **30 May 2024**: [Open-YOLO 3D]() released on arXiv. 📝
* **30 May 2024**: Code released. 💻

### Abstract

 Recent works on open-vocabulary 3D instance segmentation show strong promise, but at the cost of slow inference speed and high computation requirements. This high computation cost is typically due to their heavy reliance on 3D clip features, which require computationally expensive 2D foundation models like Segment Anything (SAM) and CLIP for multi-view aggregation into 3D. As a consequence, this hampers their applicability in many real-world applications that require both fast and accurate predictions. To this end, we propose a fast yet accurate open-vocabulary 3D instance segmentation approach, named Open-YOLO 3D, that effectively leverages only 2D object detection from multi-view RGB images for open-vocabulary 3D instance segmentation. 
 We address this task by generating class-agnostic 3D masks for objects in the scene and associating them with text prompts.
 We observe that the projection of class-agnostic 3D point cloud instances already holds instance information; thus, using SAM might only result in redundancy that unnecessarily increases the inference time.
We empirically find that a better performance of matching text prompts to 3D masks can be achieved in a faster fashion with a 2D object detector.  We validate our Open-YOLO 3D on two benchmarks, ScanNet200 and Replica, 
 under two scenarios: (i) with ground truth masks, where labels are required for given object proposals, and (ii) with class-agnostic 3D proposals generated from a 3D proposal network. Our Open-YOLO 3D achieves state-of-the-art performance on both datasets while obtaining up to 16x speedup compared to the best existing method in literature. On ScanNet200 val. set, our Open-YOLO 3D achieves mean average precision (mAP) of 24.1% while operating at 22 seconds per scene.

### Qualitative results
<br>

<div align="center">
    <img src="./docs/qualitatives.png" width="100%">
</div>

## Installation guide

Kindly check [Installation guide](./docs/Installation.md) on how to setup the Conda environment.

## Data Preparation

Kindly check [Data Preparation guide](./docs/Data_prep.md) on how to prepare ScanNet200 and Replica datasets.

</div>

## BibTeX :pray:
```

```

