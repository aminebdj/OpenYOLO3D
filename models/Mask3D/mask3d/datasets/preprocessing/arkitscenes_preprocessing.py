import re
from pathlib import Path
import numpy as np
import pandas as pd
from fire import Fire
from natsort import natsorted
from loguru import logger
import os

from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply_with_normals

from datasets.scannet200.scannet200_constants import (
    VALID_CLASS_IDS_200,
    SCANNET_COLOR_MAP_200,
    CLASS_LABELS_200,
)


class ARKitScenesPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "/home/weders/scratch/scratch/scannetter/arkit/raw",
        save_dir: str = "/home/weders/scratch/scratch/scannetter/arkit/raw",
        modes: tuple = ('Validation', ),
        n_jobs: int = 1,
        git_repo: str = "./data/raw/scannet/ScanNet",
        mesh_file: str="mesh_tsdf.ply",
        scannet200: bool = False,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.scannet200 = scannet200
        git_repo = Path(git_repo)
        for mode in self.modes:
            scenes = os.listdir(os.path.join(data_dir, mode))
            scans_folder = "scans_test" if mode == "test" else "scans"
            filepaths = []
            for scene in scenes:
                if os.path.exists(os.path.join(data_dir, mode, scene, mesh_file)):
                    filepaths.append(
                        self.data_dir
                        / mode
                        / scene
                        / mesh_file)
            self.files[mode] = natsorted(filepaths)

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        scene = int(filepath.parent.name)
        print(scene)
        filebase = {
            "filepath": filepath,
            "scene": scene,
            "sub_scene": scene,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        # reading both files and checking that they are fitting
        coords, features, _ = load_ply_with_normals(filepath)
        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((coords, features))

        print(features.shape)

        points = np.concatenate((points, np.zeros((file_len, 4))), axis=1) # adding segment and label fake columns

        processed_filepath = (
            self.save_dir / mode / f"data_mask3d.npy"
        )
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        return filebase

    @logger.catch
    def fix_bugs_in_labels(self):
        if not self.scannet200:
            logger.add(self.save_dir / "fixed_bugs_in_labels.log")
            found_wrong_labels = {
                tuple([270, 0]): 50,
                tuple([270, 2]): 50,
                tuple([384, 0]): 149,
            }
            for scene, wrong_label in found_wrong_labels.items():
                scene, sub_scene = scene
                bug_file = (
                    self.save_dir / "train" / f"{scene:04}_{sub_scene:02}.npy"
                )
                points = np.load(bug_file)
                bug_mask = points[:, -1] != wrong_label
                points = points[bug_mask]
                np.save(bug_file, points)
                logger.info(f"Fixed {bug_file}")

    def _parse_scene_subscene(self, name):
        scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
        print(scene_match)
        return int(scene_match.group(1)), int(scene_match.group(2))


if __name__ == "__main__":
    Fire(ARKitScenesPreprocessing)