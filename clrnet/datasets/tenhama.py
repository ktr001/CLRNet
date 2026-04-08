"""Custom dataset for Tenryu-Hamanako (天浜線) front-facing images.

Inference-only: no annotation loading required.
"""

import os
import os.path as osp
import numpy as np
import cv2
import json
from .base_dataset import BaseDataset
from .registry import DATASETS
from clrnet.utils.visualization import imshow_lanes
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module
class Tenhama(BaseDataset):
    """Loads all .jpg images from data_root for inference.

    No lane annotations are required.  The `evaluate()` method writes
    detected lane coordinates to a JSON file instead of computing metrics.
    """

    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.load_annotations()

    def load_annotations(self):
        self.logger.info('Loading Tenhama images...')
        self.data_infos = []
        img_dir = self.data_root
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith('.jpg'):
                continue
            img_path = osp.join(img_dir, fname)
            self.data_infos.append({
                'img_name': fname,
                'img_path': img_path,
                'lanes': [],
            })
        self.max_lanes = 4
        self.logger.info(f'Found {len(self.data_infos)} images.')

    def evaluate(self, predictions, output_basedir):
        """Save detected lanes to JSON. No metric computation."""
        os.makedirs(output_basedir, exist_ok=True)
        results = []
        for idx, pred in enumerate(predictions):
            img_name = self.data_infos[idx]['img_name']
            lanes_px = [lane.to_array(self.cfg).tolist() for lane in pred]
            results.append({'img_name': img_name, 'lanes': lanes_px})

        out_path = osp.join(output_basedir, 'tenhama_predictions.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f'Predictions saved to {out_path}')
        return None
