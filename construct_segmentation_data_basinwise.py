"""
Dataset construction script from images in directories
"""

import os
import shutil
from glob import glob

source_dir = 'segment_data'
target_dir = 'basinwise_segment_data'

os.makedirs(target_dir, exist_ok=True)
for split in ['Train', 'Val', 'Test']:
    os.makedirs(os.path.join(target_dir, split, 'Images'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, split, 'Masks'), exist_ok=True)

# Basinwise Splits
basin_splits = {
    'Train': ['SA_95533_64512', 'SA_76565_64512', 'NA_45958_64512', 'Af_70423_64512', 'Af_155997_64512', 'Af_124775_64512'],
    'Val': ['NA_427766_64512', 'SA_84479_64512', 'Af_126767_64512'],
    'Test': ['NA_239852_64512']
}

for split, basins in basin_splits.items():
    for basin in basins:
        image_source_path = os.path.join(source_dir, basin, 'Images')
        mask_source_path = os.path.join(source_dir, basin, 'Masks')

        image_target_path = os.path.join(target_dir, split, 'Images')
        mask_target_path = os.path.join(target_dir, split, 'Masks')

        for file in glob(os.path.join(image_source_path, '*.png')):
            shutil.copy(file, image_target_path)
        for file in glob(os.path.join(mask_source_path, '*.png')):
            shutil.copy(file, mask_target_path)