from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soilfauna.config import SegmentationConfig

from soilfauna.operators import (
    HSVBackgroundRemoval,
    BinaryTransform,
    WatershedSegmentation,
    ContourDetection, 
    CentersDetection,
    SAMSegmentation
)

from soilfauna.data import generate_datasets
from soilfauna.runners import DatasetRunner
from soilfauna.export import OutputHandler

from soilfauna.logging import LOGGER

def segment(config: SegmentationConfig, dry=False):
    datasets = generate_datasets(config.datasets)
    
    LOGGER.info('START SEGMENTATION')
    
    operators = [
            HSVBackgroundRemoval(save=config.save_intermediate_images),
            BinaryTransform(save=config.save_intermediate_images),
            WatershedSegmentation(save=config.save_intermediate_images),
            CentersDetection(save=config.save_intermediate_images),
            SAMSegmentation(config.model, save=config.save_intermediate_images),
        ]
    
    for i, dataset in enumerate(datasets, 1):
        LOGGER.info(f"Dataset: {i}/{len(datasets)}")
        if dataset.length > 0:
            out = OutputHandler(
                base_dir=config.base_output_dir,
                subname=dataset.root.stem
            )
            
            out.generate_output_folders()
            
            dataset_runner = DatasetRunner(
                dataset=dataset,
                operators=operators,
                output_handler=out,
                config=config
            )
    
            dataset_runner.run()