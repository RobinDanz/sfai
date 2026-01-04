from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soilfauna.config import SegmentationConfig

from soilfauna.pipeline import Pipeline
from soilfauna.operators import (
    HSVBackgroundRemoval, 
    BinaryTransform, 
    WatershedSegmentation, 
    ContourDetection, 
    CentersDetection, 
    SAMSegmentation
)

from soilfauna.data import ImageTiler, generate_datasets
from soilfauna.runners import DatasetRunner, ImagePipelineRunner
from soilfauna.export import OutputHandler


def segment(config: SegmentationConfig, dry=False):
    datasets = generate_datasets(config.datasets)
    
    if dry:
        print('Dry run. Dataset informations')
        
        return
    
    tiler = ImageTiler()
    
    pipeline = Pipeline(
        operators=[
            HSVBackgroundRemoval(),
            BinaryTransform(),
            WatershedSegmentation(),
            ContourDetection(),
            CentersDetection(),
            SAMSegmentation(config.model),
        ]
    )
    
    image_runner = ImagePipelineRunner(
        tiler=tiler,
        pipeline=pipeline,
        config=config
    )
    
    for dataset in datasets:
        if dataset.length > 0:
            out = OutputHandler(
                base_dir=config.base_output_dir,
                subname=dataset.root.stem
            )
            out.generate_output_folders()
            
            dataset_runner = DatasetRunner(
                dataset=dataset,
                image_runner=image_runner,
                output=out
            )
    
            dataset_runner.run()