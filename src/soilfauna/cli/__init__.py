from .segment import add_segment_parser
from .coco2biigle import add_coco2biigle_parser
from .cpfiles import add_cpfiles_parser

__all__ = [
    "add_segment_parser",
    "add_coco2biigle_parser",
    "add_cpfiles_parser"
]
