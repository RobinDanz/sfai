from sfai.config import SegmentationConfig
from sfai.segmentation import segment


def add_segment_parser(subparsers):
    """_summary_

    Args:
        subparsers (_type_): _description_
    """
    parser = subparsers.add_parser(
        "segment",
        help="Automatic segmentation tool"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Config file"
    )
    
    parser.add_argument(
        "-d", "--dry",
        action='store_true',
        help="Dry Run. Display images informations"
    )
    
    parser.set_defaults(func=run_segmentation)
    
def run_segmentation(args):
    """
    CLI entrypoint for segmentation tool
    """
    cfg = SegmentationConfig.from_file(args.config)
    cfg.create_run_folder()
    
    segment(cfg, dry=args.dry)
