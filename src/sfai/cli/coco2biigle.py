from pathlib import Path

from sfai.config import default
from sfai.scripts import convert

def add_coco2biigle_parser(subparsers):
    """_summary_

    Args:
        subparsers (_type_): _description_
    """
    parser = subparsers.add_parser(
        "coco2biigle",
        help="Convert a COCO annotation file to a Biigle volume."
    )
    
    parser.add_argument(
        "-c", 
        "--coco",
        help="Path to a coco file.",
        required=True
    )

    parser.add_argument(
        "-t", 
        "--label_tree_path",
        help="Path to a Biigle Label Tree zip file.",
        required=True
    )

    parser.add_argument(
        "-o", 
        "--out_dir",
        default=default.DEFAULT_COCO2BIIGLE_OUTPUT_DIR,
        help=f"Output path. Default: Current directory ({default.DEFAULT_COCO2BIIGLE_OUTPUT_DIR})"
    )

    parser.add_argument(
        "-p", 
        "--project_name",
        default="project01",
        help="Output project name. Default: project01"
    )

    parser.add_argument(
        "-v", 
        "--volume_name",
        default="volume01",
        help="Output volume name. Default: volume01"
    )

    parser.set_defaults(func=run_coco2biigle)
    
def run_coco2biigle(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    coco_file_path = Path(args.coco)
    label_tree_path = Path(args.label_tree_path)
    output_dir = Path(args.out_dir)
    project_name = args.project_name
    volume_name = args.volume_name

    convert(
        coco_file=coco_file_path,
        label_tree_path=label_tree_path,
        output_dir=output_dir,
        project_name=project_name,
        volume_name=volume_name
    )