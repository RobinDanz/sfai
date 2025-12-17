from soilfauna.config import ConfigParser

def add_segment_parser(subparsers):
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
    
    parser.set_defaults(func=run_segmentation)
    
def run_segmentation(args):
    print("Automatic segmentation")
    print("Configuration file: ", args.config)
    parser = ConfigParser()
    config = parser.load_from_path(args.config)
    
    