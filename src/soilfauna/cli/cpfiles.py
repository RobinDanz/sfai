from pathlib import Path
from soilfauna.scripts import copy

def add_cpfiles_parser(subparsers):
    parser = subparsers.add_parser(
        "cpfiles",
        help="Tool to recursively copy or move files from source to destination."
    )
    
    parser.add_argument(
        '-s',
        '--source',
        help='Source folder. Subfolders will be explored',
        required=True
    )

    parser.add_argument(
        '-d',
        '--dest',
        help='Destination folder.',
        required=True
    )

    parser.add_argument(
        '-m',
        '--move',
        help='Move files instead copy',
        action='store_true',
    )
    
    parser.add_argument(
        '-e',
        '--extensions',
        nargs='*',
        default=[],
        help='File extensions'
    )
    
    parser.set_defaults(func=run_cpfiles)
    
def run_cpfiles(args):
    source = Path(args.source).absolute()
    destination = Path(args.dest).absolute()
    move = args.move
    extensions = args.extensions
    
    copy(source=source, destination=destination, extensions=extensions, move=move)