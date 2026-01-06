import argparse

from soilfauna.cli import add_segment_parser, add_coco2biigle_parser, add_cpfiles_parser

parser = argparse.ArgumentParser(
    prog='soilfauna',
    description='Set of tools to handle image files.'
)

parser.add_argument(
    "-l", "--log",
    help="Log level (0: Warning, 1: Info, 2: Debug) Defaults to 1.",
    required=False
)

subparsers = parser.add_subparsers(
    title="subcommands",
    dest="command",
    required=True
)

add_segment_parser(subparsers)
add_coco2biigle_parser(subparsers)
add_cpfiles_parser(subparsers)


def main():
    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__':
    main()