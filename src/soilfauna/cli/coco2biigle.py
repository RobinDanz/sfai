def add_coco2biigle_parser(subparsers):
    parser = subparsers.add_parser(
        "coco2biigle",
        help="Convert a COCO annotation file to a Biigle volume."
    )
    
    parser.add_argument("-c", "--coco",
                    help="""Directory name""")

    # parser.add_argument("-o", "--out_dir",
    #                     default=DEFAULT_OUTPUT,
    #                     help="""Output directory""")

    parser.add_argument("-p", "--project_name",
                        default='project01',
                        help="""Project name""")

    parser.add_argument("-v", "--volume_name",
                        default='volume01',
                        help="""Volume name""")

    parser.add_argument('-t', '--label_tree_name')
    
def run_coco2biigle(args):
    print(args)
    
    