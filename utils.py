import argparse
from const import cmdline_args


def parse_cmdline() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for arg in cmdline_args:
        name_info, type_info, default_info, help_info = arg
        parser.add_argument("--" + name_info, type=type_info, default=default_info, help=help_info)

    return parser.parse_args()
