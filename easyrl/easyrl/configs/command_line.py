import argparse

from dataclasses import asdict
from dataclasses import fields


def cfg_from_cmd(cfg, parser=None):
    field_types = {field.name: field.type for field in fields(cfg)}
    default_cfg = asdict(cfg)
    if parser is None or not isinstance(parser, argparse.ArgumentParser):
        parser = argparse.ArgumentParser()

    for key, val in default_cfg.items():
        # add try except here so that we can define the
        # configs in the parser manually with
        # different default values
        try:
            if isinstance(val, bool):
                if not val:
                    parser.add_argument('--' + key, action='store_true')
                else:
                    parser.add_argument('--no_' + key, dest=key, action='store_false')
            else:
                parser.add_argument('--' + key, type=field_types[key], default=val)
        except argparse.ArgumentError:
            pass

    args = parser.parse_args()
    default_args = parser.parse_args([])
    args_dict = vars(args)
    default_args_dict = vars(default_args)
    diff_hps = {key: val for key, val in args_dict.items() if
                args_dict[key] != default_args_dict[key]}

    for key, val in args_dict.items():
        setattr(cfg, key, val)

    if len(diff_hps) > 0:
        setattr(cfg, 'diff_cfg', diff_hps)
    return args, diff_hps
