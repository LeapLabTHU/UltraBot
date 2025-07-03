import os
from easydict import EasyDict
from .logger import print_log

def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger=logger)

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger=logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger=logger)

def save_experiment_config(args, config, logger=None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))
    print_log(f'Copy the Config file from {args.config} to {config_path}', logger=logger)
