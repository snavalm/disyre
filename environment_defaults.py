import os
disyre_root = os.path.dirname(os.path.abspath(__file__))

# env_default = None

## Use this example to set different directories for different machines
# import socket
# hostname = socket.gethostname()
# if 'aaa' in hostname:
#     env_default = {}
# elif 'bbb' in hostname:
#     env_default = {}
# else:
#     env_default = {}

env_default = {
    "checkpoint_dir": '/workspace/checkpoints/disyre/',
    "results_dir": '/workspace/disyre/results_csv/',
    "log_dir": '/workspace/logs/',
    "brats_json": os.path.join(disyre_root, 'experiments/brats_upd_wTestSplit.json'),
    "brats_base_dir": '/workspace/data/brats20/',
    "atlas_json": os.path.join(disyre_root, 'experiments/atlas_upd_issues_wTestSplit.json'),
    "atlas_base_dir": '/workspace/data/atlas/',
    "camcan_json": os.path.join(disyre_root, 'experiments/camcan_upd.json'),
    "camcan_base_dir": '/workspace/data/camcan_upd/',
    "shape_dataset": os.path.join(disyre_root, "experiments/shapes/dataset.json"),
    "shape_base_dir": os.path.join(disyre_root, "experiments/shapes/"),
}

env_default["wandb_entity"] = "snavalm"

if env_default is not None:
    for k,v in env_default.items():
        if k not in os.environ:
            os.environ[k] = v

