from pathlib import Path
import shutil
import os

try:
    import wandb
except Exception("[WARNING]: Module wandb is not installed"):
    pass


def init_wandb(debug=False):
    if not debug:
        wandb.init(
            project="latenciaga",
            sync_tensorboard=True
        )


def try_upload_artefacts(local_art_dir):
    """Uploads artefacts to path specified by env variable

    :param local_art_dir:
    :return:
    """
    if os.environ.get("OUTPUT_ARTEFACTS"):
        artefacts_dir = Path(os.environ.get("OUTPUT_ARTEFACTS"))
        shutil.rmtree(artefacts_dir, ignore_errors=True)
        shutil.copytree(local_art_dir, artefacts_dir)
