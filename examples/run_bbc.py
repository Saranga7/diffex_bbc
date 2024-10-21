from templates import bbc_autoenc
from experiment import train
import torch
import wandb

if __name__ == "__main__":
    # run name
    name = "run_name" # used to name the checkpoints folder & the wandb run

    # select GPUS
    gpus = [0, 1, 2, 3]

    # saranga
    conf = bbc_autoenc(
        base_dir="path/to/dir",
        data_cache_dir="path/to/dir",
        work_cache_dir="path/to/dir",
        data_path="path/to/dir",
        classifier_path="path/to/dir",
        name=name,
    )
    # set torch hub dir 
    torch.hub.set_dir("path/to/dir")
    conf.classifier_loss_start_step = (
        250_000  # After how many steps to trigger the KL Loss)
    )
    conf.annealing_steps = 250_000  # Over how many steps to anneal the KL Loss
    conf.batch_size = 64
    conf.accum_batches = 2
    conf.include_classifier = True  # Keep this True for the classifier component
    conf.cls_weight = 0.3  # Weight of the classifier loss

    wandb.init(config = conf.as_dict_jsonable(), name=name)

    train(conf, gpus=gpus)
