from experiment import *


def autoenc_base(
    base_dir: str,
    data_cache_dir: str,
    work_cache_dir: str,
    name: str,
    data_path: str,
    classifier_path: str,
):
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig(
        base_dir=base_dir,
        data_cache_dir=data_cache_dir,
        work_cache_dir=work_cache_dir,
        name=name,
        data_path=data_path,
        classifier_path=classifier_path,
        batch_size=32,
        accum_batches=4,
        beatgans_gen_type=GenerativeType.ddim,
        beta_scheduler="linear",
        diffusion_type="beatgans",
        eval_ema_every_samples=200_000,
        eval_every_samples=200_000,
        fp16=True,
        lr=1e-4,
        model_name=ModelName.beatgans_autoenc,
        net_attn=(16,),
        net_beatgans_attn_head=1,
        net_beatgans_embed_channels=512,
        net_beatgans_resnet_two_cond=True,
        net_ch_mult=(1, 2, 4, 8),
        net_ch=64,
        net_enc_channel_mult=(1, 2, 4, 8, 8),
        net_enc_pool="adaptivenonzero",
        sample_size=32,
        T_eval=20,
        T=1000,
    )
    conf.make_model_conf()
    return conf


# saranga
def bbc_autoenc_base(
    base_dir: str,
    data_cache_dir: str,
    work_cache_dir: str,
    name: str,
    data_path: str,
    classifier_path: str,
):
    conf = autoenc_base(
        base_dir, data_cache_dir, work_cache_dir, name, data_path, classifier_path
    )
    conf.data_name = "bbc021_simple"  # name of the dataset, this must be a key present in the data_paths dictionary in dataset.py
    conf.scale_up_gpus(4)
    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    return conf


# saranga
def bbc_autoenc(
    base_dir: str,
    data_cache_dir: str,
    work_cache_dir: str,
    name: str,
    data_path: str,
    classifier_path: str,
):
    conf = bbc_autoenc_base(
        base_dir, data_cache_dir, work_cache_dir, name, data_path, classifier_path
    )
    conf.total_samples = 9_000_000  # total number of samples to train on, Adjust this to increase or decrease the training time
    conf.eval_ema_every_samples = 500_000  # how often to evaluate the model
    conf.eval_every_samples = 500_000  #  how often to evaluate the eval model
    conf.include_classifier = True  # must be true.
    return conf
