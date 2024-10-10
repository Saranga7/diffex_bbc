
from experiment import *



def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.accum_batches = 4
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


# saranga
def bbc_autoenc_base(): 
    conf = autoenc_base()
    conf.data_name = 'bbc021_simple'    # name of the dataset, this must be a key present in the data_paths dictionary in dataset.py
    conf.scale_up_gpus(4)
    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.make_model_conf()
    return conf


# saranga
def bbc_autoenc(): 
    conf = bbc_autoenc_base()
    conf.total_samples = 9_000_000  # total number of samples to train on, Adjust this to increase or decrease the training time
    conf.eval_ema_every_samples = 500_000 # how often to evaluate the model
    conf.eval_every_samples = 500_000 #  how often to evaluate the eval model 
    conf.name = 'bbc_autoenc' # name of the configuration. The model will be saved as a directory of the same name inside checkpoints/
    conf.include_classifier = True # must be true.
    return conf

