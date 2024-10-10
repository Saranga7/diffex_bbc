from templates import *

if __name__ == '__main__':
    # train the autoenc moodel
    gpus = [0, 1, 2, 3]

    #saranga
    conf = bbc_autoenc()
    conf.classifier_loss_start_step = 250_000 # After how many steps to trigger the KL Loss
    conf.batch_size = 128 
    conf.accum_batches = 2
    conf.include_classifier = True  # Keep this True for the classifier component
    conf.name = 'bbc_autoenc_KL_0.3'    # name of the configuration. The model will be saved as a directory of the same name inside checkpoints/

    train(conf, gpus=gpus)


    