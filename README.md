# Diff-Ex on BBBC021

BBBC021 Data Path: /projects/deepdevpath/Anis/diffusion-comparison-experiments/datasets/bbc021_simple

## Running the trainings

### 0. Set-up

```
git clone <>
cd <>

conda env create -f environment.yml
conda activate diffae
```

### 1. Classifier training

```
cd Classifier
```

You can try different architectures from torchvision by `modifying model.py`

You can modify other hyperparameters in `config.py`

```
python classifier_training.py
```

After training is complete, you will have the weights for your classifier inside the directory `classifier_saved_models/`




### 2. Diff-Ex training

Go back to the parent directory

```
cd ..
```

Run Diff-Ex training on BBBC021 dataset.

```
python run_bbc.py
```


After the training is complete, you will find the weights of the the Diff-Ex model inside the directory, `checkpoints`


## Changes made w.r.t DiffAE

All the comments that I have made start with "saranga", so you can Ctrl + F and type "saranga" to see the changes that I have made.

I have made changes in the following files:

- `config.py`
- `dataset.py`
- `experiment.py` maximum changes here
- `renderer.py`
- `model/unet_autoenc.py`
- `diffusion/base.py`
- `templates.py`
- `run_bbc.py`

NOTE: For running baseline DiffAE (no classifier component), I would suggest to do it on a separate repository altogether instead of using the same repo for both Diff-Ex and DiffAE.





















