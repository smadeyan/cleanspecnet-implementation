# CleanSpecNet Implementation

This repo contains the code for implementing the architecture specified in the [CleanUNet2](https://arxiv.org/abs/2309.05975) paper. Currently, we have implemented the CleanSpecNet model.

For downloading the dataset and creating the training/validation data, please run the `denoising-baselines.ipynb` notebook. Instructions on the parameters to use for synthesizing the data are mentioned in the comments in that notebook.

> The data is around 200 GB, so please plan accordingly.

If you'd like to run the baselines, please check the [CleanUNet](https://github.com/NVIDIA/CleanUNet) and [FullSubNet](https://github.com/Audio-WestlakeU/FullSubNet) sections in `denoising-baselines.ipynb`.

The entire pipeline, form dataloaders to inference, is specified in `CleanSpecNet.ipynb`, which was used to train/test the model. we have also modularized the code for ease of reference and use. The notebook has been split into the relevant sections, including ones that specify the requirements and to load relevant checkpoints for inference.