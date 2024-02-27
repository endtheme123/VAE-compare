# Variational Autoencoder for Anomaly Detection:A Comparative Study

This Git is the implementation of the paper "Variational Autoencoder for Anomaly Detection:A Comparative Study"

The model can be directly tested on the [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) and [MiAD](https://miad-2022.github.io/) dataset, but downloading these 2 dataset prior to the experiment run is required. Change the dataset link in `datasets.py` to link these dataset to the experiment.

The libraries we used in our enviroments was exported to the `requirements.txt` file => use `pip install -r requirements.txt` on your environment to rebuild ours. Running this in Conda enviroment is recommemded since we use Conda on Windows for the experiments.

To train a model run the file: use `sh vae_train.sh`. 
- to train VAE and VAE-GRF as our experiments, the `batch_size` should be 8, `latent_image_size` should be 32, and `latent_dim` should be 256, `image_size` should be 256, you also need to change the `mad = mad.repeat(16, axis=0).repeat(16, axis=1)` in `vae_test.py` part from 16 to 8 to run VAE and VAE-GRF

- to train ViT-VAE and as our experiments, the `batch_size` should be 8, `latent_image_size` should be 14, and `latent_dim` should be 384, `image_size` should be 224, you also need to change the `mad = mad.repeat(8, axis=0).repeat(8, axis=1)` in `vae_test.py` part from 8 to 16 to run ViT-VAE


To test a model run the file: `sh vae_test.sh` with appropriate parameters as above.

The code is built with PyTorch and other standard librairies.

For more details, refer to the publication.
