
# Variational Autoencoder for Anomaly Detection: A Comparative Study

Team member:
- Nguyen Ha Huy Hoang
- Nguyen Cuong Nhat
- Dao Xuan Tung
- Duong Quoc Trung

This repository contains the implementation of the paper titled "Variational Autoencoder for Anomaly Detection: A Comparative Study". This paper code base is developed based on the code from [Hugo's original paper](https://github.com/HGangloff/vae_grf).

## Getting Started

Before running the experiments, ensure you have downloaded the [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) and [MiAD](https://miad-2022.github.io/) datasets. Modify the dataset links in `datasets.py` accordingly to link these datasets to your experiment.

### Prerequisites

The required libraries are listed in `requirements.txt`. Install them using the following command:

```bash
pip install -r requirements.txt
```

It is recommended to run the installation in a Conda environment, especially on Windows.

## Training

To train the model, execute the following command:

```bash
sh vae_train.sh
```

For training Variational Autoencoder (VAE) and VAE-GRF as per our experiments, ensure the following parameters:

- `batch_size`: 8
- `latent_image_size`: 32
- `latent_dim`: 256
- `image_size`: 256

In `vae_test.py`, modify `mad = mad.repeat(16, axis=0).repeat(16, axis=1)` to `mad = mad.repeat(8, axis=0).repeat(8, axis=1)` to run VAE and VAE-GRF.

For training ViT-VAE as per our experiments, ensure the following parameters:

- `batch_size`: 8
- `latent_image_size`: 14
- `latent_dim`: 384
- `image_size`: 224

In `vae_test.py`, modify `mad = mad.repeat(8, axis=0).repeat(8, axis=1)` to `mad = mad.repeat(16, axis=0).repeat(16, axis=1)` to run ViT-VAE.

## Testing

To test the model, use the following command:

```bash
sh vae_test.sh
```

Ensure to provide the appropriate parameters as mentioned above.

## Built With

The code is built using PyTorch and other standard libraries.

## More Information

For more details, please refer to the publication.

---

