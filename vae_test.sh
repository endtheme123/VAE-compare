#!/bin/bash

python vae_test.py\
    --exp=test_vit_vae_mvtec_wood_new\
    --dataset=mvtec\
    --lr=1e-4\
    --img_size=224\
    --batch_size=8\
    --batch_size_test=8\
    --latent_img_size=14\
    --z_dim=384\
    --beta=1\
    --nb_channels=3\
    --model=vitvae\
    --corr_type=corr_id\
    --params_id=100\
