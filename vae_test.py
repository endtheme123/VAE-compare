

import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity
import torch
# from vqvae import VQVAE
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from utils import (get_train_dataloader,
                   get_test_dataloader,
                   load_model_parameters,
                   load_vqvae,
                   parse_args
                   )

def ssim(a, b, win_size):
    "Structural di-SIMilarity: SSIM"
    a = a.detach().cpu().permute(1, 2, 0).numpy()
    b = b.detach().cpu().permute(1, 2, 0).numpy()
    # print(a)
    # b = gaussian_filter(b, sigma=2)

    try:
        score, full = structural_similarity(a, b, #multichannel=True,
            channel_axis=2, full=True, win_size=win_size,data_range=1.0)
    except ValueError: # different version of scikit img
        score, full = structural_similarity(a, b, multichannel=True,
            channel_axis=2, full=True, win_size=win_size,data_range=1.0)
    #return 1 - score, np.median(1 - full, axis=2)  # Return disim = (1 - sim)
    return 1 - score, np.product((1 - full), axis=2)

def get_error_pixel_wise(model, x, loss="rec_loss"):
    
    
    x_rec, _ = model(x)
    
    return x_rec

def create_segmentation(amaps, amaps2, rec_maps, dataset):
    if isinstance(amaps, torch.Tensor):
        
        amaps = amaps.detach().cpu().numpy()
    torch.set_printoptions(threshold=10_000)
    
    mask = (amaps>= 0.01).astype(np.int8) #NOTE 0.59 if M=512 !!!
    print(np.amax(mask))
    
    mask2 = ((amaps2 >= 0.01)).astype(np.int8) #0.01 for VQVAE

    mask2 = binary_dilation(mask2, iterations=4).astype(np.uint8)

    ## get intersection
    L, _ = label(mask2)
    lbls_interest = L[(mask == 1)]
    lbls_interest = lbls_interest[lbls_interest != 0]
    mask_ = mask.copy()
    mask = (np.isin(L, lbls_interest)).astype(np.int8)
    print(np.amax(mask))
    return mask, mask2, mask_, L

def test(args):
    ''' livestock testing pipeline '''
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print("Pytorch device:", device)

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    checkpoints_dir ="./torch_checkpoints"
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    checkpoints_saved_dir ="./torch_checkpoints_saved"

    predictions_dir ="./" + args.dataset + "_predictions"
    if not os.path.isdir(predictions_dir):
        os.mkdir(predictions_dir)

    # Load dataset
    train_dataloader, train_dataset = get_train_dataloader(
        args,
        fake_dataset_size=256,
    )
    # NOTE force test batch size to be 1
    args.batch_size_test = 1
    # fake_dataset_size=None leads a test on all the test dataset
    test_dataloader, test_dataset = get_test_dataloader(
        args,
        fake_dataset_size=None
    )

    # Load model
    model = load_vqvae(args)
    model.to(device)

    try:
        file_name = f"{args.exp}_{args.params_id}.pth"
        model = load_model_parameters(model, file_name, checkpoints_dir,
            checkpoints_saved_dir, device)
    except FileNotFoundError:
        raise RuntimeError("The model checkpoint does not exist !")

    dissimilarity_func = ssim

    classes = {}

    model.eval()

    aucs = []
    aucs_sm = []
    aucs_mad_sm = []

    pbar = tqdm(test_dataloader)
    i=0
    for imgs, gt in pbar:
        imgs = imgs.to(device)
        i+=1
        if args.dataset in ["livestock","mvtec","miad"]:
            # gt is a segmentation mask
            gt_np = gt[0].permute(1, 2, 0).cpu().numpy()[..., 0]
            gt_np = (gt_np - np.amin(gt_np)) / (np.amax(gt_np) - np.amin(gt_np))

            

        with torch.no_grad():
            x_rec = get_error_pixel_wise(model, imgs)
            x_rec = model.mean_from_lambda(x_rec)

        if args.dataset == "livestock" or args.dataset == "mvtec" or args.dataset == "miad":
            score, ssim_map = dissimilarity_func(x_rec[0], imgs[0], 11)

        ssim_map = ((ssim_map - np.amin(ssim_map)) / (np.amax(ssim_map)
        - np.amin(ssim_map)))

        
        # model.mu = F.interpolate(model.mu, size=(28, 28), mode='bilinear', align_corners=False)
        
        mad = torch.mean(torch.abs(model.mu - torch.mean(model.mu,
            dim=(0,1))), dim=(0,1))

        mad = mad.detach().cpu().numpy()

        mad = ((mad - np.amin(mad)) / (np.amax(mad)
            - np.amin(mad)))

        mad = mad.repeat(8, axis=0).repeat(8, axis=1)

        # MAD metric
        amaps = mad
        
        # SM metric
        amaps_sm = ssim_map

        # MAD*SM metric
        amaps_mad_sm = mad * ssim_map

        amaps = ((amaps - np.amin(amaps)) / (np.amax(amaps)
            - np.amin(amaps)))

        amaps_sm = ((amaps_sm - np.amin(amaps_sm)) / (np.amax(amaps_sm)
            - np.amin(amaps_sm)))

        amaps_mad_sm = ((amaps_mad_sm- np.amin(amaps_mad_sm)) / (np.amax(amaps_mad_sm)
            - np.amin(amaps_mad_sm)))

        if  args.dataset in ["livestock","mvtec", "miad"]:
            preds = amaps.copy() 
            preds_sm = amaps_sm.copy()
            preds_mad_sm = amaps_mad_sm.copy()
            mask = np.zeros(gt_np.shape)

            try:
                auc = roc_auc_score(gt_np.astype(np.int8).flatten(), preds.flatten())
                auc_sm = roc_auc_score(gt_np.astype(np.int8).flatten(), preds_sm.flatten())
                auc_mad_sm = roc_auc_score(gt_np.astype(np.int8).flatten(), preds_mad_sm.flatten())
                aucs.append(auc)
                # print("auc_sm", auc_sm)
                aucs_sm.append(auc_sm)
                aucs_mad_sm.append(auc_mad_sm)
            except ValueError:
                pass
                # ROCAUC will not be defined when one class only in y_true
        m_aucs = np.mean(aucs)
        m_aucs_sm = np.mean(aucs_sm)
        m_aucs_mad_sm = np.mean(aucs_mad_sm)
        print(m_aucs_mad_sm )
        pbar.set_description(f"mean ROCAUC: {m_aucs_mad_sm :.3f}")

        ori = imgs[0].permute(1, 2, 0).cpu().numpy()
        gt = gt[0].permute(1, 2, 0).cpu().numpy()
        rec = x_rec[0].detach().permute(1, 2, 0).cpu().numpy()
        path_to_save = args.dataset + '_predictions/'
        img_to_save = Image.fromarray((ori * 255).astype(np.uint8))
        img_to_save = Image.fromarray((ori * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'ori_'+ args.exp+'.png')
        img_to_save = Image.fromarray((gt_np * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'gt_'+ args.exp+'.png')
        img_to_save = Image.fromarray((rec * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'rec_'+ args.exp+'.png')
        cm = plt.get_cmap('jet')
        amaps_mad_sm = cm(amaps_mad_sm)
        img_to_save = Image.fromarray((amaps_mad_sm[..., :3] * 255).astype(np.uint8))
        img_to_save.save(path_to_save + 'final_amap_'+ args.exp+'.png')

    m_auc = np.mean(aucs)
    m_auc_sm = np.mean(aucs_sm, dtype=object)
    m_auc_mad_sm = np.mean(aucs_mad_sm)
    std_auc = np.std(aucs)
    std_auc_sm = np.std(aucs_sm, dtype=object)
    std_auc_mad_sm = np.std(aucs_mad_sm)
    print("Mean auc on", args.category, args.defect, m_auc)
    print("Mean auc_sm on", args.category, args.defect, m_auc_sm)
    print("Mean auc_mad_sm on", args.category, args.defect, m_auc_mad_sm)
    print("std auc on", args.category, args.defect, std_auc)
    print("std auc_sm on", args.category, args.defect, std_auc_sm)
    print("std auc_mad_sm on", args.category, args.defect, std_auc_mad_sm)
    return m_auc, m_auc_sm, m_auc_mad_sm



def test_on_train(args, model):
    ''' livestock testing pipeline '''
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print("Pytorch device:", device)

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    checkpoints_dir ="./torch_checkpoints"
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    checkpoints_saved_dir ="./torch_checkpoints_saved"

    predictions_dir ="./" + args.dataset + "_predictions"
    if not os.path.isdir(predictions_dir):
        os.mkdir(predictions_dir)

    # Load dataset
    train_dataloader, train_dataset = get_train_dataloader(
        args,
        fake_dataset_size=256,
    )
    # NOTE force test batch size to be 1
    args.batch_size_test = 1
    # fake_dataset_size=None leads a test on all the test dataset
    test_dataloader, test_dataset = get_test_dataloader(
        args,
        fake_dataset_size=None
        
    )

    

    dissimilarity_func = ssim

    classes = {}

    model.eval()

    aucs = []

    pbar = tqdm(test_dataloader)
    for imgs, gt in pbar:
        imgs = imgs.to(device)
        if args.dataset in ["livestock","mvtec","miad"]:
            # gt is a segmentation mask
            gt_np = gt[0].permute(1, 2, 0).cpu().numpy()[..., 0]
            gt_np = (gt_np - np.amin(gt_np)) / (np.amax(gt_np) - np.amin(gt_np))

        
        
        
        with torch.no_grad():
            x_rec = get_error_pixel_wise(model, imgs)
            x_rec = model.mean_from_lambda(x_rec)

        if args.dataset == "livestock" or args.dataset == "mvtec" or args.dataset == "miad":
            score, ssim_map = dissimilarity_func(x_rec[0], imgs[0], 11)

        ssim_map = ((ssim_map - np.amin(ssim_map)) / (np.amax(ssim_map)
        - np.amin(ssim_map)))

        # model.mu = F.interpolate(model.mu, size=(28, 28), mode='bilinear', align_corners=False)
        
        mad = torch.mean(torch.abs(model.mu - torch.mean(model.mu,
            dim=(0,1))), dim=(0,1))

        mad = mad.detach().cpu().numpy()

        mad = ((mad - np.amin(mad)) / (np.amax(mad)
            - np.amin(mad)))

        mad = mad.repeat(8, axis=0).repeat(8, axis=1)

        # MAD metric
        # amaps = mad

        # SM metric
        #amaps = ssim_map

        # MAD*SM metric
        amaps = mad * ssim_map

        amaps = ((amaps - np.amin(amaps)) / (np.amax(amaps)
            - np.amin(amaps)))
        if args.dataset in ["livestock","mvtec","miad"]:
            preds = amaps.copy() 
            mask = np.zeros(gt_np.shape)

            try:
                auc = roc_auc_score(gt_np.astype(np.int8).flatten(), preds.flatten())
                aucs.append(auc)
            except ValueError:
                pass
                # ROCAUC will not be defined when one class only in y_true

        m_aucs = np.mean(aucs)
        print(m_aucs)
        pbar.set_description(f"mean ROCAUC: {m_aucs:.3f}")


    m_auc = np.mean(aucs)
    print("Mean auc on", args.category, args.defect, m_auc)

    return m_auc

if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "livestock" or args.dataset == "mvtec" or args.dataset == "miad":
        m_auc, m_auc_sm, m_auc_mad_sm = test(
            args,
            )





