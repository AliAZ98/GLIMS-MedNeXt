#Last Modified: 07.08.2025 by Ali Azmoudeh

import argparse
import os
from functools import partial
import monai
import wandb
import torch.nn as nn

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR, PolyLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from GLIMS import GLIMS
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

import logging

from nnunet_mednext import create_mednext_v1
from nnunet_mednext.run.load_weights import upkern_load_weights

#import F
import torch.nn.functional as F

logging.disable(logging.WARNING)


#python main.py --output_dir /mnt/storage1/ziya/GLIMS/Github/outputs/ --data_dir /mnt/storage1/dataset/Medical/BraTS2023/Dataset --json_list /mnt/storage1/dataset/Medical/BraTS2023/Dataset/brats23_folds.json 

parser = argparse.ArgumentParser(description="GLIMS Brain Tumor Segmentation Pipeline")

parser.add_argument("--data_dir", default = "/Path/to/your/dataset_directory" ,type=str, help="dataset directory", required=False)
parser.add_argument("--json_list", default = "/Path/to/your/json_file", type=str, help="dataset json file", required=False)

parser.add_argument("--fold", default=5, type=int, help="data fold selected for validation")

parser.add_argument("--GLIMSweights", default = "/Path/to/your/weights_file", type=str, help="GLIMS pre-trained weights file path", required=False)
parser.add_argument("--MedNeXtweights", default = "/Path/to/your/weights_file", type=str, help="MedNeXt pre-trained weights file path", required=False)

parser.add_argument("--pretrained_dir", default='/Path/to/your/model_final.pt', type=str, help="Pretrained model directory")
parser.add_argument("--output_dir", default="/Path/to/your/Output", type=str, help="output directory")

parser.add_argument("--test_mode", default=False, type=bool, help="test mode")

parser.add_argument("--save_checkpoint", default = True, help="save checkpoint during training")  
parser.add_argument("--max_epochs", default=1000, type=int, help="max number of training epochs") #500

parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument("--batch_size_val", default=4, type=int, help="number of batch size for validation") #Does not have a backprop path, so can be larger.
parser.add_argument("--sw_batch_size", default=8, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=0.00003, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-4, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")

parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")

parser.add_argument("--amp", default=False, help="use amp for training") #AMP performs training faster, but high possibility to receive NaNs.

parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
parser.add_argument("--perform_test", default=False, type=bool, help="testing dataset check")
parser.add_argument("--distributed", default=False, help="start distributed training")

parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")

parser.add_argument("--workers", default=8, type=int, help="number of workers") #8

parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")

parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')
parser.add_argument("--infer_overlap", default=0.8, type=float, help="sliding window inference overlap")

parser.add_argument("--lrschedule", default="cosine_anneal", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")

parser.add_argument("--use_checkpoint", default=False, help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")

parser.add_argument("--seed", default=25, help="the random seed to produce deterministic results")

parser.add_argument("--wandb_enable", default=False, help="enable wandb logging")
parser.add_argument("--wandb_project_name", default="GLIMS_SSA_Project", help="the name that will be given to the WandB project")

parser.add_argument("--feature_size", default=24, type=int, help="feature size")

parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

#torch.autograd.set_detect_anomaly(True)

#torch.autograd.set_detect_anomaly(True)
wandb.login(key="WANDB_KEY") 
def main():
    args = parser.parse_args() #Parse the inputs
    
    if(args.wandb_enable):
        
        #🐝 initialize a wandb run
        wandb.init(
            project=args.wandb_project_name,
            config=args,
            reinit=True,
            name="ensemble-meta-decoder(unfreeze)",
            entity='simit'
        )

        

    #For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    monai.utils.misc.set_determinism(seed=args.seed)

    main_worker(args=args)

def main_worker(args):

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    torch.cuda.set_device(args.gpu)

    loader = get_loader(args) #Loader of the dataset (both training and validation)

    print(args.rank, " gpu", args.gpu)

    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    #Create the model.
    model1 = GLIMS(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        ensemble=False, 
    ) 

    model2 = create_mednext_v1(
        num_input_channels=4,
        num_classes=3,
        model_id='B',
        kernel_size=3,
        deep_supervision=True,
        ensemble=False,
    )

    #print(model2)

    class EnsembleFusion(nn.Module):
        def __init__(self, glims, mednext, out_ch=3, glims_weight=0.6, mednext_weight=0.4):
            super().__init__()
            
            self.glims = glims
            self.mednext = mednext
            self.glims_weight = glims_weight
            self.mednext_weight = mednext_weight

            # Fusion convolution layer after weighted summation
            self.fuse = nn.Sequential(
                nn.Conv3d(out_ch * 2, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, out_ch, kernel_size=1),
            )
            
            # load & freeze GLIMS
            ckpt1 = torch.load('/ari/users/aazmoudeh/GLIMS/Outputs/model.pt', map_location='cpu')
            self.glims.load_state_dict(ckpt1['state_dict'], strict=False)
            for p in self.glims.parameters(): p.requires_grad = True

            # load & freeze MedNeXt
            ckpt2 = torch.load('/ari/users/aazmoudeh/mednextbyGLIMS/SSA_Out/model_final.pt', map_location='cpu')
            self.mednext.load_state_dict(ckpt2['state_dict'], strict=False)
            for p in self.mednext.parameters(): p.requires_grad = True
        ###############################################################
        #### USE IN CASE OF FREEZING ENCODERS and DECODERS ############
        ###############################################################
        #     for module in (self.glims.dmsu_4, self.glims.dmsf_4, self.glims.out1):
        #         for p in module.parameters():
        #             p.requires_grad = True

        # # MedNeXt: final up‐3 block, its dec_block_3, and the full‐res out_0 head
        #     for module in (self.mednext.up_3, self.mednext.dec_block_3, self.mednext.out_0):
        #         for p in module.parameters():
        #             p.requires_grad = True

        def forward(self, x, val=False):
            gl_heads = self.glims(x, mode="Train")
            md_all   = self.mednext(x)
            md_heads = md_all[:-1]

            fused = []
            up_fused = []
            H, W, D = gl_heads[0].shape[2:]

            for g, m in zip(gl_heads, md_heads):
                if isinstance(g, (list, tuple)):
                    g = g[-1]
                if isinstance(m, (list, tuple)):
                    m = m[-1]

                # Apply fusion weights
                g = g * self.glims_weight
                m = m * self.mednext_weight

                # Concatenate and fuse
                cat = torch.cat([g, m], dim=1)  # shape: [B, 2*out_ch, D, H, W]
                f   = self.fuse(cat)            # [B, out_ch, d,h,w]
                fused.append(f)

                # 4) upsample back to full 96³
                up = F.interpolate(f, size=(D, H, W), mode='trilinear', align_corners=False)
                up_fused.append(up)

            if val:
                return fused[0] # or torch.stack(up_fused).mean(dim=0)
            else:
                return fused, up_fused
            

    class EnsembleAvg(nn.Module):
        def __init__(self, glims: nn.Module, mednext: nn.Module, out_ch=3):
            super().__init__()
            self.glims = glims
            self.mednext = mednext

            # Load & freeze GLIMS
            ckpt1 = torch.load(args.GLIMSweights, map_location='cpu')
            self.glims.load_state_dict(ckpt1['state_dict'], strict=False)
            for p in self.glims.parameters(): p.requires_grad = True

            # load & freeze MedNeXt
            ckpt2 = torch.load(args.MedNeXtweights, map_location='cpu')
            self.mednext.load_state_dict(ckpt2['state_dict'], strict=False)
            for p in self.mednext.parameters(): p.requires_grad = True
        ###############################################################
        #### USE IN CASE OF FREEZING ENCODERS and DECODERS ############
        ###############################################################
        #     for module in (self.glims.dmsu_4, self.glims.dmsf_4, self.glims.out1):
        #         for p in module.parameters():
        #             p.requires_grad = True

        # # MedNeXt: final up‐3 block, its dec_block_3, and the full‐res out_0 head
        #     for module in (self.mednext.up_3, self.mednext.dec_block_3, self.mednext.out_0):
        #         for p in module.parameters():
        #             p.requires_grad = True


        def forward(self, x, val=False):
            gl_heads = self.glims(x, mode="Train")
            if val:
                gl_heads = self.glims(x)
            md_all   = self.mednext(x)
            md_heads = md_all[:-1]

            fused = []
            up_fused = []
            H, W, D = gl_heads[0].shape[2:]

            for i, (g, m) in enumerate(zip(gl_heads, md_heads)):
                if isinstance(g, (list, tuple)): g = g[-1]
                if isinstance(m, (list, tuple)): m = m[-1]

                cat = torch.cat([g, m], dim=1)
                f = self.fuse_blocks[i](cat)
                fused.append(f)

                up = F.interpolate(f, size=(D, H, W), mode='trilinear', align_corners=False)
                up_fused.append(up)

            if val:
                return up_fused[0]  # or torch.stack(up_fused).mean(dim=0)
            else:
                return fused, up_fused
    #USING ENSEMBLE AVERAGIN MODEL       
    #model = EnsembleAvg(model1, model2)
    
    # USING ENSEMBLE FUSION MODEL
    model = EnsembleFusion(model1, model2, out_ch=args.out_channels,glims_weight=1.02, mednext_weight=0.98) #Ensemble model that fuses GLIMS and MedNeXt outputs.
    #model = upkern_load_weights(model, args.pretrained_dir) #Load the pretrained weights.

    # freeze the last encoder layer
    
    dice_loss = DiceCELoss(to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=1e-6, smooth_dr=1e-5, include_background=True)

    post_sigmoid = Activations(sigmoid=True)#output activation

    post_pred = AsDiscrete(argmax=False, threshold=0.5) 

    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True, ignore_empty=False) #Mean dice loss throught the batch.

    def freeze_encoder_except_last(model,freeze_list): #freeze
    # Freeze all layers except last encoder block (dmsf_4) and anything after
        
        for name, module in model.named_children(): 
            if name in freeze_list:
                for param in module.parameters():
                    #print(name, "is frozen\t")
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
    freeze_list = ['stem', 'dmsf_1', 'dmsf_2', 'dmsf_3', 'dmsu_1','dmsu_2','dmsu_3']
        
    # freeze_encoder_except_last(model,freeze_list)
    # print(freeze_list, "are frozen")
    


    #The inferer model that will perform the validation, "sliding_window_inference"
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    #----------------------------------------- TRAINING ----------------------------------------------------
    best_acc = 0
    start_epoch = 0

    #If there is a checkpoint, load it to the model.
    if args.use_checkpoint is True:
        model_dict = torch.load(args.pretrained_dir)["state_dict"]
        model.load_state_dict(model_dict)
        print("Pretrained model loaded from", args.pretrained_dir)

    model.cuda(args.gpu)

    #Parallel training on multiple GPUs if available.
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    #Optimizers
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight)
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    #LR Schedulers
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.pretrained_dir is not None:
            scheduler.step(epoch=start_epoch)
    elif args.lrschedule == "poly":
        scheduler = PolyLRScheduler(optimizer, initial_lr=args.optim_lr, max_steps=args.max_epochs)
        if args.pretrained_dir is not None:
            scheduler.step(current_step=start_epoch)
    elif args.lrschedule == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, verbose=True)
    else:
        scheduler = None

    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch, #will be 0 if no checkpoint was imported.
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
