#Last Modified: 28.05.2024 by Ziya Ata Yazici
#ENSEMBLE TESTING - BraTS 2023 Submision - Post-Processing

import argparse
import os
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

import nibabel as nib
import numpy as np
import torch
from utils.data_utils_test import get_loader

from monai.inferers import sliding_window_inference

from GLIMS import GLIMS

from monai.transforms import Activations, AsDiscrete

import argparse

import cc3d

import SimpleITK as sitk
from nnunet_mednext import create_mednext_v1
import scipy.ndimage as ndi  

#python test_BraTS.py --data_dir /mnt/storage1/dataset/Medical/BraTS2023/Dataset/ValidationData --model_ensemble_1 /mnt/storage1/ziya/BraTS_Models/Archive/HybridEncoder/Log/adjusted_hybrid_file_4_2_poseb_fold2/model_2_new.pt --model_ensemble_2 /mnt/storage1/ziya/BraTS_Models/Archive/HybridEncoder/Log/adjusted_hybrid_file_4_2_poseb_fold4/model_4_new.pt --output_dir /mnt/storage1/ziya/GLIMS/Github

parser = argparse.ArgumentParser(description="GLIMS Brain Tumor Segmentation Pipeline")
parser.add_argument("-f")

parser.add_argument("--data_dir", default= "/mnt/storage1/ali/Medilcal/BraTS2024-SSA-Challenge-ValidationData",type=str, help="dataset directory", required=False)
parser.add_argument("--model_ensemble_1", default="/mnt/storage1/ali/Medilcal/GLIMS/Outputs/model_1.pt" ,type=str, help="pretrained model name", required=False)
parser.add_argument("--model_ensemble_2", default="/mnt/storage1/ali/Medilcal/GLIMS/Outputs/model_1.pt" , type=str, help="pretrained model name", required=False)
parser.add_argument("--output_dir", default="/mnt/storage1/ali/Medilcal/subOUT" ,type=str, help="Segmentation mask output directory", required=False)

parser.add_argument("--exp_name", default="test", type=str, help="experiment name")
parser.add_argument("--feature_size", default=24, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=2, type=int, help="number of workers")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")

def main():
    args = parser.parse_args()

    args.test_mode = True

    output_directory = os.path.join(args.output_dir, args.exp_name)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    test_loader = get_loader(args) #Get loader of the testing data

    device = torch.device("cuda")

    model1 = GLIMS(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
    )

    model2 = create_mednext_v1(
        num_input_channels=4,
        num_classes=3,
        model_id='B',
        kernel_size=3,
        deep_supervision=True,
    )

    #print(model2)

    class EnsembleFusion(nn.Module):
        def __init__(self, glims, mednext, out_ch=3):
            super().__init__()
            
            self.glims   = glims
            self.mednext = mednext
            self.fuse = nn.Conv3d(out_ch*2, out_ch, kernel_size=1)

                # our little 1×1×1 fusion conv
            self.fuse = nn.Sequential(
                                        nn.Conv3d(6, 16, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(16, 3, kernel_size=1),
                                    )

            # load & freeze GLIMS
            ckpt1 = torch.load('/mnt/storage1/ali/Medilcal/GLIMS/Outputs/model.pt', map_location='cpu',weights_only=False)
            #self.glims.load_state_dict(ckpt1['state_dict'], strict=False)
            for p in self.glims.parameters(): p.requires_grad = False

            # load & freeze MedNeXt
            ckpt2 = torch.load('/mnt/storage1/ali/Medilcal/mednextbyGLIMS/SSA_Out/model_.pt', map_location='cpu',weights_only=False)
            #self.mednext.load_state_dict(ckpt2['state_dict'], strict=False)
            for p in self.mednext.parameters(): p.requires_grad = False

            for module in (self.glims.dmsf_4, self.glims.dmsu_4, self.glims.dmsf_4, self.glims.out1):
                for p in module.parameters():
                    p.requires_grad = True

        # MedNeXt: final up‐3 block, its dec_block_3, and the full‐res out_0 head
            for module in (self.mednext.enc_block_3, self.mednext.down_3, self.mednext.up_3, self.mednext.dec_block_3, self.mednext.out_0):
                for p in module.parameters():
                    p.requires_grad = True

        def forward(self, x,val=False):
            # 1) get lists of heads
            gl_heads = self.glims(x)      # list of 4: [96³,48³,24³,12³]
            md_all   = self.mednext(x)    # list of 5: [96³,48³,24³,12³,6³]

            # 2) drop the extra MedNeXt head (6³) so we have exactly 4
            md_heads = md_all[:-1]        # [96³,48³,24³,12³]

            fused   = []
            up_fused = []
            H, W, D = gl_heads[0].shape[2:]  # full-res dims
            
            
            for g, m in zip(gl_heads, md_heads):
                # if either returned a tuple/list, grab last element
                #print("g shape:", g.shape, "m shape:", m.shape)
                if isinstance(g, (list, tuple)): g = g[-1]
                if isinstance(m, (list, tuple)): m = m[-1]

                # 3) fuse at native scale
                cat = torch.cat([g, m], dim=1)  # [B, 2*out_ch, d,h,w]
                f   = self.fuse(cat)            # [B, out_ch, d,h,w]
                fused.append(f)

                # 4) upsample back to full 96³
                up = F.interpolate(f, size=(D, H, W), mode='trilinear', align_corners=False)
                up_fused.append(up)
            if val:
                return fused
            else:
            # return both the native-scale list AND the upsampled list
                return fused # length‐4 list: [96³,48³,24³,12³]
            

    class EnsembleAvg(nn.Module):
        def __init__(self, glims: nn.Module, mednext: nn.Module, out_ch=3):
            super().__init__()
            self.glims = glims
            self.mednext = mednext

            # Load & freeze GLIMS
            ckpt1 = torch.load('/mnt/storage1/ali/Medilcal/GLIMS/Outputs/model.pt', map_location='cpu')
            self.glims.load_state_dict(ckpt1['state_dict'], strict=False)
            for p in self.glims.parameters():
                p.requires_grad = False

            # Load & freeze MedNeXt
            ckpt2 = torch.load('/mnt/storage1/ali/Medilcal/mednextbyGLIMS/SSA_Out/model_.pt', map_location='cpu')
            self.mednext.load_state_dict(ckpt2['state_dict'], strict=False)
            for p in self.mednext.parameters():
                p.requires_grad = False

        #     for module in (self.glims.dmsf_4, self.glims.dmsu_4, self.glims.dmsf_4, self.glims.out1):
        #         for p in module.parameters():
        #             p.requires_grad = True

        # # MedNeXt: final up‐3 block, its dec_block_3, and the full‐res out_0 head
        #     for module in (self.mednext.enc_block_3, self.mednext.down_3, self.mednext.up_3, self.mednext.dec_block_3, self.mednext.out_0):
        #         for p in module.parameters():
        #             p.requires_grad = True

        def forward(self, x, val=False):
            # Get lists of heads from each model
            gl_heads = self.glims(x)      # list of 4: [96³,48³,24³,12³]
            md_all = self.mednext(x)      # list of 5: [96³,48³,24³,12³,6³]

            # Drop the extra MedNeXt head (6³) to match GLIMS length
            md_heads = md_all[:-1]        # [96³,48³,24³,12³]

            fused = []
            up_fused = []
            H, W, D = gl_heads[0].shape[2:]

            for g, m in zip(gl_heads, md_heads):
                # Unwrap if outputs are lists/tuples
                if isinstance(g, (list, tuple)): g = g[-1]
                if isinstance(m, (list, tuple)): m = m[-1]

                # Element-wise averaging fusion
                f = (g + m) * 0.5
                fused.append(f)

                # Upsample back to full 96³
                up = F.interpolate(f, size=(D, H, W), mode='trilinear', align_corners=False)
                up_fused.append(up)

            if val:
                return fused
            else:
                # Return both native-scale and upsampled outputs
                return fused, up_fused
            
    #model = EnsembleAvg(model1, model2)
        
    model = EnsembleFusion(model1, model2, out_ch=args.out_channels)
    #model = upkern_load_weights(model, args.pretrained_dir) #Load the pretrained weights.
    #load weights
    model_dict = torch.load('/mnt/storage1/ali/Medilcal/mednextbyGLIMS/choosen/model_709_0.8873171011606852.pt',weights_only=False)["state_dict"]
    model.load_state_dict(model_dict, strict=True)
    model = model.to(device)

    model3 =EnsembleFusion(model1, model2, out_ch=args.out_channels)
    #load weights
    model_dict = torch.load('/mnt/storage1/ali/Medilcal/mednextbyGLIMS/choosen/model_609_0.8622510433197021.pt',weights_only=False)["state_dict"]
    model3.load_state_dict(model_dict, strict=True)
    model3 = model3.to(device)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )

    model_inferer_test2 = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model3,
        overlap=args.infer_overlap,
    )

    with torch.no_grad():

        post_sigmoid = Activations(sigmoid=True)  # output activation
        post_predTC = AsDiscrete(argmax=False, threshold=0.65)
        post_predWT = AsDiscrete(argmax=False, threshold=0.55)
        post_predET = AsDiscrete(argmax=False, threshold=0.6)

        for i, batch in enumerate(test_loader):

            image = batch["image"].cuda()
            print("Image shape:", image.shape)
            affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            img_name = num + ".nii.gz"

            print("Inference on case {}".format(img_name))

            # Get logits from both models and average them
            logits1 = model_inferer_test(image)
            logits2 = model_inferer_test2(image)
            logits = [(logits1[0] + logits2[0]) / 2]

            # Apply sigmoid and thresholds
            sigmoid = post_sigmoid(logits[0])
            TC = post_predTC(sigmoid[0])
            WT = post_predWT(sigmoid[1])
            ET = post_predET(sigmoid[2])

            seg = torch.stack([TC, WT, ET])  # shape: [3, H, W, D]
            seg_out = torch.zeros((seg.shape[1], seg.shape[2], seg.shape[3]), dtype=torch.uint8, device=seg.device)

            # Apply class-priority: ET > NCR > ED
            seg_out[seg[2] == 1] = 3  # ET
            seg_out[(seg[0] == 1) & (seg_out == 0)] = 1  # NCR
            seg_out[(seg[1] == 1) & (seg_out == 0)] = 2  # ED

            seg_out = seg_out.cpu().numpy()

            # Connected Component Filtering
            npy = sigmoid.cpu().numpy()

            # ET filtering
            ET_img = seg_out == 3
            cc = cc3d.connected_components(ET_img, connectivity=26)
            for i in np.unique(cc):
                if i == 0:
                    continue
                if (ET_img[cc == i].size < 75) and (npy[2][cc == i].mean() < 0.9):
                    seg_out[cc == i] = 1  # ET → NCR

            # NCR filtering
            NCR_img = seg_out == 1
            cc = cc3d.connected_components(NCR_img, connectivity=26)
            for i in np.unique(cc):
                if i == 0:
                    continue
                if (NCR_img[cc == i].size < 75) and (npy[0][cc == i].mean() < 0.9):
                    seg_out[cc == i] = 2  # NCR → ED

            # ED filtering
            ED_img = seg_out == 2
            cc = cc3d.connected_components(ED_img, connectivity=26)
            for i in np.unique(cc):
                if i == 0:
                    continue
                if (ED_img[cc == i].size < 500) and (npy[1][cc == i].mean() < 0.9):
                    seg_out[cc == i] = 0  # ED → background

            # Hole filling in ET
            completeVolume = sitk.GetImageFromArray(seg_out.astype(np.uint8))
            closedcompleteVolume = sitk.BinaryFillhole(completeVolume, fullyConnected=True, foregroundValue=3)
            closedCompleteVolume = sitk.GetArrayFromImage(closedcompleteVolume)

            pixCount = np.count_nonzero(closedCompleteVolume != seg_out)
            if pixCount > 0:
                print("Filling holes for:", img_name, "for", pixCount, "pixels", flush=True)
                seg_out[closedCompleteVolume != seg_out] = 1  # holes → NCR

            # Save result
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))

    # with torch.no_grad():

    #     post_sigmoid = Activations(sigmoid=True)#output activation
    #     post_predTC = AsDiscrete(argmax=False, threshold=0.65)
    #     post_predWT = AsDiscrete(argmax=False, threshold=0.55)
    #     post_predET = AsDiscrete(argmax=False, threshold=0.6)

    #     for i, batch in enumerate(test_loader):

    #         image = batch["image"].cuda()
    #         print("Image shape:", image.shape)
    #         affine = batch["image_meta_dict"]["original_affine"][0].numpy()
    #         num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
    #         img_name = num + ".nii.gz"

    #         print("Inference on case {}".format(img_name))

    #         #get logits
    #         logits = model_inferer_test(image) # 3, 240, 240, 155
    #         #logits2 = model_inferer_test2(image) # 3, 240, 240, 155

    #         logits = (logits)# + logits2)/2

    #         sigmoid = post_sigmoid(logits[0])
    #         TC = post_predTC(sigmoid[0])
    #         WT = post_predWT(sigmoid[1])
    #         ET = post_predET(sigmoid[2])

    #         val_output_convert = torch.stack([TC, WT, ET])

    #         # seg = val_output_convert
    #         # seg_out = torch.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))

    #         # seg_out[seg[1] == 1] = 2 #ED
    #         # seg_out[seg[0] == 1] = 1 #NCR
    #         # seg_out[seg[2] == 1] = 3 #ET
    #         seg = val_output_convert  # seg shape: [3, H, W, D]
    #         seg_out = torch.zeros((seg.shape[1], seg.shape[2], seg.shape[3]), dtype=torch.uint8, device=seg.device)


    #         # Apply ET first (highest priority)
    #         seg_out[seg[2] == 1] = 3  # ET

    #         # Apply NCR next (middle priority), but don't overwrite ET
    #         seg_out[(seg[0] == 1) & (seg_out == 0)] = 1  # NCR

    #         # Apply ED last (lowest priority), only where nothing else is present
    #         seg_out[(seg[1] == 1) & (seg_out == 0)] = 2  # ED

    #         seg_out = seg_out.cpu().numpy()
    #        #====================

    #         #get the ET label
    #         ET_img = seg_out == 3


    #         #apply sigmoid pytorch
    #         npy = post_sigmoid(logits[0]).cpu().numpy()
            
    #         #cc3d connected component analysis
    #         cc = cc3d.connected_components(ET_img, connectivity=26)
    #         for i in np.unique(cc):
    #             if i == 0:
    #                 continue

    #             if(ET_img[cc == i].size < 75):
    #                 if((npy[-1][cc == i].mean() < 0.9)): #Check ET probability
    #                     seg_out[cc == i] = 1 #assign ET to NCR

    #         #======================

    #         #get the NCR label
    #         NCR_img = seg_out == 1

    #         #apply sigmoid pytorch
    #         npy = post_sigmoid(logits[0]).cpu().numpy()
            
    #         #cc3d connected component analysis
    #         cc = cc3d.connected_components(NCR_img, connectivity=26)
    #         for i in np.unique(cc):
    #             if i == 0:
    #                 continue

    #             if(NCR_img[cc == i].size < 75):
    #                 if((npy[-3][cc == i].mean() < 0.9)): #Check TC probability
    #                     seg_out[cc == i] = 2 #assign NCR to ED

    #         #======================

    #         ED_img = seg_out == 2

    #         cc = cc3d.connected_components(ED_img, connectivity=26)
    #         for i in np.unique(cc):
    #             if i == 0:
    #                 continue
    #             if(ED_img[cc == i].size < 500):
    #                 if((npy[-2][cc == i].mean() < 0.9)): #Check WT probability
    #                     seg_out[cc == i] = 0 #ED to background

    #         #======================

    #         completeVolume = sitk.GetImageFromArray(seg_out.astype(np.uint8))

    #         closedcompleteVolume = sitk.BinaryFillhole(completeVolume, fullyConnected= True, foregroundValue=3)
    #         closedCompleteVolume = sitk.GetArrayFromImage(closedcompleteVolume)

    #         #count label 1 in completeVolume
    #         pixCount = np.count_nonzero(closedCompleteVolume != seg_out)
    #         if(pixCount > 0):
    #             print("Filling holes for:", img_name, "for", pixCount, "pixels", flush=True)
    #             seg_out[closedCompleteVolume != seg_out] = 1 #Empty pixels in ET assign them to NCR

    #         #=========================
    #         nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))



    # with torch.no_grad():
    #     post_sigmoid = Activations(sigmoid=True)  # output activation
    #     post_predTC = AsDiscrete(argmax=False, threshold=0.65)
    #     post_predWT = AsDiscrete(argmax=False, threshold=0.55)
    #     post_predET = AsDiscrete(argmax=False, threshold=0.6)

    #     print("Starting inference on test set...")
    #     count = 0

    #     for i, batch in enumerate(test_loader):
    #        # print(batch["image_meta_dict"]["filename_or_obj"])
    #         count += 1
    #         print("sample:", count, flush=True)

    #         image = batch["image"].cuda()
    #         affine = batch["image_meta_dict"]["original_affine"][0].numpy()
    #         num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
    #         img_name = num + ".nii.gz"

    #         print("Inference on case {}".format(img_name))

    #         # Get logits
    #         logits = model_inferer_test(image)
    #         sigmoid = post_sigmoid(logits[0])  # shape: [3, H, W, D]

    #         # === Gaussian smoothing applied to each class before thresholding ===
    #         smoothed_sigmoid = torch.empty_like(sigmoid)
    #         for c in range(3):
    #             smoothed = ndi.gaussian_filter(sigmoid[c].cpu().numpy(), sigma=1.0)
    #             smoothed_sigmoid[c] = torch.from_numpy(smoothed).to(sigmoid.device)

    #         # Apply thresholds on smoothed outputs
    #         TC = post_predTC(smoothed_sigmoid[0])  # label 1
    #         WT = post_predWT(smoothed_sigmoid[1])  # label 2
    #         ET = post_predET(smoothed_sigmoid[2])  # label 3

    #         seg = torch.stack([TC, WT, ET])  # shape: [3, H, W, D]
    #         seg_out = torch.zeros((seg.shape[1], seg.shape[2], seg.shape[3]), dtype=torch.uint8, device=seg.device)

    #         # Class-priority resolution: ET > NCR > ED
    #         seg_out[seg[2] == 1] = 3  # ET
    #         seg_out[(seg[0] == 1) & (seg_out == 0)] = 1  # NCR (TC)
    #         seg_out[(seg[1] == 1) & (seg_out == 0)] = 2  # ED (WT)

    #         # === Confidence filtering ===
    #         # max_confidence = torch.max(smoothed_sigmoid, dim=0).values  # [H, W, D]
    #         # seg_out[max_confidence < 0.5] = 0  # low-confidence → background

    #         seg_out = seg_out.cpu().numpy()
    #         npy = smoothed_sigmoid.cpu().numpy()

    #         # =================== Connected Component Filtering ===================
    #         # ET
    #         ET_img = seg_out == 3
    #         cc = cc3d.connected_components(ET_img, connectivity=26)
    #         for i in np.unique(cc):
    #             if i == 0:
    #                 continue
    #             if (ET_img[cc == i].size < 75) and (npy[2][cc == i].mean() < 0.9):
    #                 seg_out[cc == i] = 1  # ET → NCR

    #         # NCR
    #         NCR_img = seg_out == 1
    #         cc = cc3d.connected_components(NCR_img, connectivity=26)
    #         for i in np.unique(cc):
    #             if i == 0:
    #                 continue
    #             if (NCR_img[cc == i].size < 75) and (npy[0][cc == i].mean() < 0.9):
    #                 seg_out[cc == i] = 2  # NCR → ED

    #         # ED
    #         ED_img = seg_out == 2
    #         cc = cc3d.connected_components(ED_img, connectivity=26)
    #         for i in np.unique(cc):
    #             if i == 0:
    #                 continue
    #             if (ED_img[cc == i].size < 500) and (npy[1][cc == i].mean() < 0.9):
    #                 seg_out[cc == i] = 0  # ED → background

    #         # =================== Intensity-Based Outlier Removal for ET ===================
    #         # Use T1CE channel = 1 (FLAIR=0, T1CE=1, T1=2, T2=3)
    #         if batch["image"].ndim == 5:
    #             T1CE = batch["image"][0, 1].cpu().numpy()  # [H,W,D]
    #         else:
    #             T1CE = batch["image"][1].cpu().numpy()

    #         ET_mask = seg_out == 3
    #         if ET_mask.sum() > 0:
    #             cc_et = cc3d.connected_components(ET_mask, connectivity=26)
    #             t1ce_nonzero = T1CE[T1CE > 0]
    #             if t1ce_nonzero.size > 0:
    #                 t1ce_p99 = np.percentile(t1ce_nonzero, 99)
    #                 t1ce_threshold = 0.15 * t1ce_p99
    #                 for c in np.unique(cc_et):
    #                     if c == 0:
    #                         continue
    #                     region = (cc_et == c)
    #                     mean_intensity = T1CE[region].mean()
    #                     if mean_intensity < t1ce_threshold:
    #                         seg_out[region] = 1  # ET → NCR
    #                         print(f"ET region {c}: mean T1CE={mean_intensity:.2f} < threshold {t1ce_threshold:.2f}, relabel ET→NCR", flush=True)

    #         # =================== Hole Filling for ET ===================
    #         completeVolume = sitk.GetImageFromArray(seg_out.astype(np.uint8))
    #         closedcompleteVolume = sitk.BinaryFillhole(completeVolume, fullyConnected=True, foregroundValue=3)
    #         closedCompleteVolume = sitk.GetArrayFromImage(closedcompleteVolume)

    #         pixCount = np.count_nonzero(closedCompleteVolume != seg_out)
    #         if pixCount > 0:
    #             print("Filling holes for:", img_name, "for", pixCount, "pixels", flush=True)
    #             seg_out[closedCompleteVolume != seg_out] = 1  # holes → NCR

    #         # =================== Save as NIfTI ===================
    #         nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))

if __name__ == "__main__":
    main()
