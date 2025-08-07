import json
import math
import os

import numpy as np
import torch

from monai import data, transforms

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_tensor import MetaObj, MetaTensor
from monai.data.utils import no_collation
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.traits import MultiSampleTrait, RandomizableTrait
from monai.transforms.transform import MapTransform, Randomizable, RandomizableTransform
from monai.transforms.utility.array import (
    AddCoordinateChannels,
    AddExtremePointsChannel,
    AsChannelLast,
    CastToType,
    ClassesToIndices,
    ConvertToMultiChannelBasedOnBratsClasses,
    CuCIM,
    DataStats,
    EnsureChannelFirst,
    EnsureType,
    FgBgToIndices,
    Identity,
    ImageFilter,
    IntensityStats,
    LabelToMask,
    Lambda,
    MapLabelValue,
    RemoveRepeatedChannel,
    RepeatChannel,
    SimulateDelay,
    SplitDim,
    SqueezeDim,
    ToCupy,
    ToDevice,
    ToNumpy,
    ToPIL,
    TorchVision,
    ToTensor,
    Transpose,
)
from monai.transforms.utils import extreme_points_to_image, get_extreme_points
from monai.transforms.utils_pytorch_numpy_unification import concatenate
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix, TraceKeys, TransformBackends
from monai.utils.type_conversion import convert_to_dst_type

from collections.abc import Mapping, Sequence
from collections.abc import Callable, Hashable, Mapping

# class ConvertToMultiChannelBasedOnBratsClasses(Transform):
#     """
#     Convert labels to multi channels based on brats18 classes:
#     label 1 is the necrotic and non-enhancing tumor core
#     label 2 is the peritumoral edema
#     label 3 is the GD-enhancing tumor
#     The possible classes are TC (Tumor core), WT (Whole tumor)
#     and ET (Enhancing tumor).
#     """

#     backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

#     def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
#         # if img has channel dim, squeeze it
#         if img.ndim == 4 and img.shape[0] == 1:
#             img = img.squeeze(0)

#         result = [(img == 1) | (img == 3), (img == 1) | (img == 3) | (img == 2), img == 3]
#         # merge labels 1 (tumor non-enh) and 3 (tumor enh) and 2 (large edema) to WT
#         # label 3 is ET
#         return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)
# class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
#     """
#     Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
#     Convert labels to multi channels based on brats18 classes:
#     label 1 is the necrotic and non-enhancing tumor core
#     label 2 is the peritumoral edema
#     label 4 is the GD-enhancing tumor
#     The possible classes are TC (Tumor core), WT (Whole tumor)
#     and ET (Enhancing tumor).
#     """

#     backend = ConvertToMultiChannelBasedOnBratsClasses.backend

#     def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
#         super().__init__(keys, allow_missing_keys)
#         self.converter = ConvertToMultiChannelBasedOnBratsClasses()

#     def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
#         d = dict(data)
#         for key in self.key_iterator(d):
#             d[key] = self.converter(d[key])
#         return d

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



def datafold_read(datalist, basedir, fold=0, key="training"):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key] #Get the data with the specified key

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list): #Images
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str): #Label
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold: #Select the specified fold as validation and the remaining as the training set.
            val.append(d)
        else:
            tr.append(d)

    return tr, val



def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),

            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[args.roi_x, args.roi_y, args.roi_z],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

            # --- Histogram transforms ---
            transforms.RandHistogramShiftd(keys="image", num_control_points=30, prob=1.0),
            #transforms.HistogramNormalized(keys="image", num_bins=20, min=0, max=None),  # max will be auto-calculated

            # --- Intensity transforms ---
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
            transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
            transforms.RandAdjustContrastd(keys="image", prob=0.2),
            transforms.RandGaussianNoised(keys="image", prob=0.2),
            transforms.RandGaussianSmoothd(keys="image", prob=0.2),

            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode: #Just use the validation data and create a loader.

        test_ds = data.Dataset(data=validation_files, transform=test_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=test_sampler, pin_memory=True
        )

        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)

        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        
        loader = [train_loader, val_loader]

    return loader
