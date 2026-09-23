# GLIMS-MedNeXt

**An ensemble framework for brain MRI segmentation in Sub-Saharan Africa**

[Paper](https://doi.org/10.1007/978-3-032-16365-3_24) · [Source code](https://github.com/AliAZ98/GLIMS-MedNeXt)

GLIMS-MedNeXt combines the GLIMS and MedNeXt architectures to segment gliomas in multimodal MRI. The work addresses the shift from higher-quality BraTS-GLI images to BraTS-SSA images acquired in Sub-Saharan Africa. Models pretrained on BraTS-GLI are adapted to BraTS-SSA and combined with a learned fusion head. This is the research code associated with our BraTS 2025 Lighthouse submission and [published paper](https://doi.org/10.1007/978-3-032-16365-3_24).

> **Reproducibility status:** The source is public, but data and trained checkpoints are not included. Training and inference scripts contain local absolute paths that must be changed before use. The examples below describe expected inputs; they are not turnkey commands on a fresh clone.

## Method

Both models take four co-registered MRI volumes and predict three overlapping BraTS regions. GLIMS produces four deep-supervision outputs; MedNeXt-B produces five. The extra lowest-resolution MedNeXt output is omitted when matching scales. The selected training implementation weights corresponding outputs, concatenates them, and applies a small 3D convolutional fusion block. An averaging alternative is also present. Validation uses MONAI sliding-window inference with a 96 × 96 × 96 voxel region of interest by default.

| Channel | Region | Meaning |
| --- | --- | --- |
| 0 | TC | Tumor core |
| 1 | WT | Whole tumor |
| 2 | ET | Enhancing tumor |

The inference script converts these overlapping regions into a single-label NIfTI mask: `0` background, `1` non-enhancing/necrotic tumor core, `2` edema, and `3` enhancing tumor.

## Data

- **BraTS-GLI:** higher-quality source-domain glioma MRI for pretraining.
- **BraTS-SSA:** target-domain MRI for adaptation and evaluation.
- **Inference modality order:** T2-FLAIR (`t2f`), contrast-enhanced T1 (`t1c`), native T1 (`t1n`), and T2-weighted (`t2w`).

Obtain the datasets under the applicable BraTS access conditions. The training loader expects a JSON file containing a `training` array. Each case needs an `image` list of four paths, a `label` path, and a numeric `fold`; paths are resolved relative to `--data_dir`. For example:

```json
{
  "training": [
    {
      "image": [
        "case-001/case-001-t2f.nii.gz",
        "case-001/case-001-t1c.nii.gz",
        "case-001/case-001-t1n.nii.gz",
        "case-001/case-001-t2w.nii.gz"
      ],
      "label": "case-001/case-001-seg.nii.gz",
      "fold": 0
    }
  ]
}
```

Include all cases in the actual JSON. `--fold` selects validation cases; other folds are used for training. The inference loader instead scans immediate subdirectories of `--data_dir` and expects the four modality files shown above in each case directory.

## Setup

Use a CUDA-capable GPU and a Python environment compatible with your PyTorch/CUDA installation:

```bash
git clone https://github.com/AliAZ98/GLIMS-MedNeXt.git
cd GLIMS-MedNeXt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a broad, unpinned environment export, not a verified lockfile. Install a PyTorch build suited to your CUDA runtime and resolve version conflicts in your environment. Key packages used by these scripts include `torch`, `monai`, `wandb`, `nibabel`, `SimpleITK`, `scipy`, and `connected-components-3d` (`cc3d`).

Before training, replace the two absolute checkpoint paths inside `EnsembleFusion.__init__` in `main.py` with your GLIMS and MedNeXt checkpoints. The `--GLIMSweights` and `--MedNeXtweights` flags are used by the alternative `EnsembleAvg` class and **do not override** those fusion paths. Remove or change the unconditional `wandb.login(key="WANDB_KEY")` call to use your own environment-based login. Check that model and checkpoint architectures match.

Before inference, replace the absolute paths for the two ensemble checkpoints in `test_BraTS.py`. Its fusion constructors also refer to local backbone checkpoint paths. The exposed `--model_ensemble_1` and `--model_ensemble_2` flags **do not select** checkpoints in the active inference path. `Checkpoints/` and `Outputs/` currently contain placeholders only.

## Training

After configuring the data and paths above:

```bash
python main.py \
  --data_dir /path/to/brats-ssa-training \
  --json_list /path/to/folds.json \
  --fold 0 \
  --output_dir /path/to/output
```

Defaults include 96³ crops, batch size 2, AdamW, 1,000 epochs, validation every 10 epochs, and 0.8 sliding-window overlap. `trainer.py` saves the best checkpoint as `model_.pt` and the final checkpoint as `model_final.pt` under `--output_dir` when saving is enabled. Training requires CUDA and assumes GPU 0 unless `--gpu` is supplied.

## Inference

After configuring the checkpoint paths, arrange unlabeled cases like this:

```text
/path/to/validation/
└── case-001/
    ├── case-001-t2f.nii.gz
    ├── case-001-t1c.nii.gz
    ├── case-001-t1n.nii.gz
    └── case-001-t2w.nii.gz
```

```bash
python test_BraTS.py \
  --data_dir /path/to/validation \
  --output_dir /path/to/predictions \
  --exp_name fusion
```

The active inference path averages the outputs of two loaded fusion checkpoints. It applies sigmoid thresholds of 0.65 (TC), 0.55 (WT), and 0.60 (ET), followed by class-priority assignment, connected-component filtering, and hole filling. It writes one NIfTI segmentation per case to `<output_dir>/<exp_name>/`. The script requires CUDA. Other post-processing variants in the file are commented out.

## Results and citation

The paper's comparison table reports the following Dice scores for the fusion approach on BraTS-SSA:

| WT | TC | ET | Mean Dice |
| ---: | ---: | ---: | ---: |
| 93.52% | 85.51% | 85.61% | **88.73%** |

These are reported comparison results, not a challenge leaderboard rank. See the [paper](https://doi.org/10.1007/978-3-032-16365-3_24) for its evaluation protocol and complete experiments.

```bibtex
@inproceedings{azmoudeh2026glimsmednext,
  title     = {GLIMS-MedNeXt: An Ensemble Framework for Brain MRI Segmentation in Sub-Saharan Africa},
  author    = {Azmoudeh, Ali and Öksüz, İlkay and Ekenel, Hazım Kemal},
  booktitle = {Segmentation, Classification, and Synthesis for Brain Tumors and Traumatic Brain Injuries: MICCAI 2025 Challenges},
  series    = {Lecture Notes in Computer Science},
  volume    = {16376},
  pages     = {262--273},
  year      = {2026},
  publisher = {Springer},
  doi       = {10.1007/978-3-032-16365-3_24}
}
```

**Authors:** Ali Azmoudeh, İlkay Öksüz, Hazım Kemal Ekenel. The repository is distributed under the [MIT license](LICENSE). Dataset access and third-party components remain subject to their respective terms.
