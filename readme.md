# Organ Segmentation Using Deep Learning
PyTorch implementation for organ segmentation. For this, MR data from the [CHAOS challenge](https://chaos.grand-challenge.org/) is considered for segmenting Liver, Spleen and Kidneys.

## Getting Started
### Dataset Organization
CHAO MR data should be converted to nifti files in the format ```volume.nii.gz``` and then organized in the following scheme:
```
Main
├── data
│   ├── 1
|   |   ├── T1GT.nii.gz
|   |   ├── T2GT.nii.gz
|   |   ├── T1IP.nii.gz
|   |   ├── T1OP.nii.gz
|   |   └── T2.nii.gz
│   ├── 2
|   |   ├── T1GT.nii.gz
|   |   ├── T2GT.nii.gz
|   |   ├── T1IP.nii.gz
|   |   ├── T1OP.nii.gz
.   .   └── T2.nii.gz
:   :
```

### Train a model
To train a model, the script ```main.py``` should be used. The script has a dictionary of parameters that could be modified, this includes architecture to use, selected plane (frontal, sagittal, coronal), number of epochs, batch size and folder list.
During training, all the metrics are saved to the tensorboard and also to a log file iin the defined folder for the run. The final models is also saved in the folder.

### Inference with new data
After training a model, the script ```inference.py``` can be used to test new data. Different parameters can be set for inference: path to the model, type of output to save (raw output, probabilities, segmentation), volumes to segment and respective groundtruth if available.
Inference is performed in all the slices and its outputs are saved as individual images. After inference is completed, volumes are reconstructed and then compared to the annotations if available.
