from utils import data_handling as dh
from utils import data_loader as dl
from pipeline import inference
from utils import tools
from models import unet
from tqdm import tqdm
import nibabel as nib
import torch
import os, sys
import random
import numpy as np
import logging


# List configuration variables
config = {}
config["volumePaths"] = ["data/nifti/1/T1OP.nii.gz",]
# Paths of the annotations, "None" if annotations are not available or if you don't want metrics after inference
config["gtPaths"] = None#["data/Mice_Sets/nifti/H2030IC10dn573/GT.nii.gz",]
# Folder where to save the segmentations and the log file
config["outputFolder"] = "experiments/inferenceTest/"
# Path of the model to load
config["architecture"] = "UNet768"
config["modelDir"] = "experiments/Test/UNet768_Adam_lr1.0e-03_B2/Plane2/run_2/model.pt"
config["imgSize"] = 96
config["batchSize"] = 1
config["plane"] = 2
# Options to save certain type of data
config["saveSegs"] = True
config["saveProbs"] = False
config["saveLogits"] = False


# Create model folder and set the logger
if not os.path.exists(config["outputFolder"]): os.makedirs(config["outputFolder"])
tools.set_logger(os.path.join(config["outputFolder"],'inference.log'))

logging.info("Inference run")
logging.info('Parameters for run: %s', "{" + "\n".join("{}: {}".format(k, v) for k, v in config.items()) + "}")

# Load data
niftiPaths, sliceIdxs = dh.get_slice_list_inference(config["volumePaths"])
testSet = dl.inference(niftiPaths=niftiPaths, sliceIdxs=sliceIdxs, config=config)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=config["batchSize"], shuffle=False, num_workers=0)
logging.info("Data loaded")

# Set and load the model
model = tools.select_architecture(config["architecture"])
model = model().cuda() # Move to gpu
model = dh.load_model(model, path=config["modelDir"])

logging.info("Starting inference")
inference(model, testLoader, config)
del model, testLoader

logging.info("Proceding with volume reconstruction")
for studyIdx in range(0,len(config["volumePaths"])):

  inputStudyPath = config["volumePaths"][studyIdx]
  numberIdx = [(i,c) for i,c in enumerate(inputStudyPath) if c=="/"]
  study = inputStudyPath[numberIdx[-2][0]+1:numberIdx[-1][0]]
  modality = inputStudyPath[numberIdx[-1][0]+1:-7]
  outStudyPath = os.path.join(config["outputFolder"], str(study))
  
  logging.info("Reconstructing %s...", outStudyPath+"/"+modality)
  dh.reconstruct_study(outStudyPath, inputStudyPath, modality, config)
  segImg = dh.read_oriented(outStudyPath+"/"+modality+"_segmentation.nii.gz")

  # If annotations available, obtain metrics
  if config["gtPaths"] != None:

      gtImg = dh.read_oriented(config["gtPaths"][studyIdx]) # Read annotations
      # Read nifti file to obtain original affine for evaluation
      inputVolume = nib.load(config["gtPaths"][studyIdx]); originalAffine = inputVolume.affine

      metrics = tools.get_metrics(gtImg, segImg, originalAffine.diagonal()[:-1])
      logging.info('Metrics for study: %s', metrics)

logging.info("Finished")