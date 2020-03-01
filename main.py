import torch
import os, sys
import random
import numpy as np
import logging
from utils import tools
from pipeline import run_session
from utils import data_handling as dh


config = {}
# List configuration for run
config = {}
config["description"] = "Model 1, DICE + BCE Loss"
config["runName"] = "Test"

# Setting parameters 
config["architecture"] = "UNet768"
config["optimizer"] = "Adam"
config["initialLR"] = 1e-3
config["batchSize"] = 2
config["numEpochs"] = 1
config["imgSize"] = 96
config["plane"] = 2
config["skip"] = 6
config["heads"] = False

# Setting data folders and output folders
config["trainingStudies"] = [1]#train#[1,2,3,5,8,10,13,15,19,20,21,22,31,32,33,34]
config["validationStudies"] = [1]#val#[36,37,38,39]
config["testStudies"] = [1]#test#[36,37,38,39]
config["niftiFolder"] = "data/nifti"
config["dataFolder"] = "data/nifti"
config["modalities"] = ["T1OP"]
config["saveLogits"] = False
config["saveProbs"] = False
config["saveSegs"] = True
config["mainDir"] = dh.get_name(config)
config["modelDir"] = dh.get_name(config)
config["segsDir"] = os.path.join(config["modelDir"], "Outputs")

# Setting data augmentations for training
config["rotate"] = True
config["crop"] = True
config["normalize"] = False

# Create model folder and set the logger
if not os.path.exists(config["modelDir"]): os.makedirs(config["modelDir"])
tools.set_logger(os.path.join(config["mainDir"],'train.log'))
logging.info("####################################################")
logging.info('Parameters for run: %s', "{" + "\n".join("{}: {}".format(k, v) for k, v in config.items()) + "}")
run_session(config)
torch.cuda.empty_cache()
logging.info("Finished")
logging.info("####################################################")
