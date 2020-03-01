import logging
import os
from fnmatch import fnmatch
import dcmstack
import pydicom
import numpy as np
import scipy.ndimage
from skimage import io
import nibabel as nib
import logging
from tensorboardX import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from models import unet
from scipy.ndimage import morphology


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
        self.avg = self.total/float(self.steps)
    
    def __call__(self):
        return self.total/float(self.steps)    


def select_architecture(model):
    """
    Returns a model based on a specified architecture
    """
    if model == "UNetOriginal":
        logging.info('Using UNet Original as architecture')
        return unet.UNet1024

    if model == "UNet768":
        logging.info('Using UNet768 as architecture')
        return unet.UNet768        


def set_logger(log_path):
    """Set the logger to log info in terminal and file`log_path.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)        


def tensorboard_logger(modelDir, *args, **kwargs):
    """ Callback intended to be executed at each epoch
        of the training which goal is to add valuable
        information to the tensorboard logs such as the losses
        and accuracies
    Args:
        description (str): Description used for folder path
    """
    epoch = kwargs['epoch']+1
    tenosrboardFolder = os.path.join(modelDir, "Tensorboard")
    
    # Write files in folder for TensorBoardX for training
    if(os.path.isdir(tenosrboardFolder)==False):
        os.makedirs(tenosrboardFolder)
    writer = SummaryWriter(tenosrboardFolder) # this is a folder
    if(epoch > 0):
        writer.add_scalars('Loss', {'Training': kwargs['trainLoss'],
                                    'Validation': kwargs['valLoss']},
                                    epoch)
        writer.add_scalars('Liver', {'Training': kwargs['trainDiceLiver'],
                                        'Validation': kwargs['valDiceLiver']},
                                    epoch)
        writer.add_scalars('KidneyR', {'Training': kwargs['trainDiceKidneyR'],
                                        'Validation': kwargs['valDiceKidneyR']},
                                    epoch)
        writer.add_scalars('KidneyL', {'Training': kwargs['trainDiceKidneyL'],
                                        'Validation': kwargs['valDiceKidneyL']},
                                    epoch)
        writer.add_scalars('Spleen', {'Training': kwargs['trainDiceSpleen'],
                                        'Validation': kwargs['valDiceSpleen']},
                                    epoch)
    else:
        writer.add_scalar('Testing/Precision', kwargs['aggregated_test_PnR']['precision'], epoch_id)
        writer.add_scalar('Testing/Recall', kwargs['aggregated_test_PnR']['recall'], epoch_id)
        writer.add_scalar('Testing/DiscreteDICE', kwargs['aggregated_test_PnR']['discreteDICE'], epoch_id)
    writer.close()


def dice(gt, seg):
    """
        Estmates the dice coefficient based on two input images, works for 2D/3D
    """
    
    intersection = np.logical_and(gt, seg)
    dice = (2. * intersection.sum() / (gt.sum() + seg.sum()))
    
    return dice    


def get_metrics(volImg, segImg, affine):

    oneHotGT = np.zeros((5,volImg.shape[0],volImg.shape[1],volImg.shape[2]))
    oneHotSeg = np.zeros((5,volImg.shape[0],volImg.shape[1],volImg.shape[2]))
    oneHotMask = np.zeros((5,volImg.shape[0],volImg.shape[1],volImg.shape[2]))

    metrics = {}
    for label in range(1,5):
        oneHotGT[label,:,:,:][np.where(volImg==label)] = 1
        oneHotSeg[label,:,:,:][np.where(segImg==label)] = 1

        metrics[label] = {}
        metrics[label]["DICE"] = 0#dice(oneHotGT[label,:,:,:], oneHotSeg[label,:,:,:])
        metrics[label]["RAVD"] = 0#RAVD(oneHotGT[label,:,:,:], oneHotSeg[label,:,:,:])
        #surfaceDistance = surfd(oneHotGT[label,:,:,:], oneHotSeg[label,:,:,:], affine,1)
        metrics[label]["ASSD"] = 0#surfaceDistance.mean()
        metrics[label]["MSSD"] = 0# surfaceDistance.max()

    return metrics


def RAVD(Vref,Vseg):
    """ Estimates the Relative Absolute Volume Difference """

    ravd=(abs(Vref.sum() - Vseg.sum())/Vref.sum())*100

    return ravd


def surfd(input1, input2, sampling=1, connectivity=1):
    """ Estimates the symmetric surface distance """
    
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
    
    # Extract the edges
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
    S = input_1.astype(np.float32) - morphology.binary_erosion(input_1.astype(np.float32), conn)
    Sprime = input_2.astype(np.float32) - morphology.binary_erosion(input_2.astype(np.float32), conn)

    # Use the pixel size to estimate distance
    dta = morphology.distance_transform_edt(~(S.astype("bool")),sampling)
    dtb = morphology.distance_transform_edt(~(Sprime.astype("bool")),sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
    
    return sds    