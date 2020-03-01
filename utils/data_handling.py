from libtiff import TIFF
import logging
import os
from fnmatch import fnmatch
import torch
import pydicom
import numpy as np
import scipy.ndimage
from skimage import io
import nibabel as nib
import logging
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import glob
import cv2
import scipy
import re
from tqdm import tqdm
from utils import tools
from skimage.morphology import ball, disk, binary_closing, binary_dilation, binary_opening, binary_erosion
import scipy.misc
from skimage.measure import label   


def sorted_nicely( l ):
    """ Sorts the given iterable in the way that is expected.
 
    Required arguments:
    l -- The iterable to be sorted.
 
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def read_gt(pngPath):
    """ Load groind truth slice from a given path """    
    gt = io.imread(os.path.join(pngPath))
    
    return gt


def create_vol_dcm(paths):
    """ Create a volume from a sequence of dcm images specified in paths """

    vol = []
    for path in paths:
        vol.append(pydicom.dcmread(path).pixel_array)
        
    vol = np.array(vol)
    newVol = scipy.ndimage.zoom(vol,(5, 1, 1))
    newVol = newVol[::-1,:,:]
    
    return newVol


def load_gt_volume(paths, flip=False, plane=2):
    """ Create the gt volume from a sequence of images specified in paths """

    vol = []
    for path in paths:
        vol.append(cv2.imread(path, -1))

    vol = np.array(vol)
    
    if plane == 0: 
        vol = np.transpose(vol,[0,2,1])
        if flip:
            vol = np.flip(vol,2)
            vol = np.flip(vol,0)
    
    if plane == 1: 
        vol = np.transpose(vol,[2,0,1])
        if flip:
            vol = np.flip(vol,2)
            vol = np.flip(vol,0)
    
    if plane == 2: 
        vol = np.transpose(vol,[2,1,0])
        if flip:
            vol = np.flip(vol,0)
            vol = np.flip(vol,1)
            
    return vol


def list_files(pattern, path):
    """ List all the files that contain a specific string or pattern on a certain folder path """

    listOfFiles = list()
    for path, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern):
                listOfFiles += [os.path.join(path, name)]    

    return listOfFiles


def convert_gt2labels(volume):
    ''' The groundtruth comes with a range of values for the organs, this function 
    convert those ranges to specific values that are later used as labels '''

    volume[np.where((volume>=55)&(volume<=70))] = 1 #Liver: 63 (55<<<70)
    volume[np.where((volume>=110)&(volume<=135))] = 2 #Right kidney: 126 (110<<<135)
    volume[np.where((volume>=175)&(volume<=200))] = 3 #Left kidney: 189 (175<<<200)
    volume[np.where((volume>=240)&(volume<=255))] = 4 #Spleen: 252 (240<<<255)
    #print("Total number of voxels for liver: ", len(np.where(volume==1)[0]))
    #print("Total number of voxels for right kidney: ", len(np.where(volume==2)[0]))
    #print("Total number of voxels for left kidney: ", len(np.where(volume==3)[0]))
    #print("Total number of voxels for spleen: ", len(np.where(volume==4)[0]))
    #print("Total number of voxels: ", np.prod(volume.shape))


    return volume


def resample_numpy_volume(volume,desiredShape, label=False):
    ''' Resamples a Numpy volume to a desired shape using spline interpolation. '''
    oldShape = np.asarray(volume.shape)
    if(np.sum(desiredShape==oldShape)==3): 
        resampledVolume = volume
        if label: resampledVolume = convert_gt2labels(resampledVolume)
        return resampledVolume
    else:
        zoomFactors = desiredShape / oldShape
        if label:
            resampledVolume = scipy.ndimage.zoom(volume,zoomFactors, order=0)
            resampledVolume = convert_gt2labels(resampledVolume)

            return resampledVolume    

        else:
            resampledVolume = scipy.ndimage.zoom(volume,zoomFactors)
            
            return resampledVolume    


def get_slice_list(niftiList, gtList, validationSet=False, plane=2):
    """ Based on a list of nifti files, list all the relevant slices.
    Returns the nifti paths with its respective gt paths and slice indexes """
    
    niftiPaths = []
    gtPaths = []
    sliceIdxs = []
    
    for item in range(0,len(niftiList)):
        #Load nifti Volume
        #volume = nib.load(niftiList[item])
        
        #Load nifti annotation volume
        gt = nib.load(gtList[item])
        gtVol = gt.get_fdata()
        
        if validationSet:
            a = [i for i in range(0, gtVol.shape[plane])]

        else:
            #Ignore empty slices
            
            arr = np.sum(gtVol, axis=tuple(set([0,1,2])-set([plane])))
            z = np.where(arr==0)[0]
            slices = [i for i in range(0, gtVol.shape[plane])]
            a = [i for i in slices if i not in z]
            
            #print("For study", niftiList[item], "Number of slices ignored:", len(slices)-len(a))
            logging.info("For study {}, Number of slices ignored: {}".format(niftiList[item], len(slices)-len(a)))

        sliceIdxs += a
        niftiPaths += [niftiList[item]]*len(a)
        gtPaths += [gtList[item]]*len(a)
            
    return niftiPaths, gtPaths, sliceIdxs


def remove_slices(niftiList, gtList, slicePaths, gtPaths, plane):
    """ Based on a list of nifti files, list all the relevant slices.
    Removes the empty slices from the lists of slice paths """
    
    for item in range(0,len(niftiList)):
        #Load nifti Volume
        volume = nib.load(niftiList[item])
        indices = [i for i, a in enumerate(niftiList[item]) if a == "/"]
        study = niftiList[item][indices[-2]+1:indices[-1]]
        modality = niftiList[item][indices[-1]+1:-7]
        
        #Load nifti annotation volume
        gt = nib.load(gtList[item])
        gtVol = gt.get_fdata()

        #Ignore empty slices
        arr = np.sum(gtVol, axis=tuple(set([0,1,2])-set([plane])))
        z = np.where(arr==0)[0]
        
        imgRemove = ["/"+study+"/"+modality+"_"+ str(s)+".tiff" for s in z]
        gtRemove = ["/"+study+"/"+modality[:2]+"GT_"+ str(s)+".tiff" for s in z]
        logging.info("For study {}, Number of slices ignored: {}".format(niftiList[item], len(z)))
        slicePaths = [word for word in slicePaths if not any(bad in word for bad in imgRemove)]
        gtPaths = [word for word in gtPaths if not any(bad in word for bad in gtRemove)]
            
    return slicePaths, gtPaths


def get_slice_list_inference(niftiList, plane=2):
    """ Based on a list of nifti files, list all the relevant slices.
    Returns the nifti paths with its respective slice indexes """
    
    niftiPaths = []
    sliceIdxs = []
    
    for item in range(0,len(niftiList)):
        #Load nifti Volume
        niftiVol = nib.load(niftiList[item])
        vol = niftiVol.get_fdata()
        
        a = [i for i in range(0, vol.shape[plane])]

        sliceIdxs += a
        niftiPaths += [niftiList[item]]*len(a)
            
    return niftiPaths, sliceIdxs


def get_studies_paths(dataFolder, studies, modalities, channels=False):
    """ Based on a folder, list of studies and motalities to be included, return the list
    of paths for the nifti volumes and groundtruth volumes """
    
    niftiPaths = []
    gtPaths = []
    
    for study in studies:
        if channels:
            niftiPaths.append(os.path.join(dataFolder,str(study),"T1.nii.gz"))
            gtPaths.append(os.path.join(dataFolder,str(study),"T1GT.nii.gz"))
        else:
            for modality in modalities:
                niftiPaths.append(os.path.join(dataFolder,str(study),modality+".nii.gz"))
                if "T1" in modality: gtName = "T1GT.nii.gz"
                else: gtName = "T2GT.nii.gz"
                gtPaths.append(os.path.join(dataFolder,str(study),gtName))
        
    return niftiPaths, gtPaths


def get_slice_paths(dataFolder, studies, modalities):
    """ Based on a folder, list of studies and motalities to be included, return the list
    of individual slices and annotations """
    
    slicePaths = []
    gtPaths = []
    
    for study in studies:
        for modality in modalities:
            slices = sorted_nicely([f for f in glob.glob(dataFolder+"/"+str(study)+"/*"+modality+"_*", recursive=True)])
            slicePaths += slices
            if "T1" in modality: gtName = "T1GT"
            else: gtName = "T2GT"
            gtSlices = sorted_nicely([f for f in glob.glob(dataFolder+"/"+str(study)+"/*"+gtName+"_*", recursive=True)])
            gtPaths += gtSlices
        
    return slicePaths, gtPaths

    
def get_norm_vals(dataloader):
    """ Returns mean and standard deviation for images in a given dataloader """

    mean = 0; std = 0
    
    for i, data in enumerate(dataloader, 0):

        img = data[0][0,:,:]
        mean += img.mean()
        std += img.std()
    normalizationValues = [float(mean/(i+1)), float(std/(i+1))]
    
    return normalizationValues    


def get_name(params, kFold=False):
    """ Generate the name given to the folder where to save the results.
        Takes into account the given parameters for the run """


    folderDescription = params['architecture'] + '_' + str(params['optimizer']) + \
                '_lr' + str(str(format(params['initialLR'], ".1e"))) + \
                '_B' + str(params['batchSize'])


    folderName = os.path.join("experiments", params["runName"], folderDescription, "Plane"+str(params['plane']))
    if kFold: return folderName
    
    if not os.path.exists(folderName): os.makedirs(folderName)

    runsInFolder = len(next(os.walk(folderName))[1])
    runNumber = "run_" + str(runsInFolder+1)

    folderName = os.path.join(folderName, runNumber)

    return folderName    


def save_images(logits, paths, config):
    """ Saves individual slices as tiff or png """

    #probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()

    idx = 0
    for path in paths:
        imgOutput = probs[idx,:,:,:]
        imgOutput = np.argmax(imgOutput, axis = 0)
        
        if config["saveLogits"]:
            for label in range(0,5):
                newPathLogits = path[:-4]+"_logits"+str(label)+".tiff"
                logitsLabel = logits[idx,label,:,:]
                if not os.path.exists(os.path.dirname(newPathLogits)): os.makedirs(os.path.dirname(newPathLogits))
                writeTiff(newPathLogits, logitsLabel)

        if config["saveProbs"]:
            for label in range(0,5):
                newPathProb = path[:-4]+"_probs"+str(label)+".tiff"
                probLabel = probs[idx,label,:,:]
                if not os.path.exists(os.path.dirname(newPathProb)): os.makedirs(os.path.dirname(newPathProb))
                writeTiff(newPathProb, probLabel)

        if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
        if config["saveSegs"]: cv2.imwrite(path, imgOutput)
        idx += 1 
    

def writeTiff(path, image):
    """ Save an incomping numpy array as a tiff image """


    tiff = TIFF.open(path, mode='w')
    tiff.write_image(image)
    tiff.close()


def convert_tiff2nifti(paths, name, affine, shape, config, postProcessing = False):
    """ Reads a list of images/slices on disk and creates a nifti volume from those """

    volume = load_gt_volume(paths,flip=True, plane=config["plane"])
    resVolume = resample_numpy_volume(volume, shape, label=True)
    #print("After resample",resVolume.shape)
    if postProcessing: resVolume = post_process(resVolume)

    volNifti = nib.Nifti1Image(resVolume, affine)
    nib.save(volNifti,filename=name)


def post_process(segImg):

    oneHot = np.zeros((5,segImg.shape[0],segImg.shape[1],segImg.shape[2]))
    segm = np.zeros((segImg.shape[0],segImg.shape[1],segImg.shape[2]))
    for labelIdx in range(1,5): 
        oneHot[labelIdx,:,:,:][np.where(segImg==labelIdx)] = 1

        oneHot[labelIdx,:,:,:] = binary_closing(oneHot[labelIdx,:,:,:], ball(3))
        oneHot[labelIdx,:,:,:] = getLargestCC(oneHot[labelIdx,:,:,:])
        #oneHot[labelIdx,:,:,:] = binary_closing(oneHot[labelIdx,:,:,:], ball(5))
    for labelIdx in range(1,5): segm[np.where(oneHot[labelIdx,:,:,:]==1)] = 1*labelIdx

    return segm


def getLargestCC(segmentation):
    labels = label(segmentation,connectivity=3)
    #print(labels.max(), np.argmax(np.bincount(labels.flat)[1:])+1)
    #assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max() == 0: return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    
    return largestCC


def reconstruct_study(outputStudy, inputStudy, modality, config):
    """ After inference, reconstruct a nifti volume for specific outputs from saved images """

    inputVolume = nib.load(inputStudy)
    originalAffine = inputVolume.affine
    originalShape = inputVolume.dataobj.shape

    for label in range(0,5):

        logitLabel = "logits"+str(label)
        probLabel = "probs"+str(label)
        listOfTiff = list_files("*.tiff", os.path.join(outputStudy))

        if config["saveLogits"]:
            logitsList = [ x for x in listOfTiff if logitLabel in x ]
            logitsList = [ x for x in logitsList if modality in x ]
            logitsList = sorted_nicely(logitsList)
            logsName =  outputStudy + "/" + modality+"_"+logitLabel+".nii.gz"
            convert_tiff2nifti(logitsList, logsName, originalAffine, originalShape, config)
            [os.remove(f) for f in logitsList]

        if config["saveProbs"]:
            probsList = [ x for x in listOfTiff if probLabel in x ]
            probsList = [ x for x in probsList if modality in x ]
            probsList = sorted_nicely(probsList)
            probsName =  outputStudy + "/" + modality+"_"+probLabel+".nii.gz"
            convert_tiff2nifti(probsList, probsName, originalAffine, originalShape, config)
            [os.remove(f) for f in probsList]

    if config["saveSegs"]:
        segList = list_files("*"+modality+"*.png", os.path.join(outputStudy))
        segList = sorted_nicely(segList)
        segName = outputStudy + "/" + modality + "_segmentation.nii.gz"
        convert_tiff2nifti(segList, segName, originalAffine, originalShape, config, postProcessing=True)
        [os.remove(f) for f in segList]


def one_hot_encoding_mask(volImg, segImg):
    """
        Creates a mask containing FN, Tp and FP given a segmentation and respective annotation
    """
    #oneHot encoding
    oneHotGT = np.zeros((5,volImg.shape[0],volImg.shape[1],volImg.shape[2]))
    oneHotSeg = np.zeros((5,volImg.shape[0],volImg.shape[1],volImg.shape[2]))
    oneHotMask = np.zeros((5,volImg.shape[0],volImg.shape[1],volImg.shape[2]))

    for label in range(1,5):

        oneHotGT[label,:,:,:][np.where(volImg==label)] = 1
        oneHotSeg[label,:,:,:][np.where(segImg==label)] = 1

    oneHotMask[np.where(oneHotGT+oneHotSeg==2)]=1 # TP
    oneHotMask[np.where(oneHotGT-oneHotSeg==1)]=2 # FN
    oneHotMask[np.where(oneHotSeg-oneHotGT==1)]=3 # FP
    
    return oneHotMask


def read_oriented(niftiPath):
    """
        Reads a nifti volume and reorients it for appropiate visualization
    """
    volume = nib.load(niftiPath)
    volImg = np.transpose(volume.get_data(),[1,0,2])
    volImg = np.flip(volImg,0)
    volImg = np.flip(volImg,1)
    
    return volImg    


def load_model(model, path, mode = "train"):
    """
    Loads a model given a path
    """

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if mode == "eval":
        model.eval()
    else:
        model.train()

    return model    


def save_slices(volume, axis, name, gt=False):
    """ Save single slices to disk depending on the input volume """

    with tqdm(total=volume.shape[axis]) as (t):
        for sliceIdx in range(volume.shape[axis]):
            t.set_description(name)

            nameTiff = name+"_"+str(sliceIdx)+".tiff"
            if axis == 0: img = np.float32(volume[sliceIdx,:,:].T[::-1])
            if axis == 1: img = np.float32(volume[:,sliceIdx,:].T[::-1])
            if axis == 2: img = np.float32(volume[:,:,sliceIdx].T[::-1])
            #print(nameTiff)
            #if gt: print("GT")
            tiff = TIFF.open(nameTiff, mode='w')
            tiff.write_image(np.fliplr(img))
            tiff.close()
            

            t.update()


def load_all_planes(mainFolder, outputFolder, studies, niftiType, planes, metrics=False, segsFolder=None):
    """ Function to aggregate different planes, works with either segmentations or probabilities """

    for study in studies:
        for modality in ["T1OP_","T1IP_","T2_"]:
            for run in get_runs4study(study, mainFolder):

                logging.info('Reconstructing study: %s', os.path.join(mainFolder,"Plane"+str(0),run,"Outputs",str(study),modality[:-1]+"..."))
                inputVolume = nib.load(os.path.join(mainFolder,"Plane"+str(0),run,"Outputs",str(study),modality+"probs"+str(2)+".nii.gz"))
                originalAffine = inputVolume.affine
                originalShape = inputVolume.dataobj.shape
                
                if niftiType == "probs":

                    oneHot = np.zeros((5,originalShape[0],originalShape[1],originalShape[2]))
                    for label in [0,1,2,3,4]:
                        
                        if 0 in planes: 
                            path = os.path.join(mainFolder,"Plane"+str(0),run,"Outputs",str(study),modality+"probs"+str(label)+".nii.gz")
                            volume = nib.load(path).get_fdata()
                            oneHot[label,:,:,:] += volume

                        if 1 in planes: 
                            path = os.path.join(mainFolder,"Plane"+str(1),run,"Outputs",str(study),modality+"probs"+str(label)+".nii.gz")
                            volume = nib.load(path).get_fdata()
                            oneHot[label,:,:,:] += volume

                        if 2 in planes: 
                            path = os.path.join(mainFolder,"Plane"+str(2),run,"Outputs",str(study),modality+"probs"+str(label)+".nii.gz")
                            volume = nib.load(path).get_fdata()
                            oneHot[label,:,:,:] += volume
                    
                    oneHot /= len(planes)
                    segm = np.argmax(oneHot, axis = 0)

                if niftiType == "segmentation":

                    oneHot = np.zeros((3, 5,originalShape[0],originalShape[1],originalShape[2]))
                    segm = np.zeros((originalShape[0],originalShape[1],originalShape[2]))
                    
                    if 0 in planes: 
                        path = os.path.join(mainFolder,"Plane"+str(0),run,"Outputs",str(study),modality+"segmentation.nii.gz")
                        volImg = nib.load(path).get_fdata()
                        for label in range(1,5):
                            oneHot[0, label,:,:,:][np.where(volImg==label)] = 1
                            oneHot[0, label,:,:,:] = binary_closing(oneHot[0, label,:,:,:], ball(3))

                    if 1 in planes: 
                        path = os.path.join(mainFolder,"Plane"+str(1),run,"Outputs",str(study),modality+"segmentation.nii.gz")
                        volImg = nib.load(path).get_fdata()
                        for label in range(1,5): 
                            oneHot[1, label,:,:,:][np.where(volImg==label)] = 1
                            oneHot[1, label,:,:,:] = binary_closing(oneHot[1, label,:,:,:], ball(3))

                    if 2 in planes: 
                        path = os.path.join(mainFolder,"Plane"+str(2),run,"Outputs",str(study),modality+"segmentation.nii.gz")
                        volImg = nib.load(path).get_fdata()
                        for label in range(1,5):
                            oneHot[2, label,:,:,:][np.where(volImg==label)] = 1
                            oneHot[2, label,:,:,:] = binary_closing(oneHot[2, label,:,:,:], ball(3))
                    
                    oneHotSum = np.sum(oneHot, axis = 0)
                    oneHotSum[np.where(oneHotSum<2)] = 0
                    oneHotSum[np.where(oneHotSum!=0)] = 1

                    for label in range(1,5): segm[np.where(oneHotSum[label,:,:,:]==1)] = 1*label

                isoVolNifti = nib.Nifti1Image(segm.astype(np.int16), originalAffine)
                path = os.path.join(outputFolder,run,str(study))
                if not os.path.exists(path): os.makedirs(path)
                nib.save(isoVolNifti,path+"/"+modality+"segmentation.nii.gz")


                if metrics:
                    gtImg = read_oriented(segsFolder+"/"+str(study)+"/"+modality[:2]+"GT.nii.gz")
                    segImg = read_oriented(path+"/"+modality+"segmentation.nii.gz")
                    metrics = tools.get_metrics(gtImg, segImg, originalAffine)
                    logging.info('Metrics for study: %s', metrics)
                    logging.info("####################################################")            


def load_probs(mainFolder, outputFolder, studies, niftiType, planes, metrics=False, segsFolder=None):
    """ Function to aggregate probabilities for T1 modalities """

    
    for study in studies:
        for run in get_runs4study(study, mainFolder):
            logging.info('Reconstructing study: %s', os.path.join(mainFolder,"Plane"+str(planes[0]),run,"Outputs",str(study),"T1..."))
            inputVolume = nib.load(os.path.join(mainFolder,"Plane"+str(planes[0]),run,"Outputs",str(study),"T1OP_"+"probs"+str(2)+".nii.gz"))
            originalAffine = inputVolume.affine
            originalShape = inputVolume.dataobj.shape
            
            oneHot = np.zeros((5,originalShape[0],originalShape[1],originalShape[2]))

            for modality in ["T1OP_","T1IP_"]:
            
                for label in [0,1,2,3,4]:
                    
                    if 0 in planes: 
                        path = os.path.join(mainFolder,"Plane"+str(0),run,"Outputs",str(study),modality+"probs"+str(label)+".nii.gz")
                        volume = nib.load(path).get_fdata()
                        oneHot[label,:,:,:] += volume

                    if 1 in planes: 
                        path = os.path.join(mainFolder,"Plane"+str(1),run,"Outputs",str(study),modality+"probs"+str(label)+".nii.gz")
                        volume = nib.load(path).get_fdata()
                        oneHot[label,:,:,:] += volume

                    if 2 in planes: 
                        path = os.path.join(mainFolder,"Plane"+str(2),run,"Outputs",str(study),modality+"probs"+str(label)+".nii.gz")
                        volume = nib.load(path).get_fdata()
                        oneHot[label,:,:,:] += volume
                
            oneHot /= len(planes)
            segm = np.argmax(oneHot, axis = 0)

            isoVolNifti = nib.Nifti1Image(segm.astype(np.int16), originalAffine)
            path = os.path.join(outputFolder,run,str(study))
            if not os.path.exists(path): os.makedirs(path)
            nib.save(isoVolNifti,path+"/"+modality[:2]+"_segmentation.nii.gz")

            if metrics:
                gtImg = read_oriented(segsFolder+"/"+str(study)+"/"+modality[:2]+"GT.nii.gz")
                segImg = read_oriented(path+"/"+modality[:2]+"_segmentation.nii.gz")
                metrics = tools.get_metrics(gtImg, segImg)
                logging.info('Metrics for study: %s', metrics)
                logging.info("####################################################")            


def get_runs4study(study, mainFolder):
    """ Function to identify the run folders for a given study """

    runs = []

    allData = [f for f in glob.glob(mainFolder + "**/Outputs/"+str(study)+"/", recursive=True)]
    for idx in range(0,2):
        start = allData[idx].find("run_")
        end = allData[idx][start:].find("/")
        run = allData[idx][start:start+end]
        runs.append(run)
    return runs


def save_png(volume, path, task3=False):
    """ Given an input volume, convert the segmentation values and save individual slices to png files"""
    volume.astype(int)
    if task3:
        volume[np.where(volume==1)] = 63 #Liver: 63 (55<<<70)
        volume[np.where(volume!=63)] = 0 #Left kidney: 189 (175<<<200)
    else:
        volume[np.where(volume==1)] = 63 #Liver: 63 (55<<<70)
        volume[np.where(volume==2)] = 126 #Right kidney: 126 (110<<<135)
        volume[np.where(volume==3)] = 189 #Left kidney: 189 (175<<<200)
        volume[np.where(volume==4)] = 252 #Spleen: 252 (240<<<255)

    if "T1" in path: 
        path = path[:-2]+"DUAL/Results/"
    if "T2" in path: 
        path = path+"SPIR/Results/"
    
    for sliceIdx in range(0,volume.shape[2]):
        pngPath = path+"img"+str(sliceIdx).zfill(3)+".png"
        #scipy.misc.imsave(pngPath, volume[:,:,sliceIdx])
        cv2.imwrite(pngPath, volume[:,:,sliceIdx])