import torch
from torch.nn import functional as F
from torch.autograd import Variable
from tqdm import tqdm
from models import segmenter
from utils import data_handling as dh
from utils import data_loader as dl
from utils import tools
import nibabel as nib
import logging
from time import sleep
import os


def run_session(config):

    # Load training data
    volPaths, gtPaths = dh.get_studies_paths(config["niftiFolder"], config["trainingStudies"], config["modalities"], channels=False)
    niftiPaths, gtPaths, sliceIdxs = dh.get_slice_list(volPaths, gtPaths, plane=config["plane"])

    if config["skip"] != None: 
        niftiPaths, gtPaths, sliceIdxs = niftiPaths[::config["skip"]], gtPaths[::config["skip"]], sliceIdxs[::config["skip"]]
    logging.info("Using {} slices for training".format(len(niftiPaths)))

    trainSet = dl.train(niftiPaths, gtPaths, sliceIdxs, config)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=config["batchSize"], shuffle=True, num_workers=0)
    logging.info("Training data loaded")

    # Load validation data
    volPathsVal, gtPathsVal = dh.get_studies_paths(config["dataFolder"], config["validationStudies"], config["modalities"], channels=False)
    niftiPaths, gtPaths, sliceIdxs = dh.get_slice_list(volPathsVal, gtPathsVal, plane=config["plane"], validationSet=True)
    if config["skip"] != None: 
        niftiPaths, gtPaths, sliceIdxs = niftiPaths[::config["skip"]], gtPaths[::config["skip"]], sliceIdxs[::config["skip"]]
    logging.info("Using {} slices for validation".format(len(niftiPaths)))

    valSet = dl.validate(niftiPaths, gtPaths, sliceIdxs, config)
    valLoader = torch.utils.data.DataLoader(valSet, batch_size=config["batchSize"], shuffle=True, num_workers=0)
    logging.info("Validation data loaded")

    logging.info("####################################################################")
    model = tools.select_architecture(config["architecture"])
    model = model().cuda() # Move to gpu
    optimizer = segmenter.set_optimizer(model, config)

    logging.info("Starting training for {} epoch(s)".format(config["numEpochs"]))
    diceMetrics = run_train_val_session(model, trainLoader, valLoader, optimizer, config)
    logging.info("Training procedure finished")

    logging.info("Validation Dice scores:")
    logging.info("Liver: {:05.4f}".format(diceMetrics[1]))
    logging.info("KidneyR: {:05.4f}".format(diceMetrics[2]))
    logging.info("KidneyL:{:05.3f}".format(diceMetrics[3]))
    logging.info("Spleen:{:05.3f}".format(diceMetrics[4]))


def run_train_val_session(model, trainData, valData, optimizer, config):
    """ Performs training and validation using a learning raate scheduler. """

    bestValLoss = 1
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=8, verbose=True)
    with tqdm(total=config['numEpochs']) as (t):
        for epoch in range(config['numEpochs']):
            t.set_description('Epoch %i' % epoch)
            
            trainLoss, diceTrain = train(model, optimizer, trainData, config)
            valLoss, diceVal = evaluate(model, valData, config, epoch)
            
            scheduler.step(valLoss, epoch)
            tools.tensorboard_logger(config['modelDir'], epoch=epoch,
                trainLoss=trainLoss, valLoss=valLoss,
                trainDiceLiver=diceTrain[1], valDiceLiver=diceVal[1],
                trainDiceKidneyR=diceTrain[2], valDiceKidneyR=diceVal[2],
                trainDiceKidneyL=diceTrain[3], valDiceKidneyL=diceVal[3],
                trainDiceSpleen=diceTrain[4], valDiceSpleen=diceVal[4]
                )
            
            logging.info("For epoch {}:".format(epoch))
            logging.info("Training loss: {:05.4f}, Validation loss: {:05.4f}.".format(trainLoss, valLoss))
            logging.info("Dice scores are: Liver: {:05.4f}, KidneyR:{:05.3f}, KidneyL: {:05.3f}, Spleen: {:05.3f}, ".format(diceVal[1], diceVal[2], diceVal[3], diceVal[4]))
            t.update()

            if bestValLoss > valLoss:
                epochSaved = epoch
                bestValLoss = valLoss
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'valLoss': valLoss, 'trainLoss': trainLoss}, 
                            os.path.join(config['modelDir'],'model.pt'))
                
    logging.info("Model saved in epoch {}".format(epochSaved))                
    return diceVal


def train(model, optimizer, dataLoader, params):
    """ 
        Computes a training step for one epoch (one full pass over the training set). 
    """
    model.train()
    lossValue = tools.RunningAverage()
    diceValue = tools.RunningAverage()
    with tqdm(total=len(dataLoader)) as (t):
        t.set_description('Training')
        for i, (trainBatch, labelBatch) in enumerate(dataLoader):
            trainBatch, labelBatch = trainBatch.cuda(async=True), labelBatch.cuda(async=True)
            trainBatch, labelBatch = Variable(trainBatch), Variable(labelBatch)
            outputBatch = model(trainBatch)
            
            loss, dice = segmenter.dice_loss(labelBatch.long(), outputBatch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossValue.update(loss.item())
            diceValue.update(dice)
            t.set_postfix(loss=('{:05.3f}').format(lossValue()))
            t.update()

    return lossValue.avg, diceValue()


def evaluate(model, dataLoader, params, epoch):
    """ 
        Computes a validation step for one epoch (one full pass over the validation set). 
    """
    model.eval()
    lossValue = tools.RunningAverage()
    diceValue = tools.RunningAverage()
    with tqdm(total=len(dataLoader)) as (t):
        t.set_description('Validation')
        for i, (trainBatch, labelBatch, _) in enumerate(dataLoader):
            trainBatch, labelBatch = trainBatch.cuda(async=True), labelBatch.cuda(async=True)
            trainBatch, labelBatch = Variable(trainBatch), Variable(labelBatch)
            outputBatch = model(trainBatch)
            loss, dice = segmenter.dice_loss(labelBatch.long(), outputBatch)
            
            lossValue.update(loss.item())
            diceValue.update(dice)

            #if epoch == (params["numEpochs"]-1):
                #logging.info("Last epoch reached")
                #dh.save_images(outputBatch, savePath, saveProbs=params["saveProbs"], saveLogits=params["saveLogits"])

            t.set_postfix(loss=('{:05.3f}').format(lossValue()))
            t.update()

    #metrics = hf.get_metrics(0, probs, labels, params, train=True)
    return lossValue.avg, diceValue()


def inference(model, dataLoader, params):
    """ Computes a training step for one epoch (one full pass over the training set). 
        
        Takes into account a single input for the network
    """
    model.eval()
    with tqdm(total=len(dataLoader)) as (t):
        t.set_description('Test')
        for i, (trainBatch, savePath) in enumerate(dataLoader):
            trainBatch = trainBatch.cuda(async=True)
            trainBatch = Variable(trainBatch)
            outputBatch = model(trainBatch)
            dh.save_images(outputBatch, savePath, params)
            del trainBatch, outputBatch
            t.update()
