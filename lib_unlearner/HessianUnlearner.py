import argparse
import random
import copy
import numpy as np
import pandas as pd
import os
import torch
import logging
from time import time
from torch.nn.utils import parameters_to_vector as p2v
from lib_trainer.Trainer import Trainer
from lib_unlearner.hypercolumn import HyperC, ActivationsHook, NLP_ActivationsHook
from lib_unlearner.grad_utils import getGradObjs, gradNorm, getHessian, getVectorizedGrad, getOldPandG
from lib_unlearner.torch_foci import foci
from utils import ORIGINAL_PROPORTION

class HessianUnlearner(Trainer):
    def __init__(self, dataset_name, dataset_path, 
                 model_name, model_save_path, device):
        Trainer.__init__(self, dataset_name, dataset_path, model_name, model_save_path, device)
        # Logging
        self.logger = logging.getLogger('unlearner_log')
    
    def unlearn_exp(self, origin_model_path = None, unlearn_prop = 0.01, selectionType = "Full", unlearning_data_selection = "Random", 
                    order = "Hessian", approxType = "FD" , save_unlearned_model = False, remove_class = -1, l2lambda = 0.01):
        self.logger.info("======= unlearn_prop %.3f =======" % (unlearn_prop))
        self.logger.info("l2lambda %.3f" % (l2lambda))
        self.load_data(unlearning=1, unlearning_data_selection = unlearning_data_selection,
                          unlearning_proportion = unlearn_prop, original_proportion=ORIGINAL_PROPORTION,
                          left=False, training_batch_size = 64, sisa_selection_op = -1,
                          remove_class=remove_class)
        self.initialize_model(shadow_model = False, model_load_path = origin_model_path, 
                              pretrained = False, freeze_conv_layer=False)
        self.prepare_train_setting(learning_rate = 0.001,  epochs = False, optim = "SGD")

        t1 = time()
        self.scrub_hessian(selectionType = selectionType, order = order, approxType = approxType, l2lambda = l2lambda)
        t2 = time()

        self.logger.info("Hessian unlearning prop %.3f timing: %.4f s" % (unlearn_prop, t2 - t1))

        # self.testModel()

        if(save_unlearned_model):
            self.saveModel(unlearn_prop)

    def saveModel(self, unlearn_prop):
        save_path = None
        folder = self.model_save_path
        model_folder_name = self.dataset_name + "-HessianUnlearn"
        model_folder = os.path.join(folder, model_folder_name)
        if(not os.path.exists(model_folder)):
            os.mkdir(model_folder)

        model_name =  "unlearn_prop_" + str(unlearn_prop)
        save_path = os.path.join(model_folder, model_name + ".pt")

        torch.save(self.model.state_dict(), save_path)
        self.logger.info("Having saved model to %s." % save_path)
        return save_path

    def reverseLinearIndexingToLayers(self, selectedSlices, torchLayers):
        ind_list = []
        for myslice in selectedSlices:
            prevslicecnt = 0
            if isinstance(torchLayers[0], torch.nn.Conv2d):
                nextslicecnt = torchLayers[0].out_channels
            elif isinstance(torchLayers[0], torch.nn.Linear):
                nextslicecnt = torchLayers[0].out_features
            else:
                print(f'cannot reverse process layer: {torchLayers[0]}')
                return NotImplementedError

            for l in range(len(torchLayers)):
                if myslice < nextslicecnt:
                    modslice = myslice - prevslicecnt
                    ind_list.append([torchLayers[l], modslice])
                    break

                prevslicecnt = nextslicecnt

                if isinstance(torchLayers[l+1], torch.nn.Conv2d):
                    nextslicecnt += torchLayers[l+1].out_channels
                elif isinstance(torchLayers[l+1], torch.nn.Linear):
                    nextslicecnt += torchLayers[l+1].out_features
                else:
                    print(f'cannot reverse process layer: {torchLayers[l+1]}')
                    return NotImplementedError

        return ind_list

    def DisableBatchNorm(self, model):
        for name ,child in (model.named_children()):
            if name.find('BatchNorm') != -1:
                pass
            else:
                child.eval()
                child.track_running_stats=False

        return model

    def CR_NaiveNewton(self, weight, grad, hessian, l2lambda=0.001, hessian_device='cpu'):
        original_device = weight.device

        smoothhessian = hessian.to(hessian_device) + (l2lambda)*torch.eye(hessian.shape[0]).to(hessian_device)
        newton = torch.linalg.solve(smoothhessian, grad.to(hessian_device)).to(original_device)

        # removal, towards positive gradient direction
        new_weight = weight + newton

        return new_weight

    def updateModelParams(self, updatedParams, reversalDict, model):
        for key in reversalDict.keys():
            layername, weightbias, uu = key
            start_idx, end_idx, orig_shape, param = reversalDict[key]

            # slice this update
            vec_w = updatedParams[start_idx:end_idx]

            # reshape
            reshaped_w = vec_w.reshape(orig_shape)

            #tmp = [t for t in model.named_parameters()]
            #print(layername, weightbias, uu)
            #print('param:', id(param))
            #print('model:', id(tmp[0][1]))

            # apply position update
            #print(param[uu])
            param[uu] = reshaped_w.clone().detach()
            #print(param[uu])

        return

    def scrub_hessian(self, selectionType = "Full", order = "Hessian", approxType = "FD", l2lambda = 0.001):
        """
        Modified from scrub_tools.py (https://github.com/vsingh-group/LCODEC-deep-unlearning.git)
        """

        # data_loader = self.unlearn_loader
        # x = x.to(self.device)
        # y_true = y_true.to(self.device)
        
        dataset = self.original_unlearn_dataset
        x, y_true = dataset[0][0], dataset[0][1]
        x, y_true = torch.Tensor(x).to(self.device), torch.Tensor([y_true]).type(torch.long).to(self.device)
        x.unsqueeze_(0)
    
        model_copy = copy.deepcopy(self.model)

        myActs = ActivationsHook(self.model)

        torchLayers = myActs.getLayers()

        activations = []
        layers = None # same for all, filled in by loop below
        losses = []

        self.model = self.model.to(self.device)
        self.model.eval()

        n_perturbations = 1000

        for m in range(n_perturbations):
            tmpdata = x + (0.1)*torch.randn(x.shape).to(self.device)
            acts, out = myActs.getActivations(tmpdata.to(self.device))
            loss = self.criterion(out, y_true)
            vec_acts = p2v(acts)

            activations.append(vec_acts.detach())
            losses.append(loss.detach())

        acts = torch.vstack(activations)
        losses = torch.Tensor(losses).to(self.device)

        # descructor is not called on return for this
        # call it manually
        myActs.clearHooks()

        if selectionType == 'Full':
            # here is only the activation layer
            selectedActs = np.arange(len(vec_acts)).tolist()

        elif selectionType == 'Random':
            foci_result, _ = foci(acts, losses, earlyStop=True, verbose=False)
            selectedActs = np.random.permutation(len(vec_acts))[:int(len(foci_result))]
            
        elif selectionType == 'One':
            selectedActs = [np.random.permutation(len(vec_acts))[0]]

        elif selectionType == 'FOCI':
            selectedActs, scores = foci(acts, losses, earlyStop=True, verbose=False)
            if(len(selectedActs) > 5):
                selectedActs = selectedActs[:5]
        else: 
            raise('Unknown scrub type!')
        
        # create mask for update
        # params_mask = [1 if i in params else 0 for i in range(vec_acts.shape[1])]

        slices_to_update = self.reverseLinearIndexingToLayers(selectedActs, torchLayers)
        self.logger.info('Selected model blocks to update: %s' % slices_to_update)

        ############ Sample Forward Pass ########
        self.model.train()
        self.model = self.DisableBatchNorm(self.model)
        total_loss = 0
        total_accuracy = 0

        y_pred = self.model(x)
        sample_loss_before = self.criterion(y_pred, y_true)
        self.logger.info('Sample Loss Before: %s' % sample_loss_before)

        ####### Sample Gradient
        self.optimizer.zero_grad()
        sample_loss_before.backward()

        fullprevgradnorm = gradNorm(self.model)
        self.logger.info('Sample Gradnorm Before: %s' % fullprevgradnorm)

        sampGrad1, _ = getGradObjs(self.model)
        vectGrad1, vectParams1, reverseIdxDict = getVectorizedGrad(sampGrad1, self.model, slices_to_update, self.device)
        
        self.model.zero_grad()

        if order == 'Hessian':

            # old hessian
            #second_last_name = outString + '_epoch_'  + str(params.train_epochs-2) + "_grads.pt"
            #dwtlist = torch.load(second_last_name)
            #delwt, vectPOld, _ = getVectorizedGrad(dwtlist, model, slices_to_update, device)
            # delwt, vectPOld = getOldPandG(outString, params.train_epochs-2, model, slices_to_update, device)

            #one_last_name = outString + '_epoch_'  + str(params.train_epochs-1) + "_grads.pt"
            #dwtm1list = torch.load(one_last_name)
            #delwtm1, vectPOld_1, _ = getVectorizedGrad(dwtm1list, model, slices_to_update, device)
            # delwtm1, vectPOld_1 = getOldPandG(outString, params.train_epochs-1, model, slices_to_update, device)

            # oldHessian = getHessian(delwt, delwtm1, params.approxType, w1=vectPOld, w2=vectPOld_1, hessian_device=params.hessian_device)

            # sample hessian
            model_copy = model_copy.to(self.device)
            model_copy.train()
            model_copy = self.DisableBatchNorm(model_copy)

            # for finite diff use a small learning rate
            # default adam is 0.001/1e-3, so use it here
            optim_copy = torch.optim.SGD(model_copy.parameters(), lr=1e-3)

            y_pred = model_copy(x)
            loss = self.criterion(y_pred, y_true)
            optim_copy.zero_grad()
            loss.backward()

            # step to get model at next point, compute gradients
            optim_copy.step()

            y_pred = model_copy(x)
            loss = self.criterion(y_pred, y_true)
            optim_copy.zero_grad()
            loss.backward()

            self.logger.info('Sample Loss after Step for Hessian: %s' % loss)

            # sampGrad2, _ = getGradObjs(model_copy)
            vectGrad2, vectParams2, _ = getVectorizedGrad(sampGrad1, model_copy, slices_to_update, self.device)
            
            # torch.Size([11679912]) torch.Size([11679912]) for resnet 18
            #print(vectGrad1.shape, vectGrad2.shape)
            sampleHessian = getHessian(vectGrad1, vectGrad2, approxType, w1=vectParams1, w2=vectParams2, hessian_device = self.device)

            updatedParams = self.CR_NaiveNewton(vectParams1, vectGrad1, sampleHessian, hessian_device = self.device, l2lambda = l2lambda)

        elif order == 'BP':
            updatedParams = vectParams1 + self.model_lr*vectGrad1
        else:
            raise('unknown scrubtype')
        
        with torch.no_grad():
            self.updateModelParams(updatedParams, reverseIdxDict, self.model)

        y_pred = self.model(x)
        loss2 = self.criterion(y_pred, y_true)
        self.logger.info('Sample Loss After: %s' % loss2)
        # assert loss2.item() < 1
        
        self.optimizer.zero_grad()
        loss2.backward()

        fullscrubbedgradnorm = gradNorm(self.model)
        self.logger.info('Sample Gradnorm After: %s' % fullscrubbedgradnorm)

        self.model.zero_grad()

        foci_val = 0
    
        return self.model.state_dict()
