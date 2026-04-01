"""
This code is modified from the the demo code for the USENIX Security 22 paper "ML-Doctor: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models"
"""
import os
import glob
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
np.set_printoptions(threshold=np.inf)

from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class attack_for_blackbox():
    # def __init__(self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, target_model, shadow_model, attack_model, device):
    def __init__(self, ATTACK_SETS, attack_train_loader, attack_testloader, 
                 target_model, shadow_model, attack_model, 
                 device, logger):
        
        self.device = device
        self.logger = logger

        # self.TARGET_PATH = TARGET_PATH
        # self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS

        """
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))
        """
        self.target_model = target_model
        self.shadow_model = shadow_model
        
        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)
        
        self.target_model.eval()
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_testloader

        if(type(attack_model).__name__ == "ShadowAttackModel"):
            self.attack_model = attack_model.to(self.device)
            torch.manual_seed(0)
            self.attack_model.apply(weights_init)
            
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)
        elif(type(attack_model).__name__ == "DT"):
            self.attack_model = attack_model

        self.attack_train_path = self.ATTACK_SETS + "_train.p"
        self.attack_test_path = self.ATTACK_SETS + "_test.p"

    def _get_data(self, model, inputs, targets):
        result = model(inputs)
        #print("result.shape:", result.shape, targets.shape, inputs.shape)
        output, _ = torch.sort(result, descending=True)
        # results = F.softmax(results[:,:5], dim=1)
        _, predicts = result.max(1)

        prediction = predicts.eq(targets).float()
        
        # prediction = []
        # for predict in predicts:
        #     prediction.append([1,] if predict else [0,])

        # prediction = torch.Tensor(prediction)

        # final_inputs = torch.cat((results, prediction), 1)
        # print(final_inputs.shape)

        return output, prediction.unsqueeze(-1)

    def prepare_dataset(self):

        with open(self.attack_train_path, "wb") as f:
            # for i, (images, labels, members, idx) in enumerate(self.train_loader):
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.shadow_model, inputs, targets)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        self.logger.info("Having saved attack TRAIN Dataset to %s" % self.attack_train_path)

        with open(self.attack_test_path, "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.target_model, inputs, targets)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        self.logger.info("Having saved attack TEST Dataset to %s" % self.attack_test_path)

    def train(self, flag, result_path, epoch, save_result = False):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.attack_train_path, "rb") as f:
            while(True):
                try:
                    output, prediction, members = pickle.load(f)
                    output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                    results = self.attack_model(output, prediction)
                    results = F.softmax(results, dim=1)

                    results = results.float()
                    members = members.long()
                    losses = self.criterion(results, members)
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    if flag:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break

        if flag:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_recall_score = recall_score(final_train_gndtrth, final_train_predict)
            train_precision_score = precision_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            if(save_result):
                with open(result_path, "wb") as f:
                    pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
                self.logger.info("Saved Attack Train Ground Truth and Predict Sets to %s" % result_path)
            
            self.logger.info("Epoch %d Train F1: %f Recall: %f Precision: %f AUC: %f" % 
                             (epoch, train_f1_score, train_recall_score, train_precision_score, 
                              train_roc_auc_score))

        Acc = -1
        if(total):
            Acc = 100.*correct/total
        
        final_loss = -1
        if(batch_idx):
            final_loss = 1.*train_loss/batch_idx
            
        self.logger.info( 'Epoch %d Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (epoch, Acc, correct, total, final_loss))

        return final_result

    def test(self, flag, result_path, epoch, testset_path = None, save_result = False):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        prefix = ""
        
        with torch.no_grad():
            with open(testset_path, "rb") as f:
                while(True):
                    try:
                        output, prediction, members = pickle.load(f)
                        output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                        results = self.attack_model(output, prediction)
                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()
                        results = F.softmax(results, dim=1)

                        if flag:
                            final_test_gndtrth.append(members)
                            final_test_predict.append(predicted)
                            final_test_probabe.append(results[:, 1])

                        batch_idx += 1
                    except EOFError:
                        break

        if flag:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_recall_score = recall_score(final_test_gndtrth, final_test_predict)
            test_precision_score = precision_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)

            if(save_result):
                with open(result_path, "wb") as f:
                    pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

                self.logger.info("Saved Attack Test Ground Truth and Predict Sets to %s" % result_path)
            self.logger.info("Epoch %d %s Test F1: %f Recall: %f Precision: %f AUC: %f" % 
                             (epoch, prefix, test_f1_score, test_recall_score, test_precision_score,
                              test_roc_auc_score))

        final_result.append(1.*correct/total)

        self.logger.info( 'Epoch %d %s Test Acc: %.3f%% (%d/%d)' % (epoch, prefix, 100.*correct/(1.0*total), correct, total))

        return final_result

    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)
