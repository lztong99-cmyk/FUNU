import os
import torch
import torchvision
import torch.nn as nn
import logging
import numpy as np
from time import time
from lib_model.CNN import ConvNet
from lib_model.ResNet import ResNet18, ResNet34, ResNet50, Identity
from utils import model_selection, gen_save_name, test_model, get_statistics, \
freeze_conv_layers, get_files_with_wildcard, SHARD_NUM, ORIGINAL_PROPORTION, \
get_reference_model_path, get_presentation_similarity, save_content_to_npy, \
get_sim_matrix_theta_typical_indices_path, save_score_to_pt, FULL_SIM_MATRIX_NAME, \
locate_value_in_tensor, get_tensor_intersection, get_conf_model_path
from lib_dataset.Dataset import MyDataset

"""
This class is for training certain model
"""
class Trainer():
    def __init__(self, dataset_name, dataset_path, 
                 model_name, model_save_path, device):

        # Data path
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path

        # Logging
        self.logger = logging.getLogger('trainer_log')

        # Training setting
        self.dataset_name = dataset_name
        self.model_name = model_name

        # Device
        self.device = torch.device(device)

        self.total_epoch = 0
             
        return
    
    def update_model_name(self, model_name):
        self.model_name = model_name

    def get_typical_dataset_indices(self, ret_untypical = False):
        # return tensor of the typical dataset index
        predicted, labels = test_model(model = self.model, dataloader = self.train_loader, device = self.device, 
                                logger = self.logger, dataset_len = len(self.train_dataset), op = "label",
                                echo = False, demo = False)
        if(ret_untypical):
            return torch.nonzero(predicted.eq(labels)), torch.nonzero(predicted.ne(labels))
        else:
            return torch.nonzero(predicted.eq(labels))

    def cal_theta_for_rfmodel(self, full_similarity_matrix, typical_dataset_indices, theta_by_label = True, 
                              ret_theta_list = True, label_num = 5, method = "avg"):
        assert len(full_similarity_matrix.shape) == 2

        self.logger.info("The dataset has %d typical samples" % len(typical_dataset_indices))
        
        if(torch.is_tensor(full_similarity_matrix) == False):
            full_similarity_matrix = torch.tensor(full_similarity_matrix)
            
        if(torch.is_tensor(self.train_dataset.targets) == False):
            targets = torch.tensor(self.train_dataset.targets)
        else:
            targets = self.train_dataset.targets
            
        if(len(typical_dataset_indices.shape) != 1):
            typical_dataset_indices = typical_dataset_indices.squeeze()
        
        alpha = None
        
        if(theta_by_label):
            theta_by_label_list = []
            alpha_by_label_list = []
            
            for label in range(label_num):
                label_index = locate_value_in_tensor(targets, torch.tensor(label))
                tmp_index = get_tensor_intersection(label_index, typical_dataset_indices)
                
                if(len(tmp_index) == 0):
                    print("label:", label, "len(tmp_index):", len(tmp_index), len(typical_dataset_indices))
                    continue
                else:
                    print("label:", label, "len(tmp_index):", len(tmp_index))
                
                typical_data_similarity_matrix = torch.index_select(input = full_similarity_matrix, 
                                                dim = 0, index = tmp_index)
                typical_data_similarity_matrix = torch.index_select(input = typical_data_similarity_matrix, 
                                                dim = 1, index = tmp_index)
                
                # typical_data_similarity_matrix = torch.abs(typical_data_similarity_matrix)
                
                # print("min:", torch.min(typical_data_similarity_matrix).item(),  "max:",torch.max(typical_data_similarity_matrix).item(),  
                #       "mean:", torch.mean(typical_data_similarity_matrix).item())
                
                if(method == "min"):
                    cur_theta = torch.min(typical_data_similarity_matrix).item()
                    theta_by_label_list.append(cur_theta)
                    
                elif(method == "avg"):
                    cur_theta = torch.mean(typical_data_similarity_matrix).item()
                    
                    theta_by_label_list.append(cur_theta)
                    
                    cur_alpha = (typical_data_similarity_matrix <= cur_theta).sum().item() / len(tmp_index)
                    
                    alpha_by_label_list.append(cur_alpha)
            
            print("theta_by_label_list: ", theta_by_label_list)
            print("alpha_by_label_list: ", alpha_by_label_list)
            
            if(method == "avg"):
                theta = sum(theta_by_label_list) / len(theta_by_label_list)
                alpha = sum(alpha_by_label_list) / len(alpha_by_label_list)
            elif(method == "min"):
                theta = max(theta_by_label_list)
                
                typical_data_similarity_matrix = torch.index_select(input = full_similarity_matrix, 
                                                dim = 0, index = typical_dataset_indices)
            
                typical_data_similarity_matrix = torch.index_select(input = typical_data_similarity_matrix, 
                                                    dim = 1, index = typical_dataset_indices)
                
                assert typical_data_similarity_matrix.size(0) == typical_data_similarity_matrix.size(1)
                
                alpha = (typical_data_similarity_matrix <= theta).sum().item() // label_num
                
            self.logger.info("theta WITH grouping by label: %.4f, alpha: %.4f" % (theta, alpha))
        
        else:
            # select typical data matrix
            typical_data_similarity_matrix = torch.index_select(input = full_similarity_matrix, 
                                                dim = 0, index = typical_dataset_indices)
            
            typical_data_similarity_matrix = torch.index_select(input = typical_data_similarity_matrix, 
                                                dim = 1, index = typical_dataset_indices)
            
            assert typical_data_similarity_matrix.size(0) == typical_data_similarity_matrix.size(1)
            
            theta = torch.mean(typical_data_similarity_matrix).item()
            
            alpha = (typical_data_similarity_matrix <= theta).sum().item()
            
            self.logger.info("theta WITHOUT grouping by label: %.4f, alpha: %.4f" % (theta, alpha))
        
        if(ret_theta_list):
            return theta, alpha, theta_by_label_list, alpha_by_label_list
        else:
            return theta, alpha

    def cal_theta_for_rfmodel_untypical(self, full_similarity_matrix, typical_dataset_indices,untypical_dataset_indices, 
                                        theta_by_label = True, label_num = 5, method = "avg"):
        assert len(full_similarity_matrix.shape) == 2
        
        print("len(untypical_dataset_indices):", len(untypical_dataset_indices))
        
        if(torch.is_tensor(full_similarity_matrix) == False):
            full_similarity_matrix = torch.tensor(full_similarity_matrix)
            
        if(torch.is_tensor(self.train_dataset.targets) == False):
            targets_tensor = torch.from_numpy(self.train_dataset.targets)
        else:
            targets_tensor = self.train_dataset.targets
        
        if(len(untypical_dataset_indices.shape) != 1):
            untypical_dataset_indices = untypical_dataset_indices.squeeze()
            
        if(len(typical_dataset_indices.shape) != 1):
            typical_dataset_indices = typical_dataset_indices.squeeze()
        
        alpha = None
        
        if(theta_by_label):
            theta_by_label_list = []
            alpha_by_label_list = []
            for label in range(label_num):
                label_index = locate_value_in_tensor(targets_tensor, label)
                tmp_index = get_tensor_intersection(label_index, untypical_dataset_indices)
                
                if(len(tmp_index) == 0):
                    continue
                
                untypical_data_similarity_matrix = torch.index_select(input = full_similarity_matrix, 
                                                dim = 0, index = tmp_index)
                untypical_data_similarity_matrix = torch.index_select(input = untypical_data_similarity_matrix, 
                                                dim = 1, index = tmp_index)
                
                # print("min:", torch.min(untypical_data_similarity_matrix).item(),  "max:",torch.max(untypical_data_similarity_matrix).item(),  
                #      "mean:", torch.mean(untypical_data_similarity_matrix).item())
                
                untypical_data_similarity_matrix = torch.abs(untypical_data_similarity_matrix)
                
                if(method == "min"):              
                    cur_theta = torch.min(untypical_data_similarity_matrix).item()
                    theta_by_label_list.append(cur_theta)
                    
                elif(method == "avg"):
                    cur_theta = torch.mean(untypical_data_similarity_matrix).item()
                    
                    theta_by_label_list.append(cur_theta)
                    
                    """
                    unty_typical_data_similarity_matrix = torch.index_select(input = full_similarity_matrix, 
                                                dim = 0, index = tmp_index)
            
                    unty_typical_data_similarity_matrix = torch.index_select(input = unty_typical_data_similarity_matrix, 
                                                    dim = 1, index = typical_dataset_indices)
                
                    cur_alpha = (unty_typical_data_similarity_matrix >= cur_theta).sum().item() / len(tmp_index) # torch.max(torch.sum(unty_typical_data_similarity_matrix <= cur_theta, dim = 1)).item()
                    
                    alpha_by_label_list.append(cur_alpha)
                    """
            print("theta_by_label_list: ", theta_by_label_list)
            print("alpha_by_label_list: ", alpha_by_label_list)
            
            if(method == "avg"):
                theta = sum(theta_by_label_list) / len(theta_by_label_list)
                
                typical_data_similarity_matrix = torch.index_select(input = full_similarity_matrix, 
                                                dim = 0, index = untypical_dataset_indices)
            
                typical_data_similarity_matrix = torch.index_select(input = typical_data_similarity_matrix, 
                                                    dim = 1, index = typical_dataset_indices)
                
                assert typical_data_similarity_matrix.size(0) == len(untypical_dataset_indices)
                
                # alpha = torch.max(torch.sum(unty_typical_data_similarity_matrix <= theta, dim = 1), dim = 0).values.item()
                
                alpha = (typical_data_similarity_matrix <= theta).sum().item() / len(untypical_dataset_indices)
                
                # alpha = sum(alpha_by_label_list) / len(alpha_by_label_list) # min(alpha_by_label_list)
            elif(method == "min"):
                theta = max(theta_by_label_list)
                
                typical_data_similarity_matrix = torch.index_select(input = full_similarity_matrix, 
                                                dim = 0, index = typical_dataset_indices)
            
                typical_data_similarity_matrix = torch.index_select(input = typical_data_similarity_matrix, 
                                                    dim = 1, index = typical_dataset_indices)
                
                assert typical_data_similarity_matrix.size(0) == typical_data_similarity_matrix.size(1)
                
                alpha = (typical_data_similarity_matrix >= theta).sum().item() // label_num
            
            self.logger.info("theta WITH grouping by label: %.4f, alpha: %.4f" % (theta, alpha))
        
        else:
            # select typical data matrix
            untypical_data_similarity_matrix = torch.index_select(input = full_similarity_matrix, 
                                                dim = 0, index = untypical_dataset_indices)
            
            untypical_data_similarity_matrix = torch.index_select(input = untypical_data_similarity_matrix, 
                                                dim = 1, index = untypical_dataset_indices)
            
            print("untypical_data_similarity_matrix.shape:", untypical_data_similarity_matrix.shape)
            
            untypical_data_similarity_matrix = torch.abs(untypical_data_similarity_matrix)
            theta = torch.mean(untypical_data_similarity_matrix).item()
            
            print("min:", torch.min(untypical_data_similarity_matrix).item(),  "max:",torch.max(untypical_data_similarity_matrix).item(),  
                "mean:", torch.mean(untypical_data_similarity_matrix).item())
           
            untypical_data_similarity_matrix = torch.index_select(input = full_similarity_matrix, 
                                                dim = 0, index = untypical_dataset_indices)
            
            unty_typical_data_similarity_matrix = torch.index_select(input = untypical_data_similarity_matrix, 
                                                dim = 1, index = typical_dataset_indices)
            
            assert unty_typical_data_similarity_matrix.size(0) == len(untypical_dataset_indices)
            
            alpha = (unty_typical_data_similarity_matrix >= theta).sum().item() / len(untypical_dataset_indices)
            
            self.logger.info("theta WITHOUT grouping by label: %.4f, alpha: %.4f" % (theta, alpha))
            
        return theta, alpha
    
    def get_untypical_dataset_indices_by_conf(self, softmax_conf = True, untypical_prop = -1, typical_prop = 0.1):
        
        conf_data = test_model(model=self.model, dataloader = self.train_loader, device=self.device, logger = self.logger, dataset_len = len(self.train_dataset),
                                op = "conf_data", echo = False, demo = False)
        
        if(untypical_prop == -1 and typical_prop < 0):
            
            conf_values, pred = torch.max(conf_data, dim = 1)
            targets = torch.tensor(self.train_dataset.targets) if not torch.is_tensor(self.train_dataset.targets) else self.train_dataset.targets
            
            if(targets.device.type != pred.device.type):
                pred = pred.to(targets.device.type)
            
            acc = (pred == targets).sum().item()/len(targets)
            untypical_prop = 1 - (pred == targets).sum().item()/len(targets)
            if(untypical_prop == 1):
                untypical_prop = 0.01
                self.logger.info("ACC on training dataset is 100%. untypical_prop is set to 0.01.")  
            else:
                self.logger.info("acc: %.4f, untypical_prop is set to %.4f" % (acc, untypical_prop))
            
        conf_data = conf_data.cpu()
        
        if(softmax_conf):
            assert conf_data.dim() == 2 and conf_data.size(1) == self.class_num
            softmax_func = nn.Softmax(dim = 1)
            conf_data = softmax_func(conf_data)
            
        conf_values = torch.max(conf_data, dim = 1).values
        # print(torch.mean(conf_values))
        
        if(typical_prop < 0):
            untypical_count = int(untypical_prop * conf_values.shape[0])
            untypical_indices = torch.sort(conf_values).indices[:untypical_count]
            typical_indices = torch.sort(conf_values).indices[untypical_count:]
        else:
            typical_count = int(typical_prop * conf_values.shape[0])
            typical_indices = torch.sort(conf_values, descending = True).indices[:typical_count]
            print("torch.mean(conf_values[typical_indices]):", torch.mean(conf_values[typical_indices]), torch.min(conf_values[typical_indices]), torch.max(conf_values[typical_indices]))
            untypical_indices = torch.sort(conf_values, descending = True).indices[typical_count:]
        return typical_indices, untypical_indices 
    
    def cal_similarity_matrix_typical_theta(self, conf_epoch = 30, theta_by_label = True, presentation = "model_feature"):
        """
        Calculate similarity matrix, typical sample indices, theta, alpha.
        """

        # check if file exists
        full_similarity_matrix_path, typical_dataset_indices_path, theta_path, alpha_path = get_sim_matrix_theta_typical_indices_path(self.dataset_name, self.model_name)

        # if theta exists, read directly. Otherwise, calculate theta.
        if(0 and theta_by_label):
            if(os.path.exists(theta_path) and os.path.exists(alpha_path)):
                theta = torch.load(theta_path)
                alpha = torch.load(alpha_path)
                
                self.logger.info("Directly load theta WITH grouping by label: %s, alpha: %s" % (theta, alpha))
                return theta, alpha
        
        self.load_data(unlearning=0, unlearning_data_selection=None,
                        unlearning_proportion=None, original_proportion = ORIGINAL_PROPORTION,
                        left=False,training_batch_size = 64, sisa_selection_op = -1)
        
        if(presentation == "confidence" or presentation == "model_feature"):
            self.initialize_model(shadow_model = False, model_load_path = get_conf_model_path(dataset_name=self.dataset_name, model_name=self.model_name,
                                                                                              epoch = conf_epoch), 
                                pretrained = False, freeze_conv_layer=False)
            if(presentation == "model_feature"):
                self.model.fc = Identity()
        
        feature_data = None
        feature_extractor = "ResNet-18"
        if(presentation == "confidence" ):
            feature_data = test_model(model=self.model, dataloader = self.train_loader, device=self.device, logger = self.logger, dataset_len = len(self.train_dataset),
                                op = "conf_data", echo = False, demo = False)
        elif(presentation == "model_feature"):
            feature_data = test_model(model=self.model, dataloader = self.train_loader, device=self.device, logger = self.logger, dataset_len = len(self.train_dataset),
                                op = "model_feature_data", echo = False, demo = False)
            
        if(os.path.exists(full_similarity_matrix_path)):
            full_similarity_matrix = torch.load(full_similarity_matrix_path)
            self.logger.info("Load full_similarity_matrix from %s, full_similarity_matrix shape: %s" % (full_similarity_matrix_path, full_similarity_matrix.shape))
        else:
            full_similarity_matrix = get_presentation_similarity(dataset = self.train_dataset, dataset_name = self.dataset_name, logger = self.logger, 
                                                                theta = -1, min_similar_samples = 10, feature_extractor = feature_extractor, 
                                                                model_pretrained = True,  distance = "cos", save_dist_distribution = False, 
                                                                demo = False, ret_dist_matrix = True, presentation = presentation, feature_data = feature_data)

            if(presentation == "confidence"):
                assert (full_similarity_matrix<0).sum().item() == 0
                
            file_path = save_score_to_pt(score = full_similarity_matrix, dataset_name = self.dataset_name, model_name = self.model_name, 
                             score_type = FULL_SIM_MATRIX_NAME)
            self.logger.info("Save full_similarity_matrix to path: %s, presentation: %s" % (file_path, presentation))
            # full_similarity_matrix = torch.tensor(full_similarity_matrix)
            # torch.save(full_similarity_matrix, full_similarity_matrix_path)

        if(presentation == "model_feature" or presentation == "confidence"):
            self.initialize_model(shadow_model = False, model_load_path = get_reference_model_path(self.dataset_name, self.model_name), 
                                 pretrained = False, freeze_conv_layer=False)
        
        t1 = time()
        
        typical_dataset_indices = self.get_typical_dataset_indices()
        
        # theta, alpha = self.cal_theta_for_rfmodel(full_similarity_matrix, typical_dataset_indices, theta_by_label)
        
        # typical_dataset_indices, untypical_dataset_indices = self.get_typical_dataset_indices(ret_untypical=True)
        
        # conf_data = test_model(model=self.model, dataloader = self.train_loader, device=self.device, logger = self.logger, dataset_len = len(self.train_dataset),
        #                            op = "conf_data", echo = False, demo = False)
        
        # typical_dataset_indices, untypical_dataset_indices = self.get_untypical_dataset_indices_by_conf()
        
        # theta, alpha = self.cal_theta_for_rfmodel_untypical(full_similarity_matrix, typical_dataset_indices, untypical_dataset_indices, 
         #                                                  theta_by_label = True, label_num = 5, method = "avg")
        
        ret_theta_list = False
        sim_theta_list, sim_alpha_list = [], []
        if(ret_theta_list):
            theta, alpha, sim_theta_list, sim_alpha_list = self.cal_theta_for_rfmodel(full_similarity_matrix, typical_dataset_indices, theta_by_label, ret_theta_list)
        else:
            theta, alpha = self.cal_theta_for_rfmodel(full_similarity_matrix, typical_dataset_indices, theta_by_label, ret_theta_list)
        
        t2 = time()

        self.logger.info("Calculating similarity theta (and alpha) timing: %.4f s" % (t2 - t1))
        # append theta to the first typical_dataset_indices.

        torch.save(typical_dataset_indices, typical_dataset_indices_path)
        
        if(ret_theta_list):
            torch.save(sim_theta_list, theta_path)
            torch.save(sim_alpha_list, alpha_path)
        else:
            torch.save(theta, theta_path)
            torch.save(alpha, alpha_path)
            
        #if(alpha):
            # assert theta_by_label
        #    torch.save(alpha, alpha_path)
        
        if(ret_theta_list):
            return sim_theta_list, sim_alpha_list
        else:
            return theta, alpha

    def load_data(self, unlearning, unlearning_data_selection, unlearning_proportion,
                  original_proportion = 0.6, left = False, training_batch_size = 64,
                  unlearned_index = None, demo = False, sisa_selection_op = -1,
                  unlearning_filter = None, sim_theta = -1, sim_alpha = -1,
                  score_thres_dict = {}, remove_class = -1, shuffle_train = True):
        
        # Unlearning setting if there is any
        self.unlearning = unlearning
        self.unlearning_data_selection=unlearning_data_selection
        self.unlearning_proportion=unlearning_proportion
        
        self.unlearned_index = unlearned_index if self.unlearning else None
        
        self.training_batch_size = training_batch_size

        # Initialize dataset
        self.train_dataset = 1
        self.unlearned_dataset = 1
        self.test_dataset = 1
        
        t1 = time()

        if(unlearning_filter):
            self.unlearning_filter = unlearning_filter
        else:
            self.unlearning_filter = "NoFilter"
            
        # MNIST & CIFAR10 dataset
        if(self.unlearning in [0,2,3]):
            if(self.dataset_name in [ "MNIST", "CIFAR10", "CIFAR100","Tiny-ImageNet"]):
                self.train_dataset = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                               unlearning_data_selection=self.unlearning_data_selection, 
                                               unlearning_proportion=self.unlearning_proportion,
                                               dataset_type ="train", unlearning=self.unlearning, 
                                               original_proportion = original_proportion, left=left,
                                               demo=demo,sisa_selection_op = sisa_selection_op, unlearning_data_idx = None)
                self.test_dataset = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                               unlearning_data_selection= self.unlearning_data_selection, 
                                               unlearning_proportion=self.unlearning_proportion,
                                               dataset_type ="test", unlearning=self.unlearning, 
                                               original_proportion=original_proportion, left=left,
                                               demo=demo,sisa_selection_op = sisa_selection_op, unlearning_data_idx = None)
            else:
                raise Exception("Invalid dataset!")
        elif(self.unlearning == 1):
            if(self.dataset_name in [ "MNIST", "CIFAR10", "CIFAR100","Tiny-ImageNet"]):
                self.train_dataset = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                               unlearning_data_selection=self.unlearning_data_selection, 
                                               unlearning_proportion=self.unlearning_proportion,
                                               dataset_type ="remained", unlearning=self.unlearning, 
                                               original_proportion=original_proportion, left=left,demo=demo,
                                               sisa_selection_op = sisa_selection_op, unlearning_data_idx = unlearned_index,
                                               unlearning_filter = unlearning_filter, rf_model_name = self.model_name, 
                                               sim_theta = sim_theta, sim_alpha = sim_alpha, score_thres_dict = score_thres_dict,
                                               remove_class = remove_class)
                self.test_dataset = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                               unlearning_data_selection=self.unlearning_data_selection,
                                               unlearning_proportion=self.unlearning_proportion,
                                               dataset_type ="test", unlearning=self.unlearning, 
                                               original_proportion=original_proportion, left=left,demo=demo,
                                               sisa_selection_op = sisa_selection_op, unlearning_data_idx = unlearned_index,
                                               unlearning_filter = unlearning_filter, rf_model_name = self.model_name, 
                                               sim_theta = sim_theta, sim_alpha = sim_alpha, score_thres_dict = score_thres_dict,
                                               remove_class = remove_class)
                self.original_unlearn_dataset = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                               unlearning_data_selection=self.unlearning_data_selection, 
                                               unlearning_proportion=self.unlearning_proportion,
                                               dataset_type ="unlearned", unlearning=self.unlearning, 
                                               original_proportion=original_proportion, left=left,demo=demo,
                                               sisa_selection_op = sisa_selection_op, unlearning_data_idx = unlearned_index,
                                               unlearning_filter = None, rf_model_name = self.model_name, 
                                               sim_theta = sim_theta, sim_alpha = sim_alpha, score_thres_dict = score_thres_dict,
                                               remove_class = remove_class) # the unlearning_filter = None means we do not filter unlearned dataset
                if(unlearning_filter):
                    self.filtered_unlearn_dataset = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                                unlearning_data_selection=self.unlearning_data_selection, 
                                                unlearning_proportion=self.unlearning_proportion,
                                                dataset_type ="unlearned", unlearning=self.unlearning, 
                                                original_proportion = original_proportion, left=left,demo=demo,
                                                sisa_selection_op = sisa_selection_op, unlearning_data_idx = unlearned_index,
                                                unlearning_filter = unlearning_filter, rf_model_name = self.model_name, 
                                                sim_theta = sim_theta, sim_alpha = sim_alpha, score_thres_dict = score_thres_dict,
                                                remove_class = remove_class)
                
            else:
                raise Exception("Invalid dataset!")

        t2 = time()

        self.logger.info("Having load dataset %s with timing %.4f s" % (self.dataset_name, t2 - t1))

        # Set random seed for dataloader
        torch.manual_seed(123)

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.training_batch_size, 
                                                        drop_last=False, shuffle=shuffle_train)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=self.training_batch_size, shuffle=False)
        
        if(self.unlearning == 1):
            self.original_unlearn_loader = torch.utils.data.DataLoader(dataset=self.original_unlearn_dataset, batch_size=self.training_batch_size, shuffle=False)
            if(unlearning_filter):
                self.filtered_unlearn_loader = torch.utils.data.DataLoader(dataset=self.filtered_unlearn_dataset, batch_size=self.training_batch_size, shuffle=False)
                
    def test_on_Du_Dr_Dt(self, op = "ACC"):

        # begin evaluate
        if(self.unlearning_filter != "NoFilter" and len(self.filtered_unlearn_dataset)):
            M_Duf = test_model(self.model, self.filtered_unlearn_loader, self.device, self.logger, op = op, echo = True).pop()
        else:
            M_Duf = -1
        M_Duo = test_model(self.model, self.original_unlearn_loader, self.device, self.logger, op = op, echo = True).pop()
        M_Dr = test_model(self.model, self.train_loader, self.device, self.logger, op = op, echo = True).pop()
        M_Dt = test_model(self.model, self.test_loader, self.device, self.logger, op = op, echo = True).pop()

        self.logger.info("Model %s on Duf: %.6f, on Duo: %.6f, on Dr: %.6f, on Dt: %.6f" % (op, M_Duf, M_Duo, M_Dr, M_Dt))
        
    def initialize_model(self, shadow_model = False, model_load_path = None, pretrained = False,
                         freeze_conv_layer = False):
        """
        Initialize model or load existing model
        """
        # Initialize model
        self.class_num = self.train_dataset.class_num
        self.input_channel = self.train_dataset.input_channel
        
        self.model = model_selection(self.model_name, self.class_num, 
                                         input_channel = self.input_channel, 
                                         shadow_model = shadow_model,
                                         pretrained = pretrained)
        
        self.logger.info("Use pretrained model: %d" % pretrained)    
        
        if(model_load_path):
            self.model.load_state_dict(torch.load(model_load_path, map_location='cpu'))
            self.logger.info("Having loaded model from %s" % model_load_path)
        else:
            self.logger.info("Have not loaded model")
        
        if(freeze_conv_layer):
            freeze_conv_layers(self.model)
        
    def prepare_train_setting(self, learning_rate = 0.001,  epochs = 10, optim = "SGD"):
        
        # Loss and optimize
        self.criterion = nn.CrossEntropyLoss()
        
        if(learning_rate):
            self.model_lr = learning_rate
            if(optim == "Adam"):
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_lr)
            elif(optim == "SGD"):
                self.optimizer = torch.optim.SGD(params = self.model.parameters(), lr = self.model_lr, 
                                                 momentum = 0.9, weight_decay = 5e-4)
            else:
                raise Exception("Invalid optimizer!")
            
        if(epochs):
            self.total_epoch += epochs
            self.epochs = epochs
        
        if(learning_rate and epochs):
            self.logger.info("Having prepared model %s, training lr: %.4f, epochs: %d" % (self.model_name, self.model_lr, self.epochs))

    def train_model(self, before_epoch = 0, save_epoch = 0, do_test = True, shadow_or_attack = 0, iter_data = 1):
        """
        save model every $save_epoch$ epochs
        """
        
        self.model.train()
        self.model = self.model.to(self.device)

        for epoch in range(1, (self.epochs+1)):
            avg_loss = 0
            n_loader = 0
            
            for i, (images, labels, idx) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backprop and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()
                n_loader += 1
                
                if(iter_data < 1):
                    if(i > len(self.train_loader) * iter_data):
                        self.final_model_path = self.saveModel( epoch = iter_data, shadow_or_attack = 0 )
                        return self.model
            
            if(do_test):
                test_acc = test_model(self.model, self.test_loader, self.device, 
                                    self.logger, op = "ACC").pop()
                self.logger.info('Epoch [{}/{}], Loss: {:.4f} | ACC: {:.4f}' \
                    .format(epoch, self.epochs, avg_loss/n_loader, test_acc))
            
            if(save_epoch):
                if(save_epoch < self.epochs):
                    if(save_epoch == 1 or epoch % save_epoch == 1):
                        self.saveModel( epoch = before_epoch + epoch, shadow_or_attack = shadow_or_attack )
                    elif(epoch == self.epochs):
                        self.final_model_path = self.saveModel( epoch = before_epoch + epoch, shadow_or_attack = shadow_or_attack )
                else:
                    if(epoch == self.epochs):
                        self.final_model_path = self.saveModel( epoch = before_epoch + epoch, shadow_or_attack = shadow_or_attack )

        return self.model
    
    def gen_sisa_model_path(self, i, original = True):
        folder = self.model_save_path
        
        if(original == True):
            model_folder_name = self.dataset_name + "-sisa-" + str(SHARD_NUM) + "shard-original"
        else:
            model_folder_name = self.dataset_name + "-sisa-" + str(SHARD_NUM) + "shard"
            
        model_folder = os.path.join(folder, model_folder_name)

        if(not os.path.exists(model_folder)):
            os.makedirs(model_folder)
            
        model_name = "submodel" + str(i)
        save_path = os.path.join(model_folder, model_name + ".pt")
        return save_path

    def gen_sisa_model_folder(self, original = True):
        folder = self.model_save_path
        
        if(original == True):
            model_folder_name = self.dataset_name + "-sisa-" + str(SHARD_NUM) + "shard-original"
        else:
            model_folder_name = self.dataset_name + "-sisa-" + str(SHARD_NUM) + "shard"
            
        model_folder = os.path.join(folder, model_folder_name)
        
        return model_folder
    
    def save_sisa_model(self, i, original = False):
        # if original is False, then it's a retrained model
        save_path = self.gen_sisa_model_path(i, original = original)
        torch.save(self.model.state_dict(), save_path)
        
        self.logger.info("Having saved model to %s." % save_path)

    def test_sisa_model(self, test_unlearn = False, dataset = "test", unlearned_data_index = None):
        submodel_list = [4,9,14,19,24,29,34,39,44,49]

        output_matrix = []
        
        self.load_data(unlearning=test_unlearn, unlearning_data_selection=None,unlearned_index=unlearned_data_index,
                    unlearning_proportion=None, original_proportion = ORIGINAL_PROPORTION,
                    left=False,training_batch_size = 64, sisa_selection_op = -1, shuffle_train = False)
        
        if(dataset == "test"):
            dataloader = self.test_loader
            true_labels = torch.tensor(self.test_dataset.targets)
        elif(dataset == "unlearned"):
            dataloader = self.original_unlearn_loader
            true_labels = torch.tensor(self.original_unlearn_dataset.targets)
        elif(dataset == "remained"):
            dataloader = self.train_loader
            true_labels = torch.tensor(self.train_dataset.targets)
        
        for i in submodel_list:
            if(test_unlearn == True):
                model_path = self.gen_sisa_model_path(i, original = False)
                
                # if the model does not exist in retrained path, then we load it from the original path
                if(not os.path.exists(model_path)):
                    model_path = self.gen_sisa_model_path(i, original = True)
            else:
                model_path = self.gen_sisa_model_path(i, original = True)
                
            self.initialize_model(shadow_model=False, model_load_path=model_path, 
                                  pretrained=False, freeze_conv_layer=False)
            
            self.model.eval()
            self.model = self.model.to(self.device)

            tmp_predicted = []
            
            for images, labels, idx in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                tmp_predicted.append(predicted)
            
            sub_model_output = torch.unsqueeze(torch.cat(tmp_predicted, dim=0), dim=1)
            output_matrix.append(sub_model_output)
                
        output_matrix = torch.cat(output_matrix, axis=-1).detach().clone().cpu()

        assert output_matrix.size(1) == len(submodel_list)
        
        voted_predicted = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=output_matrix)
        # labels = torch.tensor(self.test_dataset.targets)
        correct = (torch.tensor(voted_predicted) == true_labels).sum().item()
        
        total = output_matrix.size(0)
        acc = correct / total

        self.logger.info("Model ACC on the %d test samples: %.6f" % (total, acc))

    def testModel(self):
        # use functions in utils
        test_model(self.model, self.test_loader, self.device, self.logger, op = "ACC")
    
    def get_dataset_confidence_score(self, logger, demo = False, softmax = True):
        # use functions in utils
        conf_data =  test_model(self.model, self.train_loader, self.device, logger=logger, dataset_len=len(self.train_dataset),
                            op = "conf_data", demo = demo)
        
        if(softmax):
            assert conf_data.dim() == 2 and conf_data.size(1) == self.class_num
            softmax_func = nn.Softmax(dim = 1)
            conf_data = softmax_func(conf_data)
        
        # conf_score is the max value of conf_data at each row
        conf_score, _ = torch.max(conf_data, dim = 1)
        
        assert conf_score.dim() == 1 and conf_score.size(0) == conf_data.size(0)
        
        return conf_score
    
        # return test_model(self.model, self.train_loader, self.device, logger=logger, dataset_len=len(self.train_dataset),
        #                   op = "conf", demo = demo)

    def saveModel(self, epoch, shadow_or_attack = 0, path = "default"):
        save_path = None
        if(path == "default"):
            folder = self.model_save_path
            model_folder_name = gen_save_name(dataset_name = self.dataset_name, model_name=self.model_name,
                                       unlearning_data_selection = self.unlearning_data_selection, 
                                       unlearning_proportion = self.unlearning_proportion,
                                       unlearning = self.unlearning, shadow_or_attack = shadow_or_attack,
                                       return_save_folder = True)
            model_folder = os.path.join(folder, model_folder_name)
            if(not os.path.exists(model_folder)):
                os.mkdir(model_folder)
            
            self.model_path = model_folder
            
            model_name = gen_save_name(dataset_name=self.dataset_name, model_name=self.model_name,
                                       unlearning_data_selection = self.unlearning_data_selection, 
                                       unlearning_proportion = self.unlearning_proportion,
                                       unlearning = self.unlearning, shadow_or_attack = shadow_or_attack,
                                       epoch=epoch, short = True, filter_name = self.unlearning_filter)
            save_path = os.path.join(model_folder, model_name + ".pt")
        
        torch.save(self.model.state_dict(), save_path)
        self.logger.info("Having saved model to %s." % save_path)
        
        return save_path
    
    def training_choice(self, learning_rate, epochs, save_epoch, iter_data = 1, shadow_or_attack = 0):
        if(self.dataset_name == "CIFAR10"):
            if(shadow_or_attack > 0):
                self.prepare_train_setting(learning_rate = 0.01,  epochs = 10, optim = "SGD")
            else:
                self.prepare_train_setting(learning_rate = 0.01,  epochs = 10, optim = "SGD")
            self.train_model(save_epoch = save_epoch, shadow_or_attack = shadow_or_attack, iter_data = iter_data)
        elif(self.dataset_name == "MNIST"):
            self.prepare_train_setting(learning_rate = learning_rate, epochs = epochs)
            self.train_model(save_epoch = save_epoch, shadow_or_attack = shadow_or_attack, iter_data = iter_data)
        elif(self.dataset_name == "Tiny-ImageNet" or self.dataset_name == "CIFAR100"):
            if(shadow_or_attack > 0):
                first_stage_epoch = 10
                second_stage_epoch = 20
                third_stage_epoch = 0
            else:
                first_stage_epoch = 10
                second_stage_epoch = 20
                third_stage_epoch = 0
            # originally it's training with lr 0.01 for 100 epochs and then 0.005 for 50 epochs
            self.prepare_train_setting(learning_rate = 2e-4,  epochs = first_stage_epoch , optim = "SGD")
            self.train_model(before_epoch = 0, save_epoch = save_epoch, iter_data = iter_data)

            # self.prepare_train_setting(learning_rate = 1e-5,  epochs = second_stage_epoch, optim = "SGD")
            # self.train_model(before_epoch = first_stage_epoch, save_epoch = save_epoch, shadow_or_attack = shadow_or_attack)
            
        #    self.prepare_train_setting(learning_rate = 0.001,  epochs = third_stage_epoch, optim = "Adam")
        #    self.train_model(before_epoch = first_stage_epoch + second_stage_epoch, save_epoch = save_epoch, shadow_or_attack = shadow_or_attack)
        else:
            raise Exception("Invalid dataset in config file!")

    def get_regularized_curvature_for_batch(self, batch_data, batch_labels, h=1e-3, 
                                            niter=10, temp=1):
        
        num_samples = batch_data.shape[0]
        self.model.eval()
        regr = torch.zeros(num_samples)
        eigs = torch.zeros(num_samples)
        
        for _ in range(niter):
            v = torch.randint_like(batch_data, high=2).to(self.device)
            # Generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            v = h * (v + 1e-7)

            batch_data.requires_grad_()
            outputs_pos = self.model(batch_data + v)
            outputs_orig = self.model(batch_data)
            loss_pos = self.criterion(outputs_pos / temp, batch_labels)
            loss_orig = self.criterion(outputs_orig / temp, batch_labels)
            grad_diff = torch.autograd.grad((loss_pos-loss_orig), batch_data )[0]

            regr += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()
            eigs += torch.diag(torch.matmul(v.reshape(num_samples,-1), grad_diff.reshape(num_samples,-1).T)).cpu().detach()
            self.model.zero_grad()
            if batch_data.grad is not None:
                batch_data.grad.zero_()

        curv_estimate = eigs / niter
        regr_estimate = regr / niter
        return curv_estimate, regr_estimate
    
    def get_dataset_curvature_score_epoch(self, dataset_len, model_path, train_loader):
        
        # read model from existing files
        self.initialize_model(shadow_model=False, model_load_path=model_path, 
                              pretrained=False, freeze_conv_layer=False)
        self.prepare_train_setting(learning_rate=None, epochs=None)

        scores = torch.zeros(dataset_len)
        regr_score = torch.zeros_like(scores)
        labels = torch.zeros_like(scores, dtype=torch.long)
        
        self.model.eval()
        self.model = self.model.to(self.device)

        total = 0
        dataloader = self.train_loader if train_loader else self.test_loader
        
        for (inputs, targets, idxs) in dataloader:
            total = total + len(idxs)

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad = True
            self.model.zero_grad()
            
            curv_estimate, regr_estimate = self.get_regularized_curvature_for_batch(inputs, targets, niter=10)
            scores[idxs] = curv_estimate.detach().clone().cpu()
            regr_score[idxs] = regr_estimate.detach().clone().cpu()
            labels[idxs] = targets.cpu().detach()

        scores = scores.numpy()
        regr_score = regr_score.numpy()

        return scores, regr_score

    def get_dataset_curvature_score(self, logger, train_loader = True, demo = False):

        model_folder_name = gen_save_name(dataset_name=self.dataset_name, model_name=self.model_name,
                                       unlearning_data_selection=self.unlearning_data_selection, 
                                       unlearning_proportion = self.unlearning_proportion,
                                       return_save_folder = True)
        model_folder = os.path.join(self.model_save_path, model_folder_name)
        
        fnmatch = gen_save_name(dataset_name=self.dataset_name, model_name=self.model_name,
                                unlearning_data_selection=self.unlearning_data_selection, 
                                unlearning_proportion = self.unlearning_proportion,
                                unlearning = self.unlearning, shadow_or_attack = 0,
                                epoch = 4, fnmatch=False, short=True ) + ".pt" 
        # set epoch=16, fnmatch=False to read the 16 epoch model. Otherwise there should be all epoch*.pt files
        
        self.logger.info("Fnmatch: %s" % fnmatch)
        model_file_list = get_files_with_wildcard(path = model_folder, wildcard=fnmatch)
        print("model_file_list for curvature calculation:", model_file_list)
        self.logger.info("Number of models for calculating curvature: %d" % len(model_file_list))
        
        dataset_len = len(self.train_dataset) if train_loader else len(self.test_dataset)
        model_num = len(model_file_list)
        scores = np.zeros((model_num, dataset_len))
        regr_score = np.zeros((model_num, dataset_len))
        
        if(demo):
            model_file_list = model_file_list[:1]
        else:
            model_file_list = model_file_list[:min(model_num, 15)]

        t1 = time()

        for i, model_file in enumerate(model_file_list):
            model_path = os.path.join(model_folder, model_file)
            _scores, _regr_score = self.get_dataset_curvature_score_epoch(dataset_len, model_path, train_loader)
            scores[i] = _scores
            regr_score[i] = _regr_score
        
        t2 = time()

        # Get average score
        scores = np.average(scores, axis = 0).tolist()
        regr_score = np.average(regr_score, axis = 0).tolist()

        if(logger):
            logger.info("Calculating %s timing: %.4f" % ("curvature", t2-t1))
            logger.info(get_statistics(scores))
            logger.info(get_statistics(regr_score))

        return scores, regr_score

    # save: "Model-Confidence","Loss-Curvature" such information should be saved for later selection.
    # When calculating the Loss-Curvature, the inner and outer loops of the training process are different from the normal training process, 
    # i.e., the outer loop iterates over the data and the inner loop iterates over the training rounds. See samples for calculating Loss-Curvature(score_imagenet_checkpoint.py).
    # save model when training is done.
    