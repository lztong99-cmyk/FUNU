# Save dataset D*, Dr, Dt for verification.
# Split dataset based on unlearning data selection.
import os
import logging
import torch
import torchvision
import torchvision.transforms as transforms
from time import time
from torch.utils.data import Dataset
from PIL import Image
import random
from utils import get_attack_dataset_with_shadow, read_CIFAR_data, default_loader, \
            make_dataset_tiny_imagenet, get_shard_slice_dataset_idx, SHARD_NUM, SLICE_NUM, \
            get_sim_matrix_theta_typical_indices_path, CLUSTERING_THRES, CURVATURE_THRES, \
            CONFIDENCE_THRES, get_clustering_curvature_confidence_path, locate_value_in_tensor, \
            normalize_tensor, get_tensor_intersection

class MyDataset(Dataset):
    """
    dataset_type: train/remained/unlearned/test data.
    unlearning: define whether the dataset is for original training(0), unlearning/retraining(1),
                attack for original training (2) or attack for unlearning/retraining (3) 
    sisa_selection_op: the order of data slices for sisa experiment. The default number of data 
                slices is 50, thus sisa_selection_op ranges from [0, 49]
    unlearning_filter: the method to filter unlearning requests. option: ["clustering", "curvature",
        "confidence","rfmodel"]
    """
    def __init__(self, dataset_name, dataset_path, 
                 unlearning_data_selection, unlearning_proportion,
                 dataset_type = "train", unlearning = 0,
                 original_proportion = 0.6, left = False, demo = False, 
                 sisa_selection_op = -1, unlearning_data_idx = None,
                 unlearning_filter = None, rf_model_name = None, 
                 sim_theta = -1, sim_alpha = -1, score_thres_dict = {},
                 remove_class = -1):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        # Logging
        self.logger = logging.getLogger('dataset_log')

        # Unlearning setting. Related to Du, Dr and Dt split.
        self.unlearning = unlearning
        
        # set class_num
        class_num_dict = {"MNIST":10, "CIFAR10":10, "CIFAR100":100, "Tiny-ImageNet": 200}
        self.class_num = class_num_dict[self.dataset_name]
        
        self.score_thres_dict = score_thres_dict

        if(dataset_type == "unlearned" or dataset_type == "remained"):
            assert unlearning

        if(self.unlearning == 0):
            # original training scenario
            self.data, self.targets = self.get_original_dataset(dataset_type = dataset_type, 
                                                                original_proportion = original_proportion, left = left,
                                                                demo = demo, sisa_selection_op = sisa_selection_op)
        elif(self.unlearning == 1):
            # unlearning scenario
            if(dataset_type == "test"):
                self.data, self.targets = self.get_original_dataset(dataset_type = dataset_type,
                                                                    original_proportion = original_proportion, left = left,
                                                                    demo = demo, sisa_selection_op = sisa_selection_op)
            elif(dataset_type == "remained"):
                # remained dataset
                self.original_data, self.original_targets = self.get_original_dataset(dataset_type ="original",
                                                                                      original_proportion = original_proportion, left = left,
                                                                                      demo = demo, sisa_selection_op = sisa_selection_op)
                self.data, self.targets = self.split_unlearned_remained_dataset(dataset_type = "remained", unlearned_sample_idx = unlearning_data_idx,
                                                                                unlearning_proportion = unlearning_proportion,
                                                                                unlearning_data_selection = unlearning_data_selection,
                                                                                unlearning_filter = unlearning_filter, rf_model_name = rf_model_name, 
                                                                                sim_theta = sim_theta, sim_alpha = sim_alpha,
                                                                                remove_class = remove_class)
            elif(dataset_type == "unlearned"):
                # unlearned dataset
                self.original_data, self.original_targets = self.get_original_dataset(dataset_type="original",
                                                                                      original_proportion = original_proportion, left = left,
                                                                                      demo = demo, sisa_selection_op = sisa_selection_op)
                self.data, self.targets = self.split_unlearned_remained_dataset(dataset_type = "unlearned", unlearned_sample_idx = unlearning_data_idx,
                                                                                unlearning_proportion = unlearning_proportion,
                                                                                unlearning_data_selection = unlearning_data_selection,
                                                                                unlearning_filter = unlearning_filter, rf_model_name = rf_model_name, 
                                                                                sim_theta = sim_theta, sim_alpha = sim_alpha,
                                                                                remove_class = remove_class)
        elif(self.unlearning == 2):
            # attack for original training
            if(dataset_type == "train" ):
                shadow_train_data_label = self.get_original_dataset(dataset_type = "train", 
                                                                original_proportion = original_proportion, left = True,
                                                                demo = demo, sisa_selection_op = sisa_selection_op)
                shadow_test_data_label = self.get_original_dataset(dataset_type = "test", 
                                                                original_proportion = original_proportion, left = True,
                                                                demo = demo, sisa_selection_op = sisa_selection_op)
                #get_attack_dataset_with_shadow_v0(target_train_data, target_test_data, shadow_train_data, shadow_test_data)
                
                self.data, self.targets, self.member = get_attack_dataset_with_shadow(shadow_train_data_label, shadow_test_data_label)

            elif(dataset_type == "test"):
                # test dataset for the attack model
                target_train_data_label = self.get_original_dataset(dataset_type = "train", 
                                                                original_proportion = original_proportion, left = False,
                                                                demo = demo, sisa_selection_op = sisa_selection_op)
                
                target_test_data_label = None
                self.data, self.targets, self.member = get_attack_dataset_with_shadow(target_train_data_label, 
                                                                                      target_test_data_label,
                                                                                      sample = False)

            elif(dataset_type == "unlearned"):
                1

        elif(self.unlearning == 3):
            # attack for unlearning/retraining
            if(dataset_type == "train"):
                self.original_data, self.original_targets = self.get_original_dataset(dataset_type ="original",
                                                                                      original_proportion = original_proportion, left = True, 
                                                                                      demo = demo, sisa_selection_op = sisa_selection_op)
                shadow_train_data_label = self.split_unlearned_remained_dataset(dataset_type = "remained", unlearned_sample_idx = unlearning_data_idx,
                                                                                unlearning_proportion = unlearning_proportion,
                                                                                unlearning_data_selection=unlearning_data_selection)

                shadow_test_data_label = self.get_original_dataset(dataset_type = "test", 
                                                                original_proportion = original_proportion, left = True,
                                                                demo = demo, sisa_selection_op = sisa_selection_op)
                
                self.data, self.targets, self.member = get_attack_dataset_with_shadow(shadow_train_data_label, shadow_test_data_label,
                                                                                      sample = True, echo = True)
            elif(dataset_type == "test"):
                self.original_data, self.original_targets = self.get_original_dataset(dataset_type ="original",
                                                                                      original_proportion = 1 - original_proportion, left = False, # here we use 1-original so that it is consistent with original training dataset
                                                                                      demo = demo, sisa_selection_op = sisa_selection_op)
                
                # unlearnd dataset and remained dataset that compose the original dataset would make up the test dataset for attack
                target_train_data_label = self.split_unlearned_remained_dataset(dataset_type = "remained", unlearned_sample_idx = unlearning_data_idx,
                                                                                unlearning_proportion = unlearning_proportion,
                                                                                unlearning_data_selection=unlearning_data_selection)

                target_test_data_label = self.split_unlearned_remained_dataset(dataset_type = "unlearned", unlearned_sample_idx = unlearning_data_idx,
                                                                               unlearning_proportion = unlearning_proportion,
                                                                               unlearning_data_selection=unlearning_data_selection)
                
                # Here sample = False indicate that we would not tailor test dataset to make sure classes are balanced
                self.data, self.targets, self.member = get_attack_dataset_with_shadow(target_train_data_label, 
                                                                                      target_test_data_label,
                                                                                      sample = True,
                                                                                      echo = True)
            elif(dataset_type == "unlearned"):
                1

        self.logger.info("%s dataset length: %d" %(dataset_type, len(self.data)))


        # set input_channel
        input_channel_dict = {"MNIST":1, "CIFAR10":3, "CIFAR100":3, "Tiny-ImageNet": 3}
        self.input_channel = input_channel_dict[self.dataset_name]

        # set transform function
        if(self.dataset_name == "MNIST"):
            self.transform = torchvision.transforms.ToTensor()
            self.target_transform = None
        elif(self.dataset_name in ["CIFAR10", "CIFAR100"]):
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.target_transform = None
        elif(self.dataset_name in ["Tiny-ImageNet"]):
            normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
            self.transform = transforms.Compose(
                [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                normalize, ])            
            self.target_transform = None

        else:
            raise NotImplementedError

    def get_original_dataset(self, dataset_type = "test", original_proportion = 0.6, left = False, 
                             demo = False, sisa_selection_op = -1):
        """
        Return orignal train or test dataset with format (data, label)
        """

        if(self.dataset_name == "MNIST"):
            if(dataset_type == "test"):
                ret_data, ret_target = torch.load(os.path.join(self.dataset_path, self.dataset_name, "processed", "test.pt"))
            elif(dataset_type == "train" or dataset_type == "original"):
                ret_data, ret_target = torch.load(os.path.join(self.dataset_path, self.dataset_name, "processed", "training.pt"))
        elif(self.dataset_name in [ "CIFAR10" , "CIFAR100"]):
            tmp_dataset_type = dataset_type
            if(dataset_type in ["original", "train"]):
                tmp_dataset_type = "train"
            ret_data, ret_target = read_CIFAR_data(dataset_path = self.dataset_path, 
                                                   dataset_name = self.dataset_name,
                                                   data_type = tmp_dataset_type)
        elif(self.dataset_name == "Tiny-ImageNet"):
            tmp_dataset_type = dataset_type
            if(dataset_type in ["original", "train"]):
                tmp_dataset_type = "train"
            data_path = os.path.join(self.dataset_path, "tiny-imagenet-200", tmp_dataset_type)
            ret_data, ret_target = make_dataset_tiny_imagenet(directory = data_path, extensions=["jpeg"])
        else:
            raise NotImplementedError

        if(original_proportion != 1 and dataset_type != "test"):
            cut_point = int(len(ret_data) * original_proportion)
            if(left == False):
                ret_data, ret_target = ret_data[:cut_point], ret_target[:cut_point]
            else:
                ret_data, ret_target = ret_data[cut_point:], ret_target[cut_point:]

        # record dataset length for sisa use
        self.full_training_dataset_len_sisa = len(ret_data)

        if(demo):
            return ret_data[:500], ret_target[:500]
        
        if(sisa_selection_op > -1 and dataset_type != "test"):
            assert sisa_selection_op <= SHARD_NUM * SLICE_NUM
            lower,higher = get_shard_slice_dataset_idx(sisa_selection_op, self.full_training_dataset_len_sisa)
            return ret_data[lower:higher], ret_target[lower:higher]

        return ret_data, ret_target

    def filter_unlearning_request_by_score(self, unlearned_sample_indices, model_name, 
                                           score_type = "clustering"):
        score_path = get_clustering_curvature_confidence_path(dataset_name = self.dataset_name, 
                                                              model_name = model_name, score_type = score_type)
        score = torch.load(score_path).squeeze().cpu()
        
        self.logger.info("score BEFORE NORMALIZATION min %.4f, max %.4f, mean %.4f " % (torch.min(score), torch.max(score), torch.mean(score)) )
        
        if(score_type in ["clustering", "curvature"]):
            score = normalize_tensor(score, reverse=False)
        elif(score_type == "confidence"):
            score = normalize_tensor(score, reverse=True)
            
        self.logger.info("score AFTER NORMALIZATION min %.4f, max %.4f, mean %.4f " % (torch.min(score), torch.max(score), torch.mean(score)) )
            
        unlearned_sample_score = torch.index_select(input = score, dim = 0, index = torch.tensor(unlearned_sample_indices))
        
        original_unlearn_len = len(unlearned_sample_indices)

        if(len(self.score_thres_dict)):
            thres_dict = self.score_thres_dict
        else:
            thres_dict = {"clustering": CLUSTERING_THRES, "curvature": CURVATURE_THRES, "confidence": CONFIDENCE_THRES}
        
        if(thres_dict[score_type] < 0):
            # Calculate threshold automaticly. thres_dict[score_type] = -1, use mean+std. thres_dict[score_type] = -2, use mean. thres_dict[score_type] = -3, use mean-std.
            if(thres_dict[score_type] == -1):
                thres_dict[score_type] = torch.mean(score) + torch.std(score)
            elif(thres_dict[score_type] == -2):
                thres_dict[score_type] = torch.mean(score)
            elif(thres_dict[score_type] == -3):
                thres_dict[score_type] = max(torch.mean(score) - torch.std(score), 0.001)
            
        unlearned_sample_indices = torch.tensor(unlearned_sample_indices)
        filtered_indices_by_score = unlearned_sample_score <= thres_dict[score_type]
        filtered_unlearned_sample_indices = unlearned_sample_indices[filtered_indices_by_score]

        self.logger.info("Score_type: %s, threshold: %.3f, original unlearned sample length %d reduces to %d" % (score_type, thres_dict[score_type], \
                        original_unlearn_len, len(filtered_unlearned_sample_indices)))
        return filtered_unlearned_sample_indices.tolist()
    
    def select_by_class(self, cur_class = 1, sample = 0.5):
        
        if(torch.is_tensor(self.original_targets) == False):
            self.original_targets = torch.from_numpy(self.original_targets)
        
        unlearned_sample_indices = None
        
        if(unlearned_sample_indices != None):
            unlearned_sample_indices = torch.cat([unlearned_sample_indices, locate_value_in_tensor(self.original_targets, cur_class)])
        else:
            unlearned_sample_indices = locate_value_in_tensor(self.original_targets, cur_class)
        
            self.logger.info("%d samples in class %d" % (len(unlearned_sample_indices), cur_class))
                
        if(sample <= 1):
            tailered_length = int(len(unlearned_sample_indices) * sample)
        else:
            tailered_length = sample
        return unlearned_sample_indices.tolist()[:tailered_length]
        
    def split_unlearned_remained_dataset(self, dataset_type, unlearned_sample_idx = None, 
                                         unlearning_proportion = 0.1,
                                         unlearning_data_selection = "Random",
                                         unlearning_filter = None, rf_model_name = None, 
                                         sim_theta = -1, sim_alpha = -1, remove_class = 2):
        
        original_data_len = len(self.original_data)
        original_sample_indices = list(range(original_data_len))

        
        if(unlearned_sample_idx):
            unlearned_sample_indices = unlearned_sample_idx
        else:
            if(unlearning_data_selection == "Random"):
                random.seed(123)
                if(unlearning_proportion < 1):
                    unlearned_sample_len = int(unlearning_proportion * original_data_len)
                else:
                    unlearned_sample_len = unlearning_proportion
                unlearned_sample_indices = random.sample(original_sample_indices, unlearned_sample_len)
            elif(unlearning_data_selection == "Byclass"):
                unlearned_sample_indices = self.select_by_class(cur_class = remove_class, sample = unlearning_proportion)
            else:
                raise Exception("Invalid data selection method %s!" % unlearning_data_selection)
        
        # unlearning filter
        t1 = time()
        
        if(unlearning_filter in ["clustering", "confidence", "curvature"]):
            unlearned_sample_indices = self.filter_unlearning_request_by_score(unlearned_sample_indices = unlearned_sample_indices,
                                                                                model_name = rf_model_name, score_type = unlearning_filter)
        elif(unlearning_filter == "rfmodel"):
            unlearned_sample_indices = self.filter_unlearning_request_by_rfmodel(unlearned_sample_indices = unlearned_sample_indices, 
                                                      rf_model_name = rf_model_name, theta = sim_theta, alpha = sim_alpha)
        
        """
        for cur_class in range(10):
            if(unlearned_sample_idx):
                unlearned_sample_indices = unlearned_sample_idx
            else:
                if(unlearning_data_selection == "Random"):
                    # random.seed(123) # before: 123
                    if(unlearning_proportion <= 1):
                        unlearned_sample_len = int(unlearning_proportion * original_data_len)
                    else:
                        unlearned_sample_len = unlearning_proportion
                    unlearned_sample_indices = random.sample(original_sample_indices, unlearned_sample_len)
                elif(unlearning_data_selection == "Byclass"):
                    unlearned_sample_indices = self.select_by_class(cur_class = cur_class, sample = unlearning_proportion)
                else:
                    raise Exception("Invalid data selection method %s!" % unlearning_data_selection)
            
            # unlearning filter
            t1 = time()
            
            if(unlearning_filter in ["clustering", "confidence", "curvature"]):
                unlearned_sample_indices = self.filter_unlearning_request_by_score(unlearned_sample_indices = unlearned_sample_indices,
                                                                                    model_name = rf_model_name, score_type = unlearning_filter)
            elif(unlearning_filter == "rfmodel"):
                unlearned_sample_indices = self.filter_unlearning_request_by_rfmodel(unlearned_sample_indices = unlearned_sample_indices, 
                                                        rf_model_name = rf_model_name, theta = sim_theta, alpha = sim_alpha)
            # else:
            #    raise AttributeError("Invalid unlearning filter %s" % unlearning_filter)
        """    
        t2 = time()

        if(unlearning_filter):
            self.logger.info("unlearning filter %s timing: %.4f s." % (unlearning_filter, t2 - t1))

        if(len(unlearned_sample_indices)<200):
            self.logger.info("unlearned sample indices: %s" % str(unlearned_sample_indices))
        
        remained_sample_indices = [x for x in original_sample_indices if x not in unlearned_sample_indices]
        
        # test = [x for x in unlearned_sample_indices if x not in original_sample_indices]
        
        assert len(remained_sample_indices) + len(unlearned_sample_indices) == original_data_len
        
        if(dataset_type == "unlearned"):
            return self.original_data[unlearned_sample_indices], self.original_targets[unlearned_sample_indices]
        elif(dataset_type == "remained"):
            # MNIST: <class 'torch.Tensor'> torch.Size([36000, 28, 28]) <class 'list'>
            # torch.Size([36000]) <class 'list'> label type: <class 'torch.Tensor'>
            # CIFAR-10: <class 'numpy.ndarray'> (30000, 32, 32, 3) <class 'list'>
            # label type:<class 'list'>
            return self.original_data[remained_sample_indices], self.original_targets[remained_sample_indices]
    
    def filter_unlearning_request_by_rfmodel(self, unlearned_sample_indices, rf_model_name = "2-layer-CNN", 
                                             theta = -1, alpha = None, untypical = False, coe_alpha = False, filter_by_label = True):
        """
        alpha: the number of similar sample count.
        theta: the threshold to decide whether two samples are similar.
        """
        full_similarity_matrix_path, typical_dataset_indices_path, theta_path, alpha_path = get_sim_matrix_theta_typical_indices_path(self.dataset_name, rf_model_name)
        full_similarity_matrix = torch.load(full_similarity_matrix_path)
        
        # self.logger.info("theta: %s, alpha: %s" % (theta, alpha))
        
        if(coe_alpha):
            typical_dataset_indices = torch.load(typical_dataset_indices_path)
            typical_len = len(typical_dataset_indices)
        
        original_unlearn_len = len(unlearned_sample_indices)
        # indexing sub-matrix
        original_data_len = len(self.original_data)
        original_sample_indices = list(range(original_data_len))
        remained_sample_indices = torch.tensor([x for x in original_sample_indices if x not in unlearned_sample_indices])
        
        if(coe_alpha):
            alpha = alpha *  (original_data_len - original_unlearn_len) / typical_len
            self.logger.info("edited alpha: %.4f" % alpha)
            
        if(not alpha):
            typical_dataset_indices = torch.load(typical_dataset_indices_path)
            
            # get alpha
            # return the indices of co-occur samples at remained_sample_indices
            typical_remained_sample_indices = remained_sample_indices[torch.where(torch.eq(remained_sample_indices[:, None], typical_dataset_indices))[0]]
            typical_remained_sample_submatrix = torch.index_select(input = full_similarity_matrix, dim = 1, index = typical_remained_sample_indices)
            
            unlearned_sample_submatrix = torch.index_select(input = typical_remained_sample_submatrix, dim = 0, index = unlearned_sample_indices)
            typical_remained_sample_submatrix = torch.index_select(input = typical_remained_sample_submatrix, dim = 0, index = typical_remained_sample_indices)
            alpha = torch.mean(torch.sum(typical_remained_sample_submatrix >= theta, dim = 1).float())

            self.logger.info("length of remained typical samples: %d, alpha: %.4f" % (len(typical_remained_sample_indices), alpha))

            # filter by alpha
            unlearn_sample_count = torch.sum(unlearned_sample_submatrix <= theta, dim = 1)
        else:
            # alpha has already been generated when grouping data by label
            if(filter_by_label == False):
                unlearned_to_remained_data_matrix = torch.index_select(input = full_similarity_matrix, dim = 0, index = torch.tensor(unlearned_sample_indices))
                
                unlearned_to_remained_data_matrix = torch.index_select(input = unlearned_to_remained_data_matrix, dim = 1, index = torch.tensor(remained_sample_indices))
                
                unlearn_sample_count = torch.sum(unlearned_to_remained_data_matrix >= theta, dim = 1)
            else:
                unlearn_sample_count = torch.zeros(len(unlearned_sample_indices)).long()
                filtered_unlearned_sample_indices = []
                
                if(isinstance(theta, list)):
                    assert len(theta) == self.class_num
                    assert isinstance(alpha, list)
                    assert len(alpha) == self.class_num
                    
                for label in range(self.class_num):
                    label_index = locate_value_in_tensor(self.original_targets, torch.tensor(label))
                    tmp_unlearned_index = get_tensor_intersection(label_index, unlearned_sample_indices)
                    tmp_remained_index = get_tensor_intersection(label_index, remained_sample_indices)
                    
                    if(len(tmp_unlearned_index) == 0):
                        continue
                    
                    cur_data_similarity_matrix = torch.index_select(input = full_similarity_matrix, 
                                                    dim = 0, index = tmp_unlearned_index)
                    cur_data_similarity_matrix = torch.index_select(input = cur_data_similarity_matrix, 
                                                    dim = 1, index = tmp_remained_index)
                    
                    #print("cur_data_similarity_matrix.shape", cur_data_similarity_matrix.shape)
                    
                    # if the type of theta is list
                    if(isinstance(theta, list)):
                        cur_unlearn_sample_count = torch.sum(cur_data_similarity_matrix >= theta[label], dim = 1).long()
                    else:
                        cur_unlearn_sample_count = torch.sum(cur_data_similarity_matrix >= theta, dim = 1).long()
                    
                    assert len(cur_unlearn_sample_count) == len(tmp_unlearned_index)
                    
                    if(isinstance(alpha, list)):
                        cur_unlearnd_sample_indices = tmp_unlearned_index[cur_unlearn_sample_count <= alpha[label]]
                    else:
                        cur_unlearnd_sample_indices = tmp_unlearned_index[cur_unlearn_sample_count <= alpha]
                        
                    # print("len(cur_unlearnd_sample_indices): ", len(cur_unlearnd_sample_indices))
                    filtered_unlearned_sample_indices.extend(cur_unlearnd_sample_indices.tolist())
                    #unlearn_sample_count[tmp_unlearned_index] = cur_unlearn_sample_count
                
            """
            if(untypical == False):
                unlearn_sample_count = torch.sum(unlearned_to_remained_data_matrix <= theta, dim = 1)
            else:
                unlearn_sample_count = torch.sum(unlearned_to_remained_data_matrix >= theta, dim = 1)
            """
        
        if(filter_by_label == False):
            assert unlearn_sample_count.size(0) == len(unlearned_sample_indices)
            
            if(len(unlearn_sample_count.size())>1):
                unlearn_sample_count = unlearn_sample_count.squeeze()
            
            if(untypical == False):
                # trying to find those typical samples. "unlearn_sample_count <= alpha" indicates finds samples whose neighbors are less than the given threshold. And thus are typical samples that need to be forgotten.
                filtered_unlearned_sample_indices = torch.tensor(unlearned_sample_indices)[unlearn_sample_count <= alpha]
            else:
                # trying to find those untypical samples
                filtered_unlearned_sample_indices = torch.tensor(unlearned_sample_indices)[unlearn_sample_count >= alpha]
            
        self.logger.info("original unlearned sample length %d reduces to %d" % (original_unlearn_len, len(filtered_unlearned_sample_indices)))

        if(torch.is_tensor(filtered_unlearned_sample_indices) == True):
            return filtered_unlearned_sample_indices.tolist()
        else:
            return filtered_unlearned_sample_indices
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        if(self.unlearning < 2):
            if(self.dataset_name in ["MNIST","CIFAR10","CIFAR100"]):
                img, target = self.data[index], int(self.targets[index])

                # doing this so that it is consistent with all other datasets
                # to return a PIL Image
                if(self.dataset_name == "MNIST"):
                    img = img.numpy()
                    img = Image.fromarray(img, mode="L")
                elif(self.dataset_name in ["CIFAR10","CIFAR100"]):
                    img = Image.fromarray(img)

                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    target = self.target_transform(target)

                return img, target, index
            
            elif(self.dataset_name == "Tiny-ImageNet"):
                path, target = self.data[index], int(self.targets[index])
                sample = default_loader(path)
                
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                return sample, target, index
            # to return an index along with the data: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/2
            # return (data, target, index)
            return img, target, index
        
        elif(self.unlearning >= 2):
            if(self.dataset_name in ["MNIST","CIFAR10","CIFAR100"]):
                img, target, member = self.data[index], int(self.targets[index]), self.member[index]

                # doing this so that it is consistent with all other datasets
                # to return a PIL Image
                if(self.dataset_name == "MNIST"):
                    img = img.numpy()
                    img = Image.fromarray(img, mode="L")
                elif(self.dataset_name in ["CIFAR10","CIFAR100"]):
                    if(torch.is_tensor(img)):
                        img = Image.fromarray(img.numpy())
                    else:
                        img = Image.fromarray(img)
                #img = Image.fromarray(img.numpy(), mode="L")

                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    target = self.target_transform(target)

                return img, target, member