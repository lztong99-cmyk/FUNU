import os
import os.path
import argparse
import yaml
import torch
import random
import numpy as np
import pandas as pd
import pickle
import fnmatch
import sys
import torch.nn.functional as F
import shutil
from PIL import Image
from tqdm import tqdm
from time import time, strftime
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from lib_model.CNN import ConvNet, CNN
from lib_model.ResNet import Identity
from statistics import mean, variance
from torchvision.models.resnet import resnet18, resnet34, resnet50
from pathlib import Path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from torch import nn
from PIL import Image

ORIGINAL_PROPORTION = 0.9
CIFAR_10_BATCH_NUM = 5
SHARD_NUM = 10
SLICE_NUM = 5
STATISTIC_PATH = "../distribution_statistics"
CLUSTERING_THRES = 0.2
CURVATURE_THRES = 0.5
CONFIDENCE_THRES = 0.025
FULL_SIM_MATRIX_NAME = "full_sim_matrix.pt"
TYPICAL_SAMPLE_INDICES_NAME = "typical_sample_indices.pt"
SIM_THETA_NAME = "sim_theta_list.pt"
SIM_ALPHA_NAME = "sim_alpha_list.pt"
REFERENCE_MODEL_EPOCH = 1
CONF_MODEL_EPOCH = 10

def get_shard_slice_num(i):
    """
    return shard_order, slice_order
    """
    return i // SLICE_NUM, i % SLICE_NUM

def get_shard_slice_id_by_sample(i, block_len):

    return i // (block_len*SLICE_NUM), (i // block_len) % SLICE_NUM

def gen_unlearned_index(original_sample_indices, unlearning_data_selection = "Random", 
                        unlearning_proportion = 0.01, bottom = True):
    # if unlearning_proportion > 1, then unlearning_proportion is the number of unlearning requests.
    original_data_len = len(original_sample_indices)

    random.seed(123)
    if(unlearning_data_selection == "Random"):
        if(unlearning_proportion < 1):
            unlearned_sample_len = int(unlearning_proportion * original_data_len)
            unlearned_sample_indices = random.sample(original_sample_indices, unlearned_sample_len)
        else:
            unlearned_sample_indices = random.sample(original_sample_indices, unlearning_proportion)
    elif(unlearning_data_selection == "Series"):
        if(bottom):
            # generate unlearned sample indices in reverse direction (from the bottom)
            unlearned_sample_indices = list(range(original_data_len, original_data_len - int(unlearning_proportion * original_data_len), -1))
        else:
            unlearned_sample_indices = list(range(int(unlearning_proportion * original_data_len)))

    return unlearned_sample_indices

def gen_predefined_unlearned_index(dataset_name = "MNIST", delete_num = 30, filtered = False):
    if(dataset_name == "MNIST"):
        if(delete_num == 30):
            if(filtered == False):
                return [3431, 17542, 5713, 50394, 26688, 17468, 7058, 2500, 24846, 35140, 36852, 21770, 22334, 3401, 10463, 8851, 22099, 36763, 21875, 45985, 16067, 10733, 109, 28591, 50697, 5741, 39164, 24746, 4579, 432]
            else:
                return [109, 432, 24746, 26688, 7058, 36852, 3401, 17542]
        elif(delete_num == 10):
            if(filtered == False):
                return [3431, 17542, 5713, 50394, 26688, 17468, 7058, 2500, 24846, 35140]
            else:
                return [26688, 7058, 17542]
        elif(delete_num == 50):
            if(filtered == False):
                return [3431, 17542, 5713, 50394, 26688, 17468, 7058, 2500, 24846, 35140, 36852, 21770, 22334, 3401, 10463, 8851, 22099, 36763, 21875, 45985, 16067, 10733, 109, 28591, 50697, 5741, 39164, 24746, 4579, 432, 
                        20673, 47974, 29381, 6687, 2876, 6079, 43721, 9317, 8273, 51778, 1391, 19123, 28222, 37578, 31255, 17388, 30755, 2408, 50383, 19994]
            else:
                return [109, 432, 9317, 24746, 26688, 7058, 36852, 3401, 50383, 17542, 28222, 31255]
    elif(dataset_name == "CIFAR10"):
        if(delete_num == 30):
            if(filtered == False):
                return [3431, 17542, 5713, 26688, 17468, 7058, 2500, 24846, 35140, 36852, 21770, 22334, 3401, 10463, 8851, 22099, 36763, 21875, 16067, 10733, 109, 28591, 5741, 39164, 24746, 4579, 432, 20673, 29381, 6687]
            else:
                return [6687, 36763, 20673, 5713, 5741, 10463, 26688, 36852, 7058, 2500, 3431, 8851, 22099, 28591]
        elif(delete_num == 10):
            if(filtered == False):
                return [3431, 17542, 5713, 26688, 17468, 7058, 2500, 24846, 35140, 36852]
            else:
                return [5713, 26688, 36852, 7058, 2500, 3431]
        elif(delete_num == 50):
            if(filtered == False):
                return [3431, 17542, 5713, 26688, 17468, 7058, 2500, 24846, 35140, 36852, 21770, 22334, 3401, 10463, 8851, 22099, 36763, 21875, 16067, 10733, 109, 28591, 5741, 39164, 24746, 4579, 432, 20673, 29381, 6687, 
                        2876, 6079, 43721, 9317, 8273, 1391, 19123, 28222, 37578, 31255, 17388, 30755, 2408, 19994, 22509, 34188, 31651, 13556, 39905, 41861]
            else:
                return [6687, 36763, 2408, 20673, 2876, 5713, 5741, 6079, 10463, 19994, 26688, 36852, 7058, 2500, 3431, 8851, 17388, 22099, 31255, 34188, 28222, 30755, 28591, 43721]
    elif(dataset_name == "CIFAR100"):
        if(delete_num == 30):
            if(filtered == False):
                return [3431, 17542, 5713, 26688, 17468, 7058, 2500, 24846, 35140, 36852, 21770, 22334, 3401, 10463, 8851, 22099, 36763, 21875, 16067, 10733, 109, 28591, 5741, 39164, 24746, 4579, 432, 20673, 29381, 6687]
            else:
                return [20673, 7058, 3431, 29381, 21770, 5713, 10463, 17542, 24746, 36852, 4579, 26688, 432, 21875, 28591, 16067, 8851, 3401, 5741, 35140]
        elif(delete_num == 10):
            if(filtered == False):
                return [3431, 17542, 5713, 26688, 17468, 7058, 2500, 24846, 35140, 36852]
            else:
                return [7058, 3431, 5713, 17542, 36852, 26688, 35140]
        elif(delete_num == 50):
            if(filtered == False):
                return [3431, 17542, 5713, 26688, 17468, 7058, 2500, 24846, 35140, 36852, 21770, 22334, 3401, 10463, 8851, 22099, 36763, 21875, 16067, 10733, 109, 28591, 5741, 39164, 24746, 4579, 432, 20673, 29381, 6687, 
                        2876, 6079, 43721, 9317, 8273, 1391, 19123, 28222, 37578, 31255, 17388, 30755, 2408, 19994, 22509, 34188, 31651, 13556, 39905, 41861]
            else:
                return [20673, 7058, 8273, 3431, 9317, 29381, 21770, 5713, 10463, 17542, 31255, 37578, 39905, 28222, 2876, 19994, 24746, 36852, 4579, 26688, 432, 17388, 21875, 28591, 16067, 34188, 8851, 3401, 5741, 30755, 35140]
    
def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
def find_min_influenced_slices_for_shard(unlearned_sample_indices, original_sample_indices):

    dataset_len = len(original_sample_indices)
    block_len = dataset_len // (SHARD_NUM * SLICE_NUM)
    min_influenced_slices_shard = defaultdict(lambda :SLICE_NUM)

    for idx in unlearned_sample_indices:
        shard_no, slice_no = get_shard_slice_id_by_sample(idx, block_len)
        if(slice_no < min_influenced_slices_shard[shard_no]):
            min_influenced_slices_shard[shard_no] = slice_no
    
    return min_influenced_slices_shard

def get_shard_slice_dataset_idx(i, dataset_len):
    
    block_len = dataset_len // (SHARD_NUM * SLICE_NUM)
    
    shard_no, slice_no = get_shard_slice_num(i)

    lower = (shard_no * SLICE_NUM ) * block_len

    higher = (shard_no * SLICE_NUM + (slice_no + 1)) * block_len
    
    return lower, higher

def map_unlearned_index2slice(i, unlearned_data_index, dataset_len):
    """
    Map the unlearned_data_index in i-th slice with the full length of dataset (dataset_len).
    """
    
    lower, higher = get_shard_slice_dataset_idx(i, dataset_len)
    # print("[map_unlearned_index2slice] unlearned_data_index:", unlearned_data_index, lower, higher)
    unlearned_index_in_slice = [x-lower for x in unlearned_data_index if (x>=lower and x < higher)]

    return unlearned_index_in_slice

def get_tensor_intersection(t1, t2):
    if(torch.is_tensor(t1) == False):
        t1 = torch.tensor(t1)
    if(torch.is_tensor(t2) == False):
        t2 = torch.tensor(t2)
    
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    # difference = uniques[counts == 1]
    intersection = uniques[counts > 1]
    
    return intersection

def normalize_list(result_list):
    ret = np.numpy(result_list)
    return (ret- np.min(ret))/(np.max(ret) - np.min(ret))

def normalize_tensor(result_tensor, reverse = False):
    min_t = torch.min(result_tensor)
    max_t = torch.max(result_tensor)
    if(not reverse):
        return (result_tensor- min_t)/(max_t - min_t)
    else:
        return 1-((result_tensor- min_t)/(max_t - min_t))
    
def freeze_conv_layers(model):
    
    models=model.modules()
    for p in models:
        if p._get_name()!='Linear':
            p.requires_grad_=False
    
    print("Freeze Conv layers.")

def get_dataset_clustering_score(dataset, n_components = 2, min_cluster_size = 5, 
                                 outlier_score = True, logger = None, demo = False):
    """
    return a list composed of dataset clustering score regarding indices order.
    """
    X = dataset.data[:500] if demo else dataset.data

    n_sample = len(X)
    #print(type(X), X.shape)
    X = X.reshape(n_sample, -1)
    
    if(n_components):
        t1 = time()
        X_embeded = TSNE(n_components=n_components, learning_rate='auto',
                        init='random', perplexity=3).fit_transform(X)
        t2 = time()
        if(logger):
            logger.info("TSNE execution timing: %.4f" % (t2-t1))
    else:
        X_embeded = X

    t1 = time()
    hdb = HDBSCAN(min_cluster_size=min_cluster_size) 
    hdb.fit(X_embeded)
    t2 = time()
    if(logger):
        logger.info("HDBSCAN execution timing: %.4f" % (t2-t1))

    t1 = time()
    n_samples = len(dataset)
    result_list = []
    if(outlier_score):
        result_list = hdb.outlier_scores_
    else:
        for i in range(n_samples):
            dist_to_centroid = -1
            if(hdb.labels_[i] >= 0):
                dist_to_centroid = euclidean_distances(X_embeded[i], hdb.centroids_[hdb.labels_[i]])
            result_list.append(dist_to_centroid)
    t2 = time()

    if(logger):
        logger.info("Calculating %s timing: %.4f" % ("outlier_score" if outlier_score else "dist_to_centroid", t2-t1))
        logger.info(get_statistics(result_list))
    
    return result_list

def get_statistics(result_list):
    return "Result min: %.4f, max: %.4f, mean: %.4f, variance: %.4f" % (min(result_list), max(result_list),mean(result_list), variance(result_list))

def get_param_distance(model1, model2, logger, device, p=2):
        """
        To calculate average distance over all parameters.
        p=1: L1 norm
        P=2: L2 norm
        """
        model1 = model1.to(device)
        model2 = model2.to(device)
        
        param_objs1 = []
        for module in model1.modules():
            for (name, param)  in module.named_parameters():
                param_objs1.append(param.data)

        param_objs2 = []
        for module in model2.modules():
            for (name, param)  in module.named_parameters():
                param_objs2.append(param.data)
        
        if(len(param_objs1)!=len(param_objs2)):
            raise Exception("Two models are of different scales!")
            
        dist = 0
        cnt = 0
        for i in range(len(param_objs1)):
            cnt += param_objs1[i].numel()
            if(len(param_objs1[i].shape)==1):
                param_objs1_t = param_objs1[i].unsqueeze(0)
                param_objs2_t = param_objs2[i].unsqueeze(0)
            else:
                param_objs1_t = param_objs1[i]
                param_objs2_t = param_objs2[i]
            dist += torch.nn.functional.pairwise_distance(param_objs1_t, param_objs2_t, p=p, eps=1e-06).sum()
        
        avg_dist = dist/cnt
        
        logger.info("Parameter distance (%d norm): %.5f" % (p, avg_dist))
        
        return avg_dist

def gen_save_name(dataset_name, model_name, unlearning_data_selection, unlearning_proportion, 
                  epoch = 0 ,distribution_mining_exp = False, unlearning = False, 
                  shadow_or_attack = 0, fnmatch = False, return_save_folder = False,
                  short = False, sisa_exp = False, filter_name = None):
    """
    Generate filename for log and saved models.
    The filename is composed of dataset, model and unlearning information.
    shadow_or_attack: 1 for shadow model, 2 for attack model.
    """
    if(distribution_mining_exp):
        return "distribution_mining-" + strftime("%m-%d_%H-%M")

    if(sisa_exp):
        return "SISA_exp-" + strftime("%m-%d_%H-%M")
    
    model_name_part = [dataset_name, model_name]

    appendix_dict = {0:"model", 1:"shadow", 2:"attack"}
    if(unlearning):
        model_name_part.extend([unlearning_data_selection, str(unlearning_proportion)])
        model_name_part.extend([appendix_dict[shadow_or_attack]])
    elif(unlearning == 0):
        model_name_part.append("original")
        model_name_part.extend([appendix_dict[shadow_or_attack]])
    
    if(return_save_folder):
        return "-".join(model_name_part)

    if(short):
        model_name_part = []

    if(fnmatch):
        model_name_part.append("epoch*")
    elif(epoch):
        model_name_part.append("epoch%s" % epoch)
        
    if(filter_name):
        model_name_part.append(filter_name)

    model_name = "-".join(model_name_part)
    return model_name

def get_class_num(dataset_name):
    if(dataset_name == "MNIST" or dataset_name == "CIFAR10"):
        return 10
    elif(dataset_name == "CIFAR100"):
        return 100
    elif(dataset_name == "ImageNet"):
        return 1000
    else:
        raise Exception("Invalid dataset %s!" % dataset_name)
    
def get_files_with_wildcard(path, wildcard):
    files = os.listdir(path)
    filtered_files = [file for file in files if fnmatch.fnmatch(file, wildcard)]
    return filtered_files

def get_attack_dataset_with_shadow_v0(target_train, target_test, shadow_train, shadow_test):
    mem_train, nonmem_train = list(shadow_train), list(shadow_test), 
    mem_test, nonmem_test = list(target_train), list(target_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)

    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test), len(nonmem_test))

    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
    mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
    non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
    
    attack_train = mem_train + non_mem_train
    attack_test = mem_test + non_mem_test

    return attack_train, attack_test

def get_attack_dataset_with_shadow(train_data_label, test_data_label, sample = True, echo = False):
    mem_train = list(train_data_label)
    if( test_data_label ):
        nonmem_train = list(test_data_label)
    else:
        nonmem_train = []
    
    if(echo):
        print("len(mem_train), len(nonmem_train):", len(mem_train[0]), len(nonmem_train[0]))
    
    if(sample):
        train_length = min(len(mem_train[0]), len(nonmem_train[0]))

        original_mem_indices, original_nonmem_indices = list(range(len(mem_train[0]))), list(range(len(nonmem_train[0])))

        selected_mem_indices = random.sample(original_mem_indices, train_length)
        selected_nonmem_indices = random.sample(original_nonmem_indices, train_length)
    else:
        selected_mem_indices= list(range(len(mem_train[0])))
        
        if(len(nonmem_train)):
            selected_nonmem_indices = list(range(len(nonmem_train[0])))
        else:
            selected_nonmem_indices = None
    
    if(echo):
        print("mem_train[0][selected_mem_indices].shape, mem_train[1][selected_mem_indices].shape:", 
            mem_train[0][selected_mem_indices].shape, mem_train[1][selected_mem_indices].shape)
        print("nonmem_train[0][selected_nonmem_indices].shape, nonmem_train[1][selected_nonmem_indices].shape:", 
            nonmem_train[0][selected_nonmem_indices].shape, nonmem_train[1][selected_nonmem_indices].shape)
    
    if(len(nonmem_train)):
        mem_train_data = mem_train[0][selected_mem_indices]
        nonmem_train_data = nonmem_train[0][selected_nonmem_indices]
        mem_train_label = mem_train[1][selected_mem_indices]
        nonmem_train_label = nonmem_train[1][selected_nonmem_indices]
        
        if(not torch.is_tensor(mem_train_data)):
            mem_train_data = torch.from_numpy(mem_train_data)
        if(not torch.is_tensor(nonmem_train_data)):
            nonmem_train_data = torch.from_numpy(nonmem_train_data)
        if(not torch.is_tensor(mem_train_label)):
            mem_train_label = torch.from_numpy(mem_train_label)
        if(not torch.is_tensor(nonmem_train_label)):
            nonmem_train_label = torch.from_numpy(nonmem_train_label)    
        
        ret_data = torch.cat([mem_train_data, nonmem_train_data], dim=0)
        ret_label = torch.cat([mem_train_label, nonmem_train_label], dim=0)
    else:
        ret_data = mem_train[0][selected_mem_indices]
        ret_label = mem_train[1][selected_mem_indices]
        
    data_len = len(ret_data)
    ret_member = torch.zeros(data_len)
    ret_member[:len(mem_train_data)] = 1

    if(echo):
        print("ret_data.shape, ret_label.shape, ret_member.shape: ", ret_data.shape, ret_label.shape, ret_member.shape)
        print("ret_member.sum: ", torch.sum(ret_member))
    
    # the return value includes data and label
    return ret_data, ret_label, ret_member

def model_selection(model_name, class_num, input_channel = 1, shadow_model = False, pretrained = False):
    #if(shadow_model == True and "CNN" in model_name):
    #    assert input_channel
    #    return CNN(input_channel=input_channel, num_classes=class_num)
            
    if(model_name == "2-layer-CNN"):
        return ConvNet(layer=2, num_classes=class_num)
    elif(model_name == "3-layer-CNN"):
        return ConvNet(layer=3, num_classes=class_num)
    elif(model_name == "4-layer-CNN"):
        return ConvNet(layer=4, num_classes=class_num)
    elif(model_name == "ResNet-18"):
        """
        pretrained = 0, means there is no pretrained weights loaded.
        """
        net = resnet18(progress=False, pretrained=pretrained)
        net.fc = nn.Linear(512, class_num)
        return net
        #return ResNet18(class_num)
    elif(model_name == "ResNet-34"):
        return resnet34(progress=False, pretrained=pretrained)
        #return ResNet34(class_num)
    elif(model_name == "ResNet-50"):
        return resnet50(progress=False, pretrained=pretrained)
        #return ResNet50(class_num)
    else:
        raise Exception("Invalid model %s!" % model_name)
        
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parameter_parser():
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--config', type=str, default='../config/test.yml')
    
    parser_dict = vars(parser.parse_args())
    config_path = parser_dict["config"]
    config = None

    with open(config_path, encoding='utf-8')as file:
        content = file.read()
        config = yaml.load(content, Loader=yaml.FullLoader)

    assert config["dataset"] in ["MNIST", "CIFAR10", "CIFAR100", "Tiny-ImageNet"]
    assert config["model"] in ["2-layer-CNN", "3-layer-CNN", "4-layer-CNN",
                               "ResNet-18", "ResNet-34", "ResNet-50",]
    assert config["unlearning_data_selection"] in ["Random", "Byclass"]
    return config

def unpickle(file, encoding = "bytes"): # Encoding latin1 for CIFAR100, bytes for CIFAR10
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding=encoding)
    return dict

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

def find_classes_tiny_imagenet(directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def default_loader(path: str) -> Any:
    return pil_loader(path)

def make_dataset_tiny_imagenet(
    directory: Union[str, Path],
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    allow_empty: bool = False,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes_tiny_imagenet(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    ret_path = []
    ret_target = []
    available_classes = set()
    
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    #item = path, class_index
                    ret_path.append(path)
                    ret_target.append(class_index)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes and not allow_empty:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return ret_path, ret_target

def read_CIFAR_data(dataset_path, dataset_name, data_type):
    """
    Read CIFAR10 and CIFAR100 dataset from raw files.
    """
    datafolder_name, datafile_list, encoding = get_datafolder_downloadlist_encoding(dataset_name = dataset_name, data_type = data_type)

    data, targets = [], []

    # now load the picked numpy arrays
    for file_name in datafile_list:
        file_path = os.path.join(dataset_path, datafolder_name, file_name)
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding=encoding)

            if(dataset_name == "CIFAR10"):
                data.append(entry[b"data"])
            else:
                data.append(entry["data"])

            if b"labels" in entry:
                targets.extend(entry[b"labels"])
            else:
                targets.extend(entry["fine_labels"])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC

    targets = np.array(targets)
    
    return data, targets

def get_datafolder_downloadlist_encoding(dataset_name = "CIFAR10", data_type = "train"):

    if(dataset_name == "CIFAR10"):
        datafolder_name = "cifar-10-batches-py"
        if(data_type == "train"):
            datafile_list = ["data_batch_1", "data_batch_2", "data_batch_3",
                            "data_batch_4", "data_batch_5"]
        elif(data_type == "test"):
            datafile_list = ["test_batch"]
        encoding = "bytes"
    elif(dataset_name == "CIFAR100"):
        datafolder_name = "cifar-100-python"
        if(data_type == "train"):
            datafile_list = ["train"]
        elif(data_type == "test"):
            datafile_list = ["test"]
        encoding = "latin1"
    else:
        raise Exception("Invalid dataset")
    
    return datafolder_name, datafile_list, encoding

def read_CIFAR10_data(datapath, data_type = "train"):
    
    data, target = None, None
    
    if(data_type == "train"):

        x=[]
        y=[]

        for i in range(1, CIFAR_10_BATCH_NUM + 1):
            batch_path = os.path.join(datapath, "cifar-10-batches-py", 'data_batch_%d'%(i))
            batch_dict = unpickle(batch_path)
            
            x.append(batch_dict[b'data'])
            y.extend(batch_dict[b'labels'])

        data = x # np.concatenate(x)
        target = y #np.concatenate(y)

    elif(data_type == "test"):
        
        testpath = os.path.join(datapath, "cifar-10-batches-py",'test_batch')
        test_dict = unpickle(testpath)
        
        data = test_dict[b'data'].astype('float')
        target = np.array(test_dict[b'labels'])
    else:
        raise Exception("Invalid data type for cifar-10 dataset!")
    
    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))
    
    #print("CIFAR10 data shape and label shape:", data.shape, target.shape)

    return data, target

def cal_cos_similarity_by_loop(X_features, distance = "cos", theta = 0.3, min_similar_samples = 3):
    
    n_sample = len(X_features)
    X_count = np.zeros(n_sample)

    t1 = time()
    # Estimated timing for MNIST dataset: ~300h
    for i in tqdm(range(n_sample)):
        if(X_count[i] >= min_similar_samples):
            continue
        x_feature = X_features[i].reshape(1, -1)
        
        for j in range(i+1, n_sample):
            y_feature = X_features[j].reshape(1, -1)
            if(distance == "cos"):
                dist = cosine_similarity(x_feature, y_feature)
            elif(distance == "euclidean"):
                dist = euclidean_distances(x_feature, y_feature).sum()

            if(dist <= theta):
                X_count[i] += 1
                X_count[j] += 1
    
    X_result = np.zeros(n_sample)
    X_result[ X_count >= min_similar_samples ] = 1
    t2 = time()

    print("cal cos sim by loop timing: %.4f" % (t2 - t1))
    return X_result

def cal_cos_similarity_by_matrix(X_features, logger, theta = 0.1, min_similar_samples = 10):

    X_result = torch.ones(len(X_features))
    
    t1 = time()
    # https://gist.github.com/agtbaskara/e60ac859e0c2d586c94e9acb12800932
    dist_matrix = cosine_similarity(X_features, X_features)
    t2 = time()

    logger.info("calculate similarity matrix timing %.4f s" % (t2 - t1))

    if(theta < 0):
        return dist_matrix

    t1 = time()
    for i in range(len(dist_matrix)):
        if( (dist_matrix[i]<=theta).sum() >= min_similar_samples):
            X_result[i] = 0
    t2 = time()

    logger.info("iterate distance matrix timing: %.4f s, number of 1 in result: %d" % (t2-t1, len(dist_matrix) - X_result.sum()))

    return X_result

def locate_value_in_tensor(input, target_value):
    # return the indices of values in input that equal the target_value.
    # return type: tensor
    if(torch.is_tensor(input) == False):
        input = torch.tensor(input)
    return torch.nonzero(torch.eq(input, target_value)).squeeze()

def get_conf_model_path(dataset_name, model_name, folder = "../models", epoch = CONF_MODEL_EPOCH):
    
    return os.path.join(folder, '-'.join([dataset_name, model_name, "original-model"]), "epoch" + str(epoch) +"-NoFilter.pt")

def get_reference_model_path(dataset_name, model_name, folder = "../models", epoch = REFERENCE_MODEL_EPOCH):
    
    return os.path.join(folder, '-'.join([dataset_name, model_name, "original-model"]), "epoch" + str(epoch) +"-NoFilter.pt")

    if(dataset_name == "MNIST"):
        if(model_name == "2-layer-CNN"):
            return "../models/MNIST-2-layer-CNN-original-model/epoch16.pt"
        elif(model_name == "3-layer-CNN"):
            return "../models/MNIST-3-layer-CNN-original-model/epoch16.pt"
        elif(model_name == "4-layer-CNN"):
            return "../models/MNIST-4-layer-CNN-original-model/epoch16.pt"
    elif(dataset_name == "CIFAR10"):
        if(model_name == "ResNet-18"):
            return "../models/CIFAR10-ResNet-18-original-model/epoch16.pt"
        elif(model_name == "ResNet-34"):
            return "../models/MNIST-3-layer-CNN-original-model/epoch16.pt"
        elif(model_name == "ResNet-50"):
            return "../models/MNIST-4-layer-CNN-original-model/epoch16.pt"

def get_statistic_folder(dataset_name, model_name):
    folder = STATISTIC_PATH
    subfolder = dataset_name + '-' + model_name

    statistic_folder = os.path.join(folder, subfolder)
    if(not os.path.exists(statistic_folder)):
        os.mkdir(statistic_folder)
    
    return statistic_folder

def get_clustering_curvature_confidence_path(dataset_name, model_name, score_type = "clustering"):
    statistic_folder = get_statistic_folder(dataset_name, model_name)

    return os.path.join(statistic_folder, score_type + ".pt" if score_type.endswith(".pt") == False else score_type )

def save_score_to_pt(score, dataset_name, model_name, score_type = "clustering"):

    file_name = get_clustering_curvature_confidence_path(dataset_name, model_name, score_type = score_type)
    score = torch.tensor(score)
    torch.save(score, file_name)
    return file_name

def get_sim_matrix_theta_typical_indices_path(dataset_name, model_name):
    
    statistic_folder = get_statistic_folder(dataset_name, model_name)

    full_sim_matrix_name = FULL_SIM_MATRIX_NAME
    typical_sample_indices_name = TYPICAL_SAMPLE_INDICES_NAME
    theta_name = SIM_THETA_NAME
    alpha_name = SIM_ALPHA_NAME

    return os.path.join(statistic_folder, full_sim_matrix_name), \
            os.path.join(statistic_folder, typical_sample_indices_name), \
            os.path.join(statistic_folder, theta_name), \
            os.path.join(statistic_folder, alpha_name)

def find_samples_similar_count(sub_matrix, theta):
    """
    Count each row of sub_matrix how many samples have similarity above theta
    return a list composed of count.
    """
    return []

def save_content_to_npy(content, filename):
    1

def cal_cos_similarity_by_chunck(input, chunk_num = 2):
    1

def get_presentation_similarity(dataset, dataset_name, logger, theta = 0.1, min_similar_samples = 10,
                                feature_extractor = "ResNet-18", model_pretrained = True,
                                distance = "cos", save_dist_distribution = False, demo = False,
                                ret_dist_matrix = False, presentation = "confidence", feature_data = None):
    
    assert distance == "cos"

    X = dataset.data[:500] if demo else dataset.data
     
    n_sample = len(X)
    
    t1 = time()
    
    if(presentation == "original_feature"):
        if(dataset_name == "MNIST"):
            X = X.reshape(n_sample, -1)
        
        if("CIFAR" in dataset_name):
            X = X.transpose(0, 3, 1, 2)
            X = torch.from_numpy(X).to(torch.float32)

        if(model_pretrained and dataset_name!= "MNIST"):
            # We use the smallest ResNet here.
            if(feature_extractor == "ResNet-18"):
                resnet18_feature_extractor = resnet18(pretrained = model_pretrained)
            # Revise the last layer
                resnet18_feature_extractor.fc = Identity()
            else:
                # If do not use pretrained model, you can use a self-trained 
                # model to generate representation.
                raise NotImplementedError
        
        if(dataset_name!= "MNIST"):
            # CIFAR10, CIFAR100 and Tiny-ImageNet

            # my_model1 = nn.Sequential(*list(pretrained_model.children())[:-1])
            X_features = resnet18_feature_extractor(X)
        else:
            X_features = X
    elif(presentation == "confidence" or presentation == "model_feature"):
        assert feature_data != None
        X_features = feature_data.cpu()
        
    t2 = time()
    
    logger.info("Extract feature timing: %.4f s. Feature matrix shape: %s" % (t2 - t1, X_features.shape))
    
    if("CIFAR" in dataset_name):
        X_features = X_features.detach().numpy()

    if(ret_dist_matrix):
        t1 = time()
        dist_matrix = cosine_similarity(X_features, X_features)
        t2 = time()
        
        if(presentation == "confidence"):
            dist_matrix = np.abs(dist_matrix)
        
        logger.info("calculate similarity matrix timing %.4f s, size of matrix: %.4f MB" % (t2 - t1, sys.getsizeof(dist_matrix)/8/1024/1024))
        return dist_matrix

    X_result = cal_cos_similarity_by_matrix(X_features=X_features, logger=logger, theta = theta, 
                                            min_similar_samples = min_similar_samples)
    
    if(save_dist_distribution):
        statistic_df = pd.DataFrame()
        #statistic_df["res_1"] = res_1
        statistic_df["X_result"] = X_result
        fig = statistic_df.plot.hist(alpha=0.5, title="cosine similarity distribution")
        fig_path = "../figures/cosine_similarity_result.png"
        fig.figure.savefig(fig_path)
        print("save figure to ", fig_path)
    
    return X_result

def test_model(model, dataloader, device, logger, dataset_len = None, op = "ACC", \
                echo = False, demo = False):
    # Test the model
    # if the op is "conf", then only return conf vector.
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    model = model.to(device)

    ret = []
    if(op == "conf"):
        ret = torch.zeros((dataset_len))
    elif(op == "label"):
        ret = torch.zeros((dataset_len), dtype = torch.long)
        ret_label = torch.zeros((dataset_len), dtype = torch.long)
    elif(op == "conf_data" or op == "model_feature_data"):
        ret = None

    t1 = time()
    with torch.no_grad():
        if(op == "ACC"):
            correct = 0
            total = 0
        
        for images, labels, idx in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if(op != "model_feature_data"):
                max_output, predicted = torch.max(outputs.data, 1)
            
            if(op == "ACC"):
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            elif(op == "conf"):
                min_output, min_predicted = torch.min(outputs.data, 1)
                ret[idx] = (max_output - min_output).detach().clone().cpu()
            elif(op == "label"):
                ret[idx] = predicted.detach().clone().cpu()
                ret_label[idx] = labels.detach().clone().cpu()
            elif(op == "conf_data" or op == "model_feature_data"):
                if(ret == None):
                    ret = torch.zeros((dataset_len, outputs.size(1))).to(device)
                
                ret[idx] = outputs

    t2 = time()

    if(op == "conf"):
        assert ret[ret<0].sum().item() == 0

        ret = ret.numpy().tolist()

        if(logger):
            logger.info("Calculating %s timing: %.4f" % ("confidence", t2-t1))
            logger.info(get_statistics(ret))

        return ret[:500] if demo else ret

    elif(op == "label"):
        return ret, ret_label

    elif(op == "ACC"):
        ret.append(correct / total)
        if(echo):
            logger.info("Model %s on the %d test samples: %.6f" % (op, total, correct / total))

    return ret
