import torch
import logging
from lib_dataset.Dataset import MyDataset
from lib_trainer.Trainer import Trainer
from utils import ORIGINAL_PROPORTION, SHARD_NUM, SLICE_NUM, get_shard_slice_num, \
    gen_unlearned_index, find_min_influenced_slices_for_shard, map_unlearned_index2slice, \
    gen_predefined_unlearned_index, clear_folder
from time import time

class SISAExecutor():
    def __init__(self, dataset_name, dataset_path,
                 model_name, model_path, device):
        
        # Logging
        self.logger = logging.getLogger('unlearner_log')

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        self.model_name = model_name
        self.model_path = model_path

        self.device = device

    def train_contituent_model(self, learning_rate = 0.001, epochs_per_submodel = 10, 
                               just_to_read_full_dataset_length = False):
        
        self.learning_rate = learning_rate
        self.epochs_per_submodel = epochs_per_submodel
        self.full_training_dataset_len = 0
        
        total_slice_num = SHARD_NUM * SLICE_NUM
        last_saved_model_path = ""
        
        t1 = time()
        
        for i in range(total_slice_num):
            
            # shard_num range from [0,4], slice_num range from [0,4]
            shard_num, slice_num = get_shard_slice_num(i)
            self.logger.info("Training model at shard %d, slice %d" % (shard_num, slice_num))

            contituent_trainer = Trainer(dataset_name = self.dataset_name, dataset_path = self.dataset_path, 
                 model_name = self.model_name, model_save_path = self.model_path,
                 device = self.device)
            
            # i indicates the number of slices
            contituent_trainer.load_data(unlearning = 0, unlearning_data_selection = None,
                                   unlearning_proportion = None, 
                                   original_proportion = ORIGINAL_PROPORTION,
                                   left = False, training_batch_size = 128, 
                                   sisa_selection_op = i) 
            
            if(self.full_training_dataset_len == 0):
                self.full_training_dataset_len = contituent_trainer.train_dataset.full_training_dataset_len_sisa

                if(just_to_read_full_dataset_length):
                    break

            if(slice_num == 0):
                pretrained = True if self.model_name == "ResNet-18" else False
                contituent_trainer.initialize_model(shadow_model = False, model_load_path = None,
                                            pretrained = pretrained, freeze_conv_layer=False)
            else:
                # load model from last epoch if this is not the first model within one shard
                contituent_trainer.initialize_model(shadow_model = False, model_load_path = last_saved_model_path,
                                            pretrained = False, freeze_conv_layer=False)

            contituent_trainer.prepare_train_setting(learning_rate = self.learning_rate, epochs = self.epochs_per_submodel)
            
            # do not save model during training using default save method
            contituent_trainer.train_model(save_epoch = 0, do_test = False)

            # save sisa model manually
            last_saved_model_path = contituent_trainer.save_sisa_model(i)
        
        t2 = time()

        self.logger.info("SISA training timing: %.4f" % (t2 -t1))

        contituent_trainer.test_sisa_model(test_unlearn=False, dataset = "test", unlearned_data_index = None)
        # contituent_trainer.test_sisa_model(test_unlearn=False, dataset = "unlearned")
        contituent_trainer.test_sisa_model(test_unlearn=False, dataset = "remained", unlearned_data_index = None)

    def unlearn_contituent_model(self, unlearning_data_selection = "Random", unlearn_prop = 0.1, predefined = False, filtered = False):

        self.logger.info("======= begin SISA unlearning. dataset: %s, unlearn_prop: %.4f, unlearning_data_selection: %s, filtered: %s =======" % (self.dataset_name, unlearn_prop, unlearning_data_selection, filtered))

        # general index for all unlearned samples
        original_sample_indices = list(range(self.full_training_dataset_len))
        if(predefined == True):
            unlearned_data_index = gen_predefined_unlearned_index(dataset_name = self.dataset_name, delete_num = unlearn_prop, filtered = filtered)
        else:
            unlearned_data_index = gen_unlearned_index(original_sample_indices = original_sample_indices, 
                                                    unlearning_data_selection = unlearning_data_selection, 
                                                    unlearning_proportion = unlearn_prop)
        self.logger.info("length of unlearned_data_index: %d" % len(unlearned_data_index))
        
        # locate influenced shard and the slice that is influenced in the earliest round
        influenced_shard_slice = find_min_influenced_slices_for_shard(unlearned_sample_indices = unlearned_data_index, 
                                                                      original_sample_indices = original_sample_indices)
        self.logger.info("influenced_shard_slice: %s" % influenced_shard_slice)

        total_slice_num = SHARD_NUM * SLICE_NUM
        unlearned_model_count = 0
        empty_dataset_flag = 0

        t1 = time()

        for i in range(total_slice_num):
            
            # shard_num range from [0,4], slice_num range from [0,4]
            shard_num, slice_num = get_shard_slice_num(i)

            if(shard_num not in influenced_shard_slice or slice_num < influenced_shard_slice[shard_num]):
                continue
            
            self.logger.info("Retraining model at shard %d, slice %d" % (shard_num, slice_num))

            contituent_trainer = Trainer(dataset_name = self.dataset_name, dataset_path = self.dataset_path, 
                 model_name = self.model_name, model_save_path = self.model_path,
                 device = self.device)
            
            # i indicates the number of slices
            # the unlearned_index falls in the range of the slice size 
            unlearned_index_per_slice = map_unlearned_index2slice(i, unlearned_data_index, self.full_training_dataset_len)
            contituent_trainer.load_data(unlearning = 1, unlearning_data_selection = None,
                                   unlearning_proportion = None, 
                                   original_proportion = ORIGINAL_PROPORTION,
                                   left = False, training_batch_size = 128, 
                                   sisa_selection_op = i, unlearned_index = unlearned_index_per_slice) 
            
            if(len(contituent_trainer.train_dataset) == 0):
                empty_dataset_flag = 1
                continue

            if(slice_num == 0 or empty_dataset_flag == 1):
                # training from the beginning
                pretrained = True if self.model_name == "ResNet-18" else False
                contituent_trainer.initialize_model(shadow_model = False, model_load_path = None,
                                            pretrained = pretrained, freeze_conv_layer=False)
                empty_dataset_flag = 0
            else:
                # load the last model
                model_load_path = contituent_trainer.gen_sisa_model_path(i-1, original=True)
                contituent_trainer.initialize_model(shadow_model = False, model_load_path = model_load_path,
                                            pretrained = False, freeze_conv_layer=False)

            contituent_trainer.prepare_train_setting(learning_rate = self.learning_rate, epochs = self.epochs_per_submodel)
            
            # do not save model during training using default save method
            contituent_trainer.train_model(save_epoch = 0, do_test = False)

            unlearned_model_count += 1
            # do not save sisa model, only record the unlearning time.
            contituent_trainer.save_sisa_model(i, original = False)
        
        original_unlearned_data_index = gen_predefined_unlearned_index(dataset_name = self.dataset_name, delete_num = unlearn_prop, filtered = False)
        contituent_trainer.test_sisa_model(test_unlearn=True, dataset = "test", unlearned_data_index = original_unlearned_data_index)
        contituent_trainer.test_sisa_model(test_unlearn=True, dataset = "unlearned", unlearned_data_index = original_unlearned_data_index)
        contituent_trainer.test_sisa_model(test_unlearn=True, dataset = "remained", unlearned_data_index = original_unlearned_data_index)

        t2 = time()

        # clear models saved at gen_sisa_model_path(i-1, original=False)
        clear_folder(contituent_trainer.gen_sisa_model_folder(original = False))
        
        self.logger.info("SISA unlearning proportion: %.4f, %d models retrained, unlearning timing: %.4f" % (unlearn_prop, unlearned_model_count, t2 -t1))