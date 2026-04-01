import os
import logging
import torch
import matplotlib.pyplot as plt
from time import time
from lib_dataset.Dataset import MyDataset
import pandas as pd
import numpy as np
from utils import get_dataset_clustering_score, ORIGINAL_PROPORTION, get_presentation_similarity, save_score_to_pt
from lib_trainer.Trainer import Trainer

class DistributionMiner():
    def __init__(self, dataset_path, statistic_path, device, demo = False):

        # Data setting
        self.demo = demo

        self.datasets = ["MNIST"] if self.demo else ["MNIST", "CIFAR10", "CIFAR100"]
        self.dataset_path = dataset_path
        self.statistic_path = statistic_path

        # Logging
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        self.logger = logging.getLogger('DistributionMinier_log')
        self.device = device

        self.model_load_paths = {"MNIST-2-layer-CNN": "../models/MNIST-2-layer-CNN-original-model/epoch10.pt", #*
                                 "MNIST-3-layer-CNN": "../models/MNIST-3-layer-CNN-original-model/epoch50.pt",
                                 "MNIST-4-layer-CNN": "../models/MNIST-4-layer-CNN-original-model/epoch50.pt",
                                 "CIFAR10-ResNet-18": "../models/CIFAR10-ResNet-18-original-model/epoch20-NoFilter.pt", #*
                                 "CIFAR10-ResNet-34": "../models/CIFAR10-ResNet-34-original-epoch60-model.pt",
                                 "CIFAR10-ResNet-50": "../models/CIFAR10-ResNet-50-original-epoch60-model.pt",
                                 "CIFAR100-ResNet-18": "../models/CIFAR100-ResNet-18-original-model/epoch100-NoFilter.pt", #*
                                 "CIFAR100-ResNet-34": "../models/CIFAR100-ResNet-34-original-model-epoch100.pt",
                                 "CIFAR100-ResNet-50": "../models/CIFAR100-ResNet-50-original-model-epoch100.pt",
                                 "Tiny-ImageNet-ResNet-18": "../models/Tiny-ImageNet-ResNet-18-original-epoch100-model.pt",
                                 "Tiny-ImageNet-ResNet-34": "../models/Tiny-ImageNet-ResNet-34-original-epoch100-model.pt",
                                 "Tiny-ImageNet-ResNet-50": "../models/Tiny-ImageNet-ResNet-50-original-epoch100-model.pt"}
        
        #self.distribution_mining(distribution_choice=self.distribution_choice, statistic_path = self.statistic_path)

    def distribution_mining(self, model_save_path, distribution_choice = ["clustering"], n_components=2,
                            statistic_path = None, normalization = True, plot = False):
        
        model_dict = {"MNIST":["2-layer-CNN"], #["2-layer-CNN","3-layer-CNN","4-layer-CNN"],
                      "CIFAR10":["ResNet-18"],
                      "CIFAR100":["ResNet-18"],}
                    #  "Tiny-ImageNet":["ResNet-18","ResNet-34","ResNet-50"],}

        for dataset in self.datasets:
            
            distributionminer_trainer = Trainer(dataset_name=dataset, dataset_path=self.dataset_path, 
                 model_name=None, model_save_path=model_save_path, device = self.device)
            distributionminer_trainer.load_data(unlearning = 0, unlearning_data_selection=None, 
                                                unlearning_proportion=None, 
                                                original_proportion=ORIGINAL_PROPORTION, left = False)
            cur_dataset_train = distributionminer_trainer.train_dataset
            cur_dataset_indices = list(range(len(cur_dataset_train)))
            
            # make result dataframe
            result_df = pd.DataFrame()
            result_df["index"] = cur_dataset_indices[:500] if self.demo else cur_dataset_indices

            if("clustering" in distribution_choice):
                t1 = time()
                cur_dataset_clustering = get_dataset_clustering_score(dataset = cur_dataset_train, 
                                                                          n_components = n_components,
                                                                          logger=self.logger, demo = self.demo)
                t2 = time()

                #result_df["clustering"] = cur_dataset_clustering
                file_path = save_score_to_pt(score = cur_dataset_clustering, dataset_name = dataset, model_name = model_dict[dataset][0], score_type = "clustering")
                self.logger.info("Calculate clustering score timing: %.4f s, path: %s" % (t2 - t1, file_path))

            if("similarity" in distribution_choice):
                t1 = time()
                cur_dataset_similarity = get_presentation_similarity(dataset = cur_dataset_train, dataset_name=dataset, logger=self.logger,
                                                                theta = 0.1, min_similar_samples = 10, feature_extractor = "ResNet-18", 
                                                                model_pretrained = True, save_dist_distribution = False, demo = self.demo)
                t2 = time()
                self.logger.info("Calculate similarity score timing: %.4f s" % (t2 - t1))
                result_df["cos_similarity"] = cur_dataset_similarity

            for cur_model_name in model_dict[dataset]:
                distributionminer_trainer.update_model_name(cur_model_name)
                distributionminer_trainer.initialize_model(shadow_model=False, model_load_path=self.model_load_paths['-'.join([dataset, cur_model_name])])
                
                cur_dataset_curvature, cur_dataset_confidence = [],[]
                
                if("curvature" in distribution_choice):
                    t1 = time()
                    cur_dataset_curvature,_ = distributionminer_trainer.get_dataset_curvature_score(logger=self.logger, train_loader = True, demo = self.demo)
                    t2 = time()
                    
                    # result_df[cur_model_name + "#curvature"] = cur_dataset_curvature
                    file_path = save_score_to_pt(score = cur_dataset_curvature, dataset_name = dataset, model_name = cur_model_name, score_type = "curvature")
                    self.logger.info("Calculate %s curvature score timing: %.4f s, path: %s" % (cur_model_name, t2 - t1, file_path))

                if("confidence" in distribution_choice):
                    t1 = time()
                    cur_dataset_confidence = distributionminer_trainer.get_dataset_confidence_score(logger=self.logger, demo = self.demo)
                    t2 = time()

                    file_path = save_score_to_pt(score = cur_dataset_confidence, dataset_name = dataset, model_name = cur_model_name, score_type = "confidence")

                    self.logger.info("Calculate %s confidence score timing: %.4f s, path: %s" % (cur_model_name, t2 - t1, file_path))

                    #result_df[cur_model_name + "#confidence"] = cur_dataset_confidence

            if(statistic_path and plot):
                result_df = result_df.set_index('index')
                if(normalization):
                    for col in list(result_df.columns):
                        if("curevature" in col or "clustering" in col):
                            result_df[col] = result_df[[col]].apply(lambda x: (x - np.min(x))/(np.max(x) - np.min(x)))
                        elif("confidence" in col):
                            # result_df[col] = result_df[[col]].apply(lambda x: x - (np.min(x)*2))
                            # reverse the tendency
                            result_df[col] = 1.0/result_df[col]
                            result_df[col] = result_df[[col]].apply(lambda x: (x - np.min(x))/(np.max(x) - np.min(x)))
                    
                    # result_df = result_df.apply(lambda x: (x - np.min(x))/(np.max(x) - np.min(x)))

                # Save the normalized result.
                self.save_statistic_result(result_df, statistic_path, distribution_choice, dataset)
                self.plot_distribution(result_df, None, statistic_path, distribution_choice, dataset, "hist")

    def save_statistic_result(self, result_df, statistic_path, distribution_choice, filename):
        
        statistics_filename = os.path.join(statistic_path, filename + '-'.join(distribution_choice) + ".csv" if "csv" not in filename else filename)
        result_df.to_csv(statistics_filename)

        self.logger.info("Having result distribution to %s " % statistics_filename)

    def plot_distribution(self, result_df, result_df_path, statistic_path, distribution_choice, filename, figure_type):
        
        # If result_df_path is not None, then read from the file (result_df_path).
        if(result_df_path):
            result_df = pd.read_csv(result_df_path)
            result_df = result_df.apply(lambda x: (x - np.min(x))/(np.max(x) - np.min(x)))
        
        # Plot figure in all
        in_all_columns = list(result_df.columns)
        if('index' in in_all_columns):
            in_all_columns.remove('index')
        fig = result_df[in_all_columns].plot.hist(alpha = 0.5)

        figure_filename = os.path.join(statistic_path, "figures", 
                                       filename +'-' + '-'.join(distribution_choice) + figure_type + "-in-all.png" if "png" not in filename else filename)
        
        fig.figure.savefig(figure_filename)
        self.logger.info("Having result plot to %s " % figure_filename)

        figure_filename = os.path.join(statistic_path, "figures", 
                                       filename +'-' + figure_type + ".png" if "png" not in filename else filename)
        
        columns = list(result_df.columns)

        def get_models_from_resultdf(columns):
            models = set()
            curvature_columns = []
            confidence_columns = []

            for col in columns:
                if('#' in col):
                    models.add(col.split('#')[0])
                if("curvature" in col):
                    curvature_columns.append(col)
                if("confidence" in col):
                    confidence_columns.append(col)
            return models, curvature_columns, confidence_columns
        
        models, curvature_columns, confidence_columns = get_models_from_resultdf(columns)
        n_model = len(models)
        models = list(models)

        fig, ((ax1, ax2, ax3)) = plt.subplots(nrows=1, ncols=3, sharey = True)

        fig.suptitle("%s Data typicality distribution" % filename)

        ax1.hist(result_df[["clustering","cos_similarity"]], label = ["clustering","cos_similarity"])
        ax1.legend(prop={'size': 6})

        
        ax2.hist(result_df[curvature_columns], label = curvature_columns)
        ax2.legend(prop={'size': 6})

        ax3.hist(result_df[confidence_columns], label = confidence_columns)
        ax3.legend(prop={'size': 6})

        fig.figure.savefig(figure_filename)
        self.logger.info("Having result plot to %s " % figure_filename)