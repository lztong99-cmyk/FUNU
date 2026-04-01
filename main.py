import os
import logging
import utils
from lib_trainer.Trainer import Trainer
from lib_evaluator.Evaluator_single import Evaluator
from lib_dataset.Distribution import DistributionMiner
from lib_unlearner.SISA import SISAExecutor
from lib_unlearner.HessianUnlearner import HessianUnlearner

def config_logger(config, unlearning, distribution_mining_exp, SISA_exp):
    
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')

    # create console handler
    logfolder = config["log_path"]
    log_name = utils.gen_save_name(dataset_name=config["dataset"], 
                                     unlearning_data_selection=config["unlearning_data_selection"],
                                     model_name=config["model"],unlearning_proportion=config["unlearning_proportion"],
                                     unlearning=unlearning, shadow_or_attack=False, 
                                     distribution_mining_exp = distribution_mining_exp,
                                     sisa_exp = SISA_exp)
    log_name = log_name + (config["log_appendix"] if "log_appendix" in config else "")
    save_path = os.path.join(logfolder, log_name + ".log")
    logging.basicConfig(filename = save_path, filemode="w")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("log file name: %s \n config: %s" % (save_path, config))
    
    return logger

def main(config):

    logger = config_logger(config, unlearning = 1 if(config["retraining_exp"]) else 0, distribution_mining_exp = config["distribution_mining_exp"],
                  SISA_exp = config["SISA_exp"])

    if(config["original_training_exp"]):
        logger.info("========= begin ORIGINAL TRAINING experiment =========")
        # "original_proportion = 0.6, left = False" indicates use the 60% of orginal dataset for training, 
        # while the left dataset is for shadow model training
        original_trainer = Trainer(dataset_name=config["dataset"], dataset_path=config["dataset_path"], 
                 model_name=config["model"], model_save_path=config["model_save_path"],
                 device = config["device"])
        original_trainer.load_data(unlearning=0, unlearning_data_selection=None,
                                   unlearning_proportion=None, original_proportion=utils.ORIGINAL_PROPORTION,
                                   left=False,training_batch_size=config["batch_size"], sisa_selection_op = -1,
                                   remove_class = config["remove_class"] if "remove_class" in config else -1)
        original_trainer.initialize_model(shadow_model = False, model_load_path = config["model_original_path"], 
                                          pretrained = config["model_pretrained"], freeze_conv_layer=False)  
        original_trainer.training_choice(learning_rate= config["learning_rate"], epochs=config["epochs"], 
                                         save_epoch = 3, 
                                         iter_data = config["iter_data"] if "iter_data" in config else 1)
    
    if(config["retraining_exp"]):
        logger.info("========= begin RETRAINING/UNLEARNING experiment =========")
        retrain_trainer = Trainer(dataset_name=config["dataset"], dataset_path=config["dataset_path"], 
                 model_name=config["model"], model_save_path=config["model_save_path"],
                 device = config["device"])
        retrain_trainer.load_data(unlearning=1, unlearning_data_selection=config["unlearning_data_selection"],
                                    unlearning_proportion=config["unlearning_proportion"], 
                                    original_proportion=utils.ORIGINAL_PROPORTION,
                                    left=False,training_batch_size=config["batch_size"],sisa_selection_op = -1,
                                    remove_class = config["remove_class"] if "remove_class" in config else -1)
        
        if("model_retrain_path" in config and config["model_retrain_path"]):
            retrain_trainer.initialize_model(shadow_model = False, model_load_path = config["model_retrain_path"],
                                         pretrained = config["model_pretrained"], freeze_conv_layer=False)
        else:
            retrain_trainer.initialize_model(shadow_model = False, model_load_path = None,
                                            pretrained = config["model_pretrained"], freeze_conv_layer=False)
            retrain_trainer.training_choice(learning_rate= config["learning_rate"], epochs=config["epochs"], save_epoch = 1e7) # set save_epoch=1e7 indicates only save the model at the last epoch
            
        if("eval_retrained_model" in config and config["eval_retrained_model"]):
            retrain_trainer.test_on_Du_Dr_Dt()
            evaluator = Evaluator(dataset_name=config["dataset"], dataset_path=config["dataset_path"],
                    model_name=config["model"], model_path = None, # filter_trainer.final_model_path(load the model from the last epoch)
                    model = retrain_trainer.model, pretrained=config["shadow_pretrained_verification"], original = False)
            evaluator.eval(config)
    
    if("unlearning_request_filter_exp" in config and config["unlearning_request_filter_exp"]):
        if("filter_method" in config):
            for filter_method in config["filter_method"]:
                logger.info("========= begin UNLEARNING REQUEST FILTER experiment: %s =========" % filter_method)
                filter_trainer = Trainer(dataset_name=config["dataset"], dataset_path=config["dataset_path"], 
                        model_name=config["model"], model_save_path=config["model_save_path"],
                        device = config["device"])
                
                if(filter_method == "rfmodel"):
                    rf_model_theta, rf_model_alpha = filter_trainer.cal_similarity_matrix_typical_theta(conf_epoch = config["conf_epoch"] if "conf_epoch" in config else utils.CONF_MODEL_EPOCH,
                                                                                                        theta_by_label=True, presentation = "model_feature")
                    filter_trainer.load_data(unlearning=1, unlearning_data_selection=config["unlearning_data_selection"],
                                        unlearning_proportion=config["unlearning_proportion"], 
                                        original_proportion=utils.ORIGINAL_PROPORTION,
                                        left=False,training_batch_size=config["batch_size"],
                                        sisa_selection_op = -1, unlearning_filter = filter_method, sim_theta = rf_model_theta,
                                        sim_alpha = rf_model_alpha, score_thres_dict = config["score_thres_dict"] if "score_thres_dict" in config else {},
                                        remove_class = config["remove_class"] if "remove_class" in config else 2)
                else:
                    filter_trainer.load_data(unlearning=1, unlearning_data_selection=config["unlearning_data_selection"],
                                        unlearning_proportion=config["unlearning_proportion"], 
                                        original_proportion=utils.ORIGINAL_PROPORTION,
                                        left=False,training_batch_size=config["batch_size"],
                                        sisa_selection_op = -1, unlearning_filter = filter_method, sim_theta = -1,
                                        sim_alpha = -1, score_thres_dict = config["score_thres_dict"] if "score_thres_dict" in config else {},
                                        remove_class = config["remove_class"] if "remove_class" in config else 2)
                
                if("model_%s_path" % filter_method in config and config["model_%s_path" % filter_method]):
                    filter_trainer.initialize_model(shadow_model = False, model_load_path = config["model_%s_path" % filter_method],
                                                pretrained = config["model_pretrained"], freeze_conv_layer=False)
                else:
                    filter_trainer.initialize_model(shadow_model = False, model_load_path = None,
                                                    pretrained = config["model_pretrained"], freeze_conv_layer=False)
                    filter_trainer.training_choice(learning_rate= config["learning_rate"], epochs=config["epochs"], save_epoch = 1e7) # set save_epoch=1e7 indicates only save the model at the last epoch
                
                if("eval_filtered_model" in config and config["eval_filtered_model"]):
                    filter_trainer.test_on_Du_Dr_Dt()
                    evaluator = Evaluator(dataset_name=config["dataset"], dataset_path=config["dataset_path"],
                            model_name=config["model"], model_path = None, # filter_trainer.final_model_path(load the model from the last epoch)
                            model = filter_trainer.model, pretrained=config["shadow_pretrained_verification"], original = False,
                            unlearning_filter = filter_method)
                    evaluator.eval(config)
                    
                    if(config["retraining_exp"]):
                        utils.get_param_distance(model1 = retrain_trainer.model, model2 = filter_trainer.model, logger = evaluator.logger, device = config["device"], p=2)

    if(config["distribution_mining_exp"]):
        print("========= begin DISTRIBUTION MINING experiment =========")
        distribution_miner = DistributionMiner(dataset_path=config["dataset_path"], 
                                               statistic_path = config["statistic_path"],
                                               device = config["device"], demo = config["demo"])
        distribution_miner.distribution_mining(distribution_choice=config["distribution_choice"], 
                                               statistic_path = config["statistic_path"],
                                               model_save_path = config["model_save_path"], 
                                               normalization=True, n_components=3)
        # distribution_miner.plot_distribution(result_df=None, result_df_path = "/home/wamdm/lizitong/UnnecessaryUnlearning/distribution_statistics/MNIST.csv", 
        #                                     statistic_path = config["statistic_path"], filename="MNIST", figure_type="hist")
    if(config["verification_exp"]):
        print("========= begin EVALUATION experiment =========")
        evaluator_original = Evaluator(dataset_name=config["dataset"], dataset_path=config["dataset_path"],
                 model_name=config["model"], model_path = "../models/MNIST-2-layer-CNN-original-.pt", 
                 model = None, pretrained=config["shadow_pretrained_verification"], original = False)
        evaluator_original.eval(config)

    if(config["SISA_exp"]):

        print("========= begin SISA experiment =========")
        """
        
        mnist_sisa_executor = SISAExecutor(dataset_name = "MNIST", dataset_path=config["dataset_path"], 
                                            model_name = "2-layer-CNN", model_path = config["model_save_path"],  device = config["device"])
        mnist_sisa_executor.train_contituent_model(learning_rate = 0.001, epochs_per_submodel = 10, just_to_read_full_dataset_length=False)
        
        mnist_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 10, predefined = True, filtered = False)
        mnist_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 10, predefined = True, filtered = True)
        mnist_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 30, predefined = True, filtered = False)
        mnist_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 30, predefined = True, filtered = True)
        mnist_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 50, predefined = True, filtered = False)
        mnist_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 50, predefined = True, filtered = True)
        
        cifar10_sisa_executor = SISAExecutor(dataset_name = "CIFAR10", dataset_path=config["dataset_path"], 
                                            model_name = "ResNet-18", model_path = config["model_save_path"],  device = config["device"])
        cifar10_sisa_executor.train_contituent_model(learning_rate = 0.001, epochs_per_submodel = 10, just_to_read_full_dataset_length=False)
        
        cifar10_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 10, predefined = True, filtered = False)
        cifar10_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 10, predefined = True, filtered = True)
        cifar10_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 30, predefined = True, filtered = False)
        cifar10_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 30, predefined = True, filtered = True)
        cifar10_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 50, predefined = True, filtered = False)
        cifar10_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 50, predefined = True, filtered = True)

        """
        
        cifar100_sisa_executor = SISAExecutor(dataset_name = "CIFAR100", dataset_path=config["dataset_path"], 
                                            model_name = "ResNet-18", model_path = config["model_save_path"],  device = config["device"])
        cifar100_sisa_executor.train_contituent_model(learning_rate = 2e-4, epochs_per_submodel = 10, just_to_read_full_dataset_length=False)
        
        cifar100_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 10, predefined = True, filtered = False)
        cifar100_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 10, predefined = True, filtered = True)
        cifar100_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 30, predefined = True, filtered = False)
        cifar100_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 30, predefined = True, filtered = True)
        cifar100_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 50, predefined = True, filtered = False)
        cifar100_sisa_executor.unlearn_contituent_model(unlearning_data_selection = "Random", unlearn_prop = 50, predefined = True, filtered = True)
        
        
    if("Hessian_unlearning_exp" in config and config["Hessian_unlearning_exp"]):
        
        print("========= begin Hessian Unlearning experiment =========")
        
        if("do_mnist_hessian" in config and config["do_mnist_hessian"]):
            mnist_hessian_unlearner = HessianUnlearner(dataset_name = "MNIST", dataset_path=config["dataset_path"], 
                                                model_name = "2-layer-CNN", model_save_path = config["model_save_path"],  
                                                device = config["device"])
            
            for unlearn_prop in [30, 50, 100]:
            
                mnist_hessian_unlearner.unlearn_exp(origin_model_path = config["mnist_model_path"], unlearn_prop = unlearn_prop, save_unlearned_model = True, 
                                                    remove_class = config["remove_class"] if "remove_class" in config else -1, l2lambda = 30)
            
                if("eval_mnist_hessian" in config and config["eval_mnist_hessian"]):
                    mnist_hessian_unlearner.test_on_Du_Dr_Dt()
                    evaluator = Evaluator(dataset_name="MNIST", dataset_path=config["dataset_path"],
                            model_name="2-layer-CNN", model_path = None, # filter_trainer.final_model_path(load the model from the last epoch)
                            model = mnist_hessian_unlearner.model, pretrained=config["shadow_pretrained_verification"], original = False)
                    evaluator.eval(config=None, unlearning_data_selection = "Random", unlearning_proportion = unlearn_prop , device = config["device"],
                                   model_save_path = config["model_save_path"])

            class_unlearn_prop = 0.5
            mnist_hessian_unlearner.unlearn_exp(origin_model_path = config["mnist_model_path"], unlearn_prop = class_unlearn_prop, save_unlearned_model = True,
                                                unlearning_data_selection = "Byclass", remove_class = config["remove_class"] if "remove_class" in config else -1,
                                                l2lambda = 30)
            
            if("eval_mnist_hessian" in config and config["eval_mnist_hessian"]):
                mnist_hessian_unlearner.test_on_Du_Dr_Dt()
                evaluator = Evaluator(dataset_name="MNIST", dataset_path=config["dataset_path"],
                        model_name="2-layer-CNN", model_path = None, # filter_trainer.final_model_path(load the model from the last epoch)
                        model = mnist_hessian_unlearner.model, pretrained=config["shadow_pretrained_verification"], original = False)
                evaluator.eval(config=None, unlearning_data_selection = "Byclass", unlearning_proportion = class_unlearn_prop , device = config["device"],
                               model_save_path = config["model_save_path"])
                    
        if("do_cifar10_hessian" in config and config["do_cifar10_hessian"]):
            cifar10_hessian_unlearner = HessianUnlearner(dataset_name = "CIFAR10", dataset_path=config["dataset_path"], 
                                                model_name = "ResNet-18", model_save_path = config["model_save_path"],  
                                                device = config["device"])
            
            """
            for unlearn_prop in [30, 50, 100]:
                cifar10_hessian_unlearner.unlearn_exp(origin_model_path = config["cifar10_model_path"], unlearn_prop = unlearn_prop, save_unlearned_model = True,
                                                      selectionType="FOCI",
                                                      remove_class = config["remove_class"] if "remove_class" in config else -1, l2lambda = 0.001)
                
                if("eval_cifar10_hessian" in config and config["eval_cifar10_hessian"]):
                    cifar10_hessian_unlearner.test_on_Du_Dr_Dt()
                    evaluator = Evaluator(dataset_name="CIFAR10", dataset_path=config["dataset_path"],
                            model_name="ResNet-18", model_path = None, # filter_trainer.final_model_path(load the model from the last epoch)
                            model = cifar10_hessian_unlearner.model, pretrained=config["shadow_pretrained_verification"], original = False)
                    evaluator.eval(config=None, unlearning_data_selection = "Random", unlearning_proportion = unlearn_prop , device = config["device"],
                                   model_save_path = config["model_save_path"])
            """
            class_unlearn_prop = 0.5
            cifar10_hessian_unlearner.unlearn_exp(origin_model_path = config["cifar10_model_path"], unlearn_prop = class_unlearn_prop, save_unlearned_model = True, selectionType="FOCI",
                                                unlearning_data_selection = "Byclass", remove_class = config["remove_class"] if "remove_class" in config else -1,
                                                l2lambda = 0.001)
            
            if("eval_cifar10_hessian" in config and config["eval_cifar10_hessian"]):
                cifar10_hessian_unlearner.test_on_Du_Dr_Dt()
                evaluator = Evaluator(dataset_name="CIFAR10", dataset_path=config["dataset_path"],
                            model_name="ResNet-18", model_path = None, # filter_trainer.final_model_path(load the model from the last epoch)
                            model = cifar10_hessian_unlearner.model, pretrained=config["shadow_pretrained_verification"], original = False)
                evaluator.eval(config=None, unlearning_data_selection = "Byclass", unlearning_proportion = class_unlearn_prop , device = config["device"],
                               model_save_path = config["model_save_path"])
                
        if("do_cifar100_hessian" in config and config["do_cifar100_hessian"]):
            cifar100_hessian_unlearner = HessianUnlearner(dataset_name = "CIFAR100", dataset_path=config["dataset_path"], 
                                                model_name = "ResNet-18", model_save_path = config["model_save_path"],  
                                                device = config["device"])
            """
            for unlearn_prop in [30, 50, 100]:
                cifar100_hessian_unlearner.unlearn_exp(origin_model_path = config["cifar100_model_path"], unlearn_prop = unlearn_prop, save_unlearned_model = True,
                                                      selectionType="FOCI",
                                                      remove_class = config["remove_class"] if "remove_class" in config else -1, l2lambda = 0.001)
                
                if("eval_cifar100_hessian" in config and config["eval_cifar100_hessian"]):
                    cifar100_hessian_unlearner.test_on_Du_Dr_Dt()
                    evaluator = Evaluator(dataset_name="CIFAR100", dataset_path=config["dataset_path"],
                            model_name="ResNet-18", model_path = None, # filter_trainer.final_model_path(load the model from the last epoch)
                            model = cifar100_hessian_unlearner.model, pretrained=config["shadow_pretrained_verification"], original = False)
                    evaluator.eval(config=None, unlearning_data_selection = "Random", unlearning_proportion = unlearn_prop , device = config["device"],
                                   model_save_path = config["model_save_path"])
            """
            class_unlearn_prop = 0.5
            cifar100_hessian_unlearner.unlearn_exp(origin_model_path = config["cifar100_model_path"], unlearn_prop = class_unlearn_prop, save_unlearned_model = True, selectionType="FOCI",
                                                unlearning_data_selection = "Byclass", remove_class = config["remove_class"] if "remove_class" in config else -1,
                                                l2lambda = 0.001)
            
            if("eval_cifar100_hessian" in config and config["eval_cifar100_hessian"]):
                cifar100_hessian_unlearner.test_on_Du_Dr_Dt()
                evaluator = Evaluator(dataset_name="CIFAR100", dataset_path=config["dataset_path"],
                            model_name="ResNet-18", model_path = None, # filter_trainer.final_model_path(load the model from the last epoch)
                            model = cifar100_hessian_unlearner.model, pretrained=config["shadow_pretrained_verification"], original = False)
                evaluator.eval(config=None, unlearning_data_selection = "Byclass", unlearning_proportion = class_unlearn_prop , device = config["device"],
                               model_save_path = config["model_save_path"])
                
    return

if __name__ == "__main__":
    config = utils.parameter_parser()
    
    main(config)
