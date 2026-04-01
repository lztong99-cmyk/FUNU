
import torch
import logging
from utils import model_selection, get_class_num, test_model, ORIGINAL_PROPORTION
from lib_dataset.Dataset import MyDataset
from lib_trainer.Trainer import Trainer
from lib_model.ShadowAttackModel import ShadowAttackModel
from lib_evaluator.Attack import attack_for_blackbox

class Evaluator():
    def __init__(self, config, original_proportion, left):

        self.config = config
        self.original_proportion=original_proportion
        self.left=left

        # Device
        self.device = torch.device(config["device"])
        
        # Logging
        self.logger = logging.getLogger('evaluator_log')

        self.dataset_name = config["dataset"]
        self.model_name = config["model"]

        self.model_retrain_path = config["model_retrain_path"]
        self.model_original_path = config["model_original_path"]

        # Prepare models to be compared
        self.model_original = model_selection(self.model_name, get_class_num(self.dataset_name), input_channel = None, shadow_model=False)
        self.model_retrain = model_selection(self.model_name, get_class_num(self.dataset_name), input_channel = None, shadow_model=False)
        
        self.model_original.load_state_dict(torch.load(self.model_original_path))
        self.model_retrain.load_state_dict(torch.load(self.model_retrain_path))

        self.logger.info("Having loaded models from %s, %s." % (self.model_original_path, 
                                                                self.model_retrain_path))
    def compare(self):
        # MIA
        self.get_MIA_result()
        
        # Parameter distance
        self.get_param_distance(model1 = self.model_original, model2 = self.model_retrain, p=2)

        # Save dataset D*, Dr, Dt.
        self.evaluate_on_Du_Dr_Dt(original_proportion = self.original_proportion, left = self.left, op="ACC")
            
    def get_param_distance(self, model1, model2, p=2):
        """
        To calculate average distance over all parameters.
        p=1: L1 norm
        P=2: L2 norm
        """
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
        
        self.logger.info("Parameter distance (%d norm): %.5f" % (p, avg_dist))
        
        return avg_dist

    def attack_one_model(self, target_model, flag = "original", original_proportion = 0.6):
        unlearning_flag_dict = {"original":0, "retrained":1}
        unlearning = unlearning_flag_dict[flag]
        batch_size = 64
        
        self.logger.info("========= begin training %s SHADOW model =========" % flag)

        print("========= begin RETRAINING/UNLEARNING experiment =========")

        trainer = Trainer(self.config, unlearning = unlearning, original_proportion = original_proportion, 
                          left = True, shadow_model=True)
        shadow_model = trainer.trainModel()
        trainer.testModel()
        trainer.saveModel(self.config, unlearning = unlearning, shadow_or_attack = 1)
        
        self.logger.info("========= begin preparing attack model =========")

        attack_model = ShadowAttackModel(get_class_num(self.dataset_name))
        attack_path = self.config["attack_path"]

        attack_train = MyDataset(self.config, dataset_type ="train", unlearning=unlearning + 2, 
                                               original_proportion = original_proportion)
        attack_test = MyDataset(self.config, dataset_type ="test", unlearning = unlearning + 2,
                                               original_proportion = original_proportion)
                
        attack_trainloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, 
                                                         shuffle=True, num_workers=2)
        attack_testloader = torch.utils.data.DataLoader(attack_test, batch_size=batch_size, 
                                                        shuffle=True, num_workers=2)
        
        self.logger.info("========= begin MIA attack =========")

        self.MIA_attack(ATTACK_PATH = attack_path, device = self.device, 
                        attack_trainloader = attack_trainloader, 
                        attack_testloader = attack_testloader,
                        target_model = target_model, shadow_model = shadow_model, 
                        attack_model = attack_model, get_attack_set = 1, attack_epoch = 10)

    # blackbox shadow
    def MIA_attack(self, ATTACK_PATH, device, 
                   attack_trainloader, attack_testloader, target_model, shadow_model, 
                   attack_model, get_attack_set, attack_epoch = 10):

        # Note that here the new MIA models would cover the old ones. 
        # We do not really need to store the MIA models.
        MODELS_PATH = ATTACK_PATH + "_meminf_attack_model.pth"
        RESULT_PATH = ATTACK_PATH + "_meminf_attack_result.p"
        ATTACK_SETS = ATTACK_PATH + "_meminf_attack_dataset_"

        attack = attack_for_blackbox(ATTACK_SETS, attack_trainloader, attack_testloader, 
                                     target_model, shadow_model, attack_model, device, self.logger)
        
        if get_attack_set:
            attack.delete_pickle()
            attack.prepare_dataset()

        attack_epoch = attack_epoch
        for i in range(attack_epoch):
            flag = 1 if i == attack_epoch -1 else 0
            res_train = attack.train(flag, RESULT_PATH, i)
            res_test = attack.test(flag=flag, result_path=RESULT_PATH, epoch = i, 
                                   testset_path=attack.attack_test_path)

        attack.saveModel(MODELS_PATH)
        self.logger.info("Saved Attack Model to %s" % MODELS_PATH)

        return res_train, res_test

    def get_MIA_result(self):
        self.attack_one_model(target_model = self.model_original, flag = "original", original_proportion = 0.6)
        self.attack_one_model(target_model = self.model_retrain, flag = "retrained", original_proportion = 0.6)
    
    def evaluate_on_Du_Dr_Dt(self, original_proportion, left, op="ACC" ):
        # generate dataset
        self.unlearned_dataset = MyDataset(self.config, dataset_type ="unlearned", unlearning=True,
                                           original_proportion=original_proportion, left=left)
        self.remained_dataset = MyDataset(self.config, dataset_type ="remained", unlearning=True,
                                          original_proportion=original_proportion, left=left)
        self.test_dataset = MyDataset(self.config, dataset_type ="test", unlearning=True,
                                      original_proportion=original_proportion, left=left)

        test_batch_size = 128
        unlearned_loader = torch.utils.data.DataLoader(dataset=self.unlearned_dataset, batch_size=test_batch_size, shuffle=False)
        remained_loader = torch.utils.data.DataLoader(dataset=self.remained_dataset, batch_size=test_batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=test_batch_size, shuffle=False)

        # begin evaluate
        Mo_Du = test_model(self.model_original, unlearned_loader, self.device, self.logger, op = "ACC").pop()
        Mo_Dr = test_model(self.model_original, remained_loader, self.device, self.logger, op = "ACC").pop()
        Mo_Dt = test_model(self.model_original, test_loader, self.device, self.logger, op = "ACC").pop()

        Mr_Du = test_model(self.model_retrain, unlearned_loader, self.device, self.logger, op = "ACC").pop()
        Mr_Dr = test_model(self.model_retrain, remained_loader, self.device, self.logger, op = "ACC").pop()
        Mr_Dt = test_model(self.model_retrain, test_loader, self.device, self.logger, op = "ACC").pop()

        self.logger.info("Original model %s on Du: %.6f, on Dr: %.6f, on Dt: %.6f" % (op, Mo_Du, Mo_Dr, Mo_Dt))
        self.logger.info("Retrained model %s on Du: %.6f, on Dr: %.6f, on Dt: %.6f" % (op, Mr_Du, Mr_Dr, Mr_Dt))

    def train_shadow_model(self):
        self.logger.info("========= begin training ORIGINAL SHADOW model =========")
        trainer = Trainer(self.config, unlearning = 0, original_proportion = 0.6, 
                          left = True, shadow_model=True)
        trainer.trainModel()
        trainer.testModel()
        trainer.saveModel(self.config, unlearning = 0, shadow_or_attack = 1)

        self.logger.info("========= begin training UNLEARNED/RETRAINED SHADOW model =========")
        trainer = Trainer(self.config, unlearning = 1, original_proportion = 0.6, 
                          left = True, shadow_model=True)
        trainer.trainModel()
        trainer.testModel()
        trainer.saveModel(self.config, unlearning = 1, shadow_or_attack = 1)