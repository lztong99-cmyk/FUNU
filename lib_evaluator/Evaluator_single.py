
import torch
import os
import logging
import pickle
from utils import model_selection, get_class_num, test_model, ORIGINAL_PROPORTION
from lib_dataset.Dataset import MyDataset
from lib_trainer.Trainer import Trainer
from lib_model.ShadowAttackModel import ShadowAttackModel, DT, LR, RF, MLP_INF
from lib_evaluator.Attack import attack_for_blackbox

class Evaluator():
    def __init__(self, dataset_name, dataset_path,
                 model_name, model_path, pretrained = False, model = None, original = False, unlearning_filter = None):
        
        # Logging
        self.logger = logging.getLogger('evaluator_log')

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path

        self.model_name = model_name
        self.model_load_path = model_path
        self.model_pretrained = pretrained

        if(unlearning_filter):
            self.unlearning_filter = unlearning_filter
        else:
            self.unlearning_filter = "NoFilter"
        
        if(model):
            self.model = model
        else:
            # Prepare model
            self.model = model_selection(self.model_name, get_class_num(self.dataset_name), input_channel = None, shadow_model=False)
            self.model.load_state_dict(torch.load(self.model_load_path))
            self.logger.info("Having loaded models from %s." % (self.model_load_path))
        
        self.original_model = original
    
    def eval(self,config, unlearning_data_selection = -1, unlearning_proportion = -1, device = -1, model_save_path = -1):
        # MIA
        self.attack_one_model(config=config, target_model = self.model, unlearning_data_selection=unlearning_data_selection, unlearning_proportion=unlearning_proportion,
                              device=device, model_save_path = model_save_path, flag = "original" if self.original_model else "retrained", unlearning_filter = self.unlearning_filter )

        # Save dataset D*, Dr, Dt.
        # self.evaluate_on_Du_Dr_Dt(config = config, left = False, op="ACC")

    def attack_one_model(self, config, target_model, unlearning_data_selection = -1, unlearning_proportion = -1, device = -1, 
                         model_save_path = -1, lr = 0.001, epochs = 10, flag = "original", unlearning_filter = None ):
        unlearning_flag_dict = {"original":0, "retrained":1}
        unlearning = unlearning_flag_dict[flag]
        batch_size = 64
        
        self.logger.info("========= begin training %s SHADOW model =========" % flag)

        shadow_trainer = Trainer(dataset_name=self.dataset_name, dataset_path=self.dataset_path, 
                 model_name=self.model_name, model_save_path = model_save_path if model_save_path != -1 else config["model_save_path"],
                 device = device if device != -1 else config["device"])
        # Note that left=True here indicating using the left data to train shadow model.
        shadow_trainer.load_data(unlearning=unlearning, unlearning_data_selection = unlearning_data_selection if unlearning_data_selection != -1 else config["unlearning_data_selection"], 
                                   unlearning_proportion = unlearning_proportion if unlearning_proportion != -1 else config["unlearning_proportion"],
                                   original_proportion = 1 - ORIGINAL_PROPORTION,
                                   left=True,training_batch_size=64)
        shadow_trainer.initialize_model(shadow_model = True, model_load_path = None, 
                                        pretrained=self.model_pretrained)
        # shadow_trainer.prepare_train_setting(learning_rate=config["learning_rate"],
        #                            epochs=config["epochs"])
        # shadow_model = shadow_trainer.train_model()
        shadow_trainer.training_choice(learning_rate= lr, epochs=epochs, save_epoch = 1e7, shadow_or_attack = 1)
        # shadow_trainer.testModel()
        # shadow_trainer.saveModel(shadow_or_attack = 1, shadow_trainer.epochs)
        
        # use the shadow model path to store related data
        self.attack_path = shadow_trainer.model_path
        
        self.logger.info("========= begin preparing attack model =========")

        attack_model = ShadowAttackModel(get_class_num(self.dataset_name))

        # Note that unlearning = unlearning+2
        attack_train = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                unlearning_data_selection = unlearning_data_selection if unlearning_data_selection != -1 else config["unlearning_data_selection"], 
                                unlearning_proportion = unlearning_proportion if unlearning_proportion != -1 else config["unlearning_proportion"],
                                dataset_type ="train", unlearning=unlearning+2, 
                                original_proportion = 1 - ORIGINAL_PROPORTION)
        # Note that dataset_type = test
        attack_test = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                unlearning_data_selection = unlearning_data_selection if unlearning_data_selection != -1 else config["unlearning_data_selection"], 
                                unlearning_proportion = unlearning_proportion if unlearning_proportion != -1 else config["unlearning_proportion"],
                                dataset_type ="test", unlearning=unlearning+2, 
                                original_proportion = 1 - ORIGINAL_PROPORTION)
                
        attack_trainloader = torch.utils.data.DataLoader(attack_train, batch_size=batch_size, 
                                                         shuffle=True, num_workers=2)
        attack_testloader = torch.utils.data.DataLoader(attack_test, batch_size=batch_size, 
                                                        shuffle=True, num_workers=2)
        
        self.logger.info("========= begin MIA attack =========")

        # Device
        self.device = torch.device(device if device != -1 else config["device"])

        self.MIA_attack(ATTACK_PATH = self.attack_path, device = self.device,
                        attack_trainloader = attack_trainloader, 
                        attack_testloader = attack_testloader,
                        target_model = target_model, shadow_model = shadow_trainer.model, 
                        attack_model = attack_model, get_attack_set = 1, attack_epoch = 10,
                        unlearning_filter = unlearning_filter)

    def gen_attack_path(self):
        return self.attack_path
    
    # blackbox shadow
    def MIA_attack(self, ATTACK_PATH, device, 
                   attack_trainloader, attack_testloader, target_model, shadow_model, 
                   attack_model, get_attack_set, attack_epoch = 10, unlearning_filter = None):

        # Note that here the new MIA models would cover the old ones. 
        # We do not really need to store the MIA models.
        MODELS_PATH = os.path.join(ATTACK_PATH, "%s_meminf_attack_model.pth" % unlearning_filter)
        RESULT_PATH = os.path.join(ATTACK_PATH, "%s_meminf_attack_result.p" % unlearning_filter)  # ATTACK_PATH + "_meminf_attack_result.p"
        ATTACK_SETS = os.path.join(ATTACK_PATH, "%s_meminf_attack_dataset_" % unlearning_filter) # ATTACK_PATH + "_meminf_attack_dataset_"

        attack = attack_for_blackbox(ATTACK_SETS, attack_trainloader, attack_testloader, 
                                     target_model, shadow_model, attack_model, device, self.logger)
        res_train, res_test = None, None
        
        if get_attack_set:
            attack.delete_pickle()
            attack.prepare_dataset()

        if(type(attack_model).__name__ in [ "DT", "LR", "RF", "MLP_INF" ]):
            with open(attack.attack_train_path, "rb") as f:
                output, prediction, members = pickle.load(f)
            
            output, members =  output.cpu().detach().numpy() , members.cpu().detach().numpy()
            attack_model.train_model(output, members)
            
            with open(attack.attack_test_path, "rb") as f:
                output, prediction, members = pickle.load(f)

            output, members =  output.cpu().detach().numpy() , members.cpu().detach().numpy()
            train_acc = attack_model.test_model_acc(output, members)
            train_auc = attack_model.test_model_auc(output, members)
            self.logger.info(f"Attack Accuracy: {100 * train_acc:.2f}%, Attack AUC: {100 * train_auc:.2f}%")
        else:
            attack_epoch = attack_epoch
            
            for i in range(attack_epoch):
                flag = 1 if i == attack_epoch -1 else 0
                res_train = attack.train(flag, RESULT_PATH, i)
                res_test = attack.test(flag=flag, result_path=RESULT_PATH, epoch = i, 
                                    testset_path=attack.attack_test_path)

            attack.saveModel(MODELS_PATH)
            self.logger.info("Saved Attack Model to %s" % MODELS_PATH)

        return res_train, res_test
    
    def evaluate_on_Du_Dr_Dt(self, config, left, op="ACC" ):

        # generate dataset
        
        self.unlearned_dataset = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                unlearning_data_selection=config["unlearning_data_selection"], 
                                unlearning_proportion=config["unlearning_proportion"],
                                dataset_type ="unlearned", unlearning=1, 
                                original_proportion = ORIGINAL_PROPORTION, left=left)
        self.remained_dataset = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                unlearning_data_selection=config["unlearning_data_selection"], 
                                unlearning_proportion=config["unlearning_proportion"],
                                dataset_type ="remained", unlearning=1,
                                original_proportion=ORIGINAL_PROPORTION, left=left)
        self.test_dataset = MyDataset(dataset_name=self.dataset_name, dataset_path=self.dataset_path,
                                unlearning_data_selection=config["unlearning_data_selection"], 
                                unlearning_proportion=config["unlearning_proportion"],
                                dataset_type ="test", unlearning=1,
                                original_proportion=ORIGINAL_PROPORTION, left=left)
        
        test_batch_size = 128
        unlearned_loader = torch.utils.data.DataLoader(dataset=self.unlearned_dataset, batch_size=test_batch_size, shuffle=False)
        remained_loader = torch.utils.data.DataLoader(dataset=self.remained_dataset, batch_size=test_batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=test_batch_size, shuffle=False)

        # begin evaluate
        M_Du = test_model(self.model, unlearned_loader, self.device, self.logger, op = "ACC", echo = True).pop()
        M_Dr = test_model(self.model, remained_loader, self.device, self.logger, op = "ACC", echo = True).pop()
        M_Dt = test_model(self.model, test_loader, self.device, self.logger, op = "ACC", echo = True).pop()

        self.logger.info("%s model %s on Du: %.6f, on Dr: %.6f, on Dt: %.6f" % ("original" if self.original_model else "retrained", op, M_Du, M_Dr, M_Dt))