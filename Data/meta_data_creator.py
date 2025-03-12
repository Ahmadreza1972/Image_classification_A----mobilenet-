import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Config.config_model1 import Config as Config1
from Config.config_model2 import Config as Config2
from Config.config_model3 import Config as Config3
from Models.DataLoad import DataLoad
import numpy as np
import torch
from collections import Counter

class mixedDataCreator():
    def __init__(self):
        # just change the second item to make MIX for normal group and super class group
        self.con1=Config1(False,False)
        self.con2=Config2(False,False)
        self.con3=Config3(False,False)
        self.models=["model1","model2","model3"]

        self.train_pathes={"model1":self.con1.directories["train_path"],
                      "model2":self.con2.directories["train_path"],
                      "model3":self.con3.directories["train_path"]}
        self.test_paths={"model1":self.con1.directories["test_path"],
                      "model2":self.con2.directories["test_path"],
                      "model3":self.con3.directories["test_path"]}
        path=self.con1.directories["data_dir"]
    
    def data_appender(self,id,item_excep,Mixed_data,Mixed_labels,Mixed_indices,path):
        
        Loader = DataLoad(None, None,None, None, None, None,path)
        images, labels,indices, =Loader.load_data(path[item_excep])
        class_num=list(Counter(labels).values())
        # Choose 250 random indices without replacement
        selected_indices = np.random.choice(len(indices), int(class_num[1]/3), replace=False)
        # Select corresponding data
        Mixed_data.extend([images[item] for i,item in enumerate(selected_indices)])
        Mixed_labels.extend([id  for i,item in enumerate(selected_indices)])
        Mixed_indices.extend(selected_indices)
        
    def main(self):
        # Train
        Mixed_train_data={}
        for id,item_excep in enumerate(self.models):
            Mixed_data=[]
            Mixed_labels=[]
            Mixed_indices=[]
            self.data_appender(id,item_excep,Mixed_data,Mixed_labels,Mixed_indices,self.train_pathes)
            Mixed_train_data["indices"]=Mixed_indices
            Mixed_train_data["labels"]=Mixed_labels
            Mixed_train_data["data"]=Mixed_data
            save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), item_excep))
            save_path = os.path.join(save_dir, "Metta"+item_excep+"_train.pth")
            torch.save(Mixed_train_data,save_path) 
        # Test
        Mixed_train_data={}
        for id,item_excep in enumerate(self.models):
            Mixed_data=[]
            Mixed_labels=[]
            Mixed_indices=[]
            self.data_appender(id,item_excep,Mixed_data,Mixed_labels,Mixed_indices,self.test_paths)
            Mixed_train_data["indices"]=Mixed_indices
            Mixed_train_data["labels"]=Mixed_labels
            Mixed_train_data["data"]=Mixed_data
            save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), item_excep))
            save_path = os.path.join(save_dir, "Metta"+item_excep+"_test.pth")
            torch.save(Mixed_train_data,save_path)     

mix=mixedDataCreator()
mix.main()