import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.fus_baseMosel import FusBaseModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from DataLoad import DataLoad
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.mobilenet import MobileNetV2ForCIFAR8M
from Config.config_model_fusion_v2 import Config
from Log import Logger
from train import Train
from test import Test
import random

class ByMetta(FusBaseModel):
    def __init__(self,Mixed,Supper,method,act_train):
        self._mixed=Mixed
        self._supper=Supper
        self._method=method
        super().__init__(self._mixed,self._supper,method)
        self._dataloader,self._dataloader_val,self._dataloader_tst,self._dataloader_base_tst=self.data_reader()
        self._results={}
        
        if act_train:
            self.train_meta_model()

            
    def train_meta_model(self):
        
        self._log.log("======== Starting Model Training ========")
        # Load dataset
        # Initialize model
        self._log.log("Initializing the model...")
        model=MobileNetV2ForCIFAR8M(3,self._height_transform,self._width_transform,self._drop_out)  
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=self._label_smoothing)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate, weight_decay=self._weight_decay)

        # Train model
        self._log.log(f"Starting training for {self._epoch} epochs...")
        train = Train(model, self._epoch, self._dataloader, self._dataloader_val, criterion, optimizer, self._device,self._log,self._save_log,self._save_graph)
        train.train_model()
                # Save results
        self._log.log("Saving trained model and training results...")

        # Test model
        self._log.log("Starting model evaluation...")
        test = Test(model,[1,2,3],["model 1","model 2","model 3"], self._dataloader_tst, criterion, self._device,self._log,self._save_graph)
        test.test_model()
        
    def meta_model_output(self):

        meta_model=MobileNetV2ForCIFAR8M(3,self._height_transform,self._width_transform,self._drop_out)
        meta_model.load_state_dict(torch.load(os.path.join(self._save_log,"model_weights.pth")), strict=False)
        meta_model.eval()
        meta_model = meta_model.to(self._device)
        probabilities = []
        pr_lable=[]
        Tr_lable=[]
        tot=0
        correct=0
        with torch.no_grad():  # No need to calculate gradients during inference
            for batch_inputs,batch_labels in tqdm(self._dataloader_tst, desc="Making predictions"):
                for i in range(batch_inputs.shape[0]):  # Assuming batch_inputs is a tensor
                    inputs = batch_inputs[i].unsqueeze(0)  # Keep the tensor structure
                    orglabel = batch_labels[i].unsqueeze(0)
                    inputs = inputs.to(self._device)
                    # Forward pass
                    outputs = meta_model(inputs)
                    # Get probabilities using softmax
                    probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                    # Get predicted class (index of the maximum probability)
                    _, predicted_classes = torch.max(probs, 1)
                    probabilities.extend(probs.tolist())
                    pr_lable.extend(predicted_classes.tolist()) 
                    Tr_lable.extend(orglabel.tolist()) 
                    tot+=1
                    pr=predicted_classes.tolist()[0]
                    b=orglabel.tolist()
                    if (pr==b[0]):
                        correct+=1
        accuracy=correct/tot
        self._results[f"Meta_pr_label"]=pr_lable
        self._results[f"Meta_tr_label"]=Tr_lable
        self._results[f"Meta_prp"]=probabilities
        self._results[f"Accuracy"]=accuracy
        self.sub_models()
        self.final_desision()
        
    def final_desision(self):
        correct=0
        tot=0
        for i,item in enumerate(self._results[f"Meta_pr_label"]):
            pr_label=self._results[f"model {item+1} label"][i]
            tr_label=self._results["True label"][i]
            if pr_label==tr_label:
                correct+=1
            tot+=1
        self._log.log(f"final accuracy {correct/tot}")
    def get_predictions_and_probabilities(self,model, orginallabels):
        model.to(self._device)
        probabilities = []
        pr_lable=[]
        Tr_lable=[]
        tot=0
        correct=0
        with torch.no_grad():  # No need to calculate gradients during inference
            for batch_inputs,batch_labels in tqdm(self._dataloader_base_tst, desc="Making predictions"):
                for i in range(batch_inputs.shape[0]):  # Assuming batch_inputs is a tensor
                    inputs = batch_inputs[i].unsqueeze(0)  # Keep the tensor structure
                    orglabel = batch_labels[i].unsqueeze(0)
                    inputs = inputs.to(self._device)
                    # Forward pass
                    outputs = model(inputs)
                    # Get probabilities using softmax
                    probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                    # Get predicted class (index of the maximum probability)
                    _, predicted_classes = torch.max(probs, 1)
                    pr=predicted_classes.tolist()[0]
                    probabilities.extend(probs.tolist())
                    pr_lable.extend([orginallabels[pr]]) 
                    Tr_lable.extend(orglabel.tolist()) 
                    tot+=1
                    a=6 if orglabel.tolist()[0] not in orginallabels else orglabel.tolist()[0]
                    b=orginallabels[pr]
                    if (a==b):
                        correct+=1
        accuracy=correct/tot
        return accuracy, probabilities,pr_lable,Tr_lable            
    
    def sub_models(self):
         # Initialize model
        self._log.log("Initializing the sub model...")
        model=MobileNetV2ForCIFAR8M(self._num_classes,self._height_transform,self._width_transform,self._drop_out)  
        for id,path in enumerate(self._models_weights_path):
            model.load_state_dict(torch.load(path), strict=False)
            model.eval()
            if self._Mixed_Class_Activation:
                lbl=self._orginal_labels[id]
                lbl.extend([6])
                self._orginal_labels[id]=sorted(lbl)
            accuracy, probabilities, pr_label, tr_label =self.get_predictions_and_probabilities(model, self._orginal_labels[id])
            if id==0:
                self._results["True label"]=tr_label
            self._results[f"model {id+1} label"]=pr_label
            self._results[f"model {id+1} prp"]=probabilities
            self._log.log(f"Model {id+1} Accuracy: {accuracy}")
            
            
            
model=ByMetta(Mixed=False,Supper=True,method="TrainMeta",act_train=False)# --DS just work with mix data
model.meta_model_output()
