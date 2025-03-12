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

class FusBaseModel:
    def __init__(self,Mixed,Supper,method):
        self._supper_class_activation=Supper
        self._Mixed_Class_Activation=Mixed
        self._method=method
        self._config=Config(self._Mixed_Class_Activation,self._supper_class_activation,self._method)
        
        # set Directories
        self._data_path = self._config.directories["data_path"] 
        self._model1_data_path=self._config.directories["model1_data_path"] 
        self._model2_data_path=self._config.directories["model2_data_path"] 
        self._model3_data_path=self._config.directories["model3_data_path"] 
        self._model1_tst_data_path=self._config.directories["model1_tst_data_path"]
        self._model2_tst_data_path=self._config.directories["model2_tst_data_path"]
        self._model3_tst_data_path=self._config.directories["model3_tst_data_path"]

        self._group_labels_path=self._config.directories["group_labels"]
        with open(self._group_labels_path, "r") as file:
            labels = [line.strip() for line in file]  # Convert each line to a float
        self._group_labels=labels
        self._model1_weights_path = self._config.directories["model1_weights"]  
        self._model2_weights_path = self._config.directories["model2_weights"]  
        self._model3_weights_path = self._config.directories["model3_weights"] 
        self._models_weights_path=[self._model1_weights_path,self._model2_weights_path,self._model3_weights_path] 
        self._orginal_labels=[]
         
        self._save_log=self._config.directories["save_log"]
        self._save_graph=self._config.directories["save_log"]
        
        self._log=Logger(self._save_log,"fusion")
        
        # set hyperparameters
        self._batch_size = self._config.hyperparameters["batch_size"]
        self._learning_rate = self._config.hyperparameters["learning_rate"]
        self._epoch = self._config.hyperparameters["epochs"]
        self._valdata_ratio=self._config.hyperparameters["valdata_ratio"]
        self._width_transform=self._config.hyperparameters["width_transform"]
        self._height_transform=self._config.hyperparameters["height_transform"]
        self._drop_out=self._config.hyperparameters["drop_out"]
        self._weight_decay=self._config.hyperparameters["weight_decay"]
        self._label_smoothing=self._config.hyperparameters["label_smoothing"]

        
        # set parameters
        if self._Mixed_Class_Activation:
            self._num_classes=self._config.model_parameters["num_classes"]+1
        else:
            self._num_classes=self._config.model_parameters["num_classes"]
        self._device=self._config.model_parameters["device"]
        self._results = pd.DataFrame(columns=["True label"])

    def shuffle_data(self,combined_data,orginal_label,baswe_label=None):
        # Convert lists to NumPy arrays if they are not already
        combined_data = np.asarray(combined_data)
        orginal_label = np.asarray(orginal_label)

        if baswe_label is not None:
            baswe_label = np.asarray(baswe_label)

        # Create shuffled indices
        indices = np.arange(len(combined_data))
        np.random.shuffle(indices)
         # Ensure combined_data is a NumPy array
        combined_data=combined_data[indices]    
        combined_data = np.asarray(combined_data)

        # Convert NumPy arrays to a list of PyTorch tensors
        combined_data = [torch.tensor(arr, dtype=torch.float32) for arr in combined_data]
        # Index arrays using shuffled indices
        if baswe_label is not None:
            return combined_data, orginal_label[indices], baswe_label[indices]
        else:
            return combined_data, orginal_label[indices]
           

    def data_reader(self):
        combined_data,combined_labels,orginal_label=[],[],[]
        combined_data_val,combined_labels_val,orginal_label_val=[],[],[]
        combined_data_tst,combined_labels_tst,orginal_label_tst,base_label_tst=[],[],[],[]
        
        models_path=[self._model1_data_path,self._model2_data_path,self._model3_data_path]
        models_tst_path=[self._model1_tst_data_path,self._model2_tst_data_path,self._model3_tst_data_path]
        for k,path in enumerate(models_path):
            self._log.log(f"Loading dataset {k}...")
            Loader = DataLoad(path, models_tst_path[k], 0.2, 
            1, self._height_transform, self._width_transform,self._group_labels_path
            )
            train_loader,val_loader, test_loade,unique_labels,unique_labels_str = Loader.DataLoad()
            if self._Mixed_Class_Activation:
                unique_labels=sorted(unique_labels)
            self._orginal_labels.append(unique_labels)
                    
            self._log.log(f"Dataset{k} loaded successfully!")

            for (data,label) in train_loader:
                batch_size = data.shape[0]
                combined_data.append(data)
                combined_labels.append(label)
                if self._method=="TrainMeta":
                    orginal_label.extend([k] * batch_size)
                else:
                    orginal_label.extend([unique_labels[label.item()]] * batch_size)
            
            if self._method=="TrainMeta":
                for (data,label) in val_loader:
                    batch_size = data.shape[0]
                    combined_data_val.append(data)
                    combined_labels_val.append(label)
                    orginal_label_val.extend([k] * batch_size)
   

                for (data,label) in test_loade:
                    batch_size = data.shape[0]
                    combined_data_tst.append(data)
                    combined_labels_tst.append(label)
                    orginal_label_tst.extend([k] * batch_size)
                    base_label_tst.extend([unique_labels[label.item()]] * batch_size)

        print(f"Original labels shape: {torch.tensor(orginal_label).shape}")
        combined_data,orginal_label=self.shuffle_data(combined_data,orginal_label)
        
        dataset = TensorDataset(torch.cat(combined_data, dim=0),torch.tensor(orginal_label))            
        if self._method=="TrainMeta": 
            combined_data_val,orginal_label_val=self.shuffle_data(combined_data_val,orginal_label_val) 
            dataset_val = TensorDataset(torch.cat(combined_data_val, dim=0),torch.tensor(orginal_label_val))
            
            combined_data_tst,orginal_label_tst,base_label_tst=self.shuffle_data(combined_data_tst,orginal_label_tst,base_label_tst)
            dataset_tst = TensorDataset(torch.cat(combined_data_tst, dim=0),torch.tensor(orginal_label_tst))
            dataset_base_tst = TensorDataset(torch.cat(combined_data_tst, dim=0),torch.tensor(base_label_tst))
              
            dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
            dataloader_val = DataLoader(dataset_val, batch_size=self._batch_size, shuffle=False)
            dataloader_tst = DataLoader(dataset_tst, batch_size=self._batch_size, shuffle=False)
            dataloader_base_tst = DataLoader(dataset_base_tst, batch_size=self._batch_size, shuffle=False)
            return dataloader,dataloader_val,dataloader_tst,dataloader_base_tst
        else:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)        
        return dataloader
        
    def get_predictions_and_probabilities(self,model, orginallabels,dataloader, device='cpu'):
        model.to(device)
        probabilities = []
        pr_lable=[]
        Tr_lable=[]
        tot=0
        correct=0
        with torch.no_grad():  # No need to calculate gradients during inference
            for inputs,orglabel in tqdm(dataloader, desc="Making predictions"):
                inputs = inputs.to(device)
                # Forward pass
                outputs = model(inputs)
                # Get probabilities using softmax
                probs = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                # Get predicted class (index of the maximum probability)
                _, predicted_classes = torch.max(probs, 1)
                probabilities.extend(probs.tolist())
                pr_lable.extend(predicted_classes.tolist()) 
                Tr_lable.extend(orglabel.tolist()) 
                tot+=1
                pr=predicted_classes.tolist()[0]
                a=6 if orglabel.tolist()[0] not in orginallabels else orglabel.tolist()[0]
                b=orginallabels[pr]
                if (a==b):
                    correct+=1
        accuracy=correct/tot
        return accuracy, probabilities,pr_lable,Tr_lable        

    def models_output_colector(self):
        self._log.log("======== Starting Model Training ========")
        # Load dataset
        self._dataloader=self.data_reader()
         # Initialize model
        self._log.log("Initializing the model...")
        
        model=MobileNetV2ForCIFAR8M(self._num_classes,self._height_transform,self._width_transform,self._drop_out)  
        for id,path in enumerate(self._models_weights_path):
            model.load_state_dict(torch.load(path), strict=False)
            model.eval()
            if self._Mixed_Class_Activation:
                lbl=self._orginal_labels[id]
                lbl.extend([6])
                self._orginal_labels[id]=sorted(lbl)
            accuracy, probabilities, pr_label, tr_label =self.get_predictions_and_probabilities(model, self._orginal_labels[id],self._dataloader, device=self._device)
            if id==0:
                self._results["True label"]=tr_label
            self._results[f"model {id+1} label"]=[self._orginal_labels[id][lb] for lb in pr_label]
            self._results[f"model {id+1} prp"]=probabilities
            self._log.log(f"Model {id+1} Accuracy: {accuracy}")

    def save_result(self,img,act_label,pre_label,pic_num):
        # Show the image
        img = img.permute(1, 2, 0)
        plt.imshow(img)
        plt.title(f"orginal:{self._group_labels[act_label]} predicted:{self._group_labels[pre_label]}")
        plt.axis("off")  # Remove axes
        # Save the figure
        plt.savefig(os.path.join(self._save_graph, f"results{pic_num}.png"), dpi=300, bbox_inches='tight')  # High-quality save
    


    