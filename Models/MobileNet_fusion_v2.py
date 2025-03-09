import torch
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

class ModelProcess:
    def __init__(self):
        self._supper_class_activation=True
        self._Mixed_Class_Activation=True
        self._method="DS" #["ByMax","DS",TrainMeta"]  --DS just work with mix data
        self._config=Config(self._Mixed_Class_Activation,self._supper_class_activation,self._method)
        
        # set Directories
        self._data_path = self._config.directories["data_path"] 
        
        self._model1_data_path=self._config.directories["model1_data_path"] 
        self._model2_data_path=self._config.directories["model2_data_path"] 
        self._model3_data_path=self._config.directories["model3_data_path"] 
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
        
        
        # set parameters
        if self._Mixed_Class_Activation:
            self._num_classes=self._config.model_parameters["num_classes"]+1
        else:
            self._num_classes=self._config.model_parameters["num_classes"]
        self._device=self._config.model_parameters["device"]
        self._results = pd.DataFrame(columns=["True label"])

    def save_result(self,img,act_label,pre_label,pic_num):

        # Show the image
        img = img.permute(1, 2, 0)
        plt.imshow(img)
        plt.title(f"orginal:{self._group_labels[act_label]} predicted:{self._group_labels[pre_label]}")
        plt.axis("off")  # Remove axes
        #plt.show()
        
        # Save the figure
        plt.savefig(os.path.join(self._save_graph, f"results{pic_num}.png"), dpi=300, bbox_inches='tight')  # High-quality save
    
    def get_predictions_and_probabilities(self,model, orginallabels,dataloader, device='cpu'):
        model.to(device)
        predictions = []
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

    def data_reader(self):
        
        # model 1
        self._log.log("Loading dataset1...")
        Loader = DataLoad(
            self._model1_data_path, None, None, 
            self._batch_size, self._height_transform, self._width_transform,self._group_labels_path
        )
        train_loader1, _, _,unique_labels1,unique_labels_str1 = Loader.DataLoad()
        
        if self._Mixed_Class_Activation:
            unique_labels1=sorted(unique_labels1)
        self._orginal_labels.append(unique_labels1)
                    
        self._log.log("Dataset1 loaded successfully!")
        
        #model 2
        self._log.log("Loading dataset2...")
        Loader = DataLoad(
            self._model2_data_path, None, None, 
            self._batch_size, self._height_transform, self._width_transform,self._group_labels_path
        )
        train_loader2, _, _ ,unique_labels2,unique_labels_str2= Loader.DataLoad()
        if self._Mixed_Class_Activation:
            unique_labels2=sorted(unique_labels2)
        self._orginal_labels.append(unique_labels2)
        self._log.log("Dataset2 loaded successfully!")
        
        #model 3
        self._log.log("Loading dataset3...")
        Loader = DataLoad(
            self._model3_data_path, None, None, 
            self._batch_size, self._height_transform, self._width_transform,self._group_labels_path
        )
        train_loader3, _, _,unique_labels3,unique_labels_str3 = Loader.DataLoad()
        if self._Mixed_Class_Activation:
            unique_labels3=sorted(unique_labels3)
        self._orginal_labels.append(unique_labels3)
        self._log.log("Dataset3 loaded successfully!")
        
        
        combined_data=[]
        combined_labels=[]
        orginal_label=[]
        for (data,label) in train_loader1:
            combined_data.append(data)
            combined_labels.append(label)
            orginal_label.append(unique_labels1[label])
            

        for (data,label) in train_loader2:
            combined_data.append(data)
            combined_labels.append(label)
            orginal_label.append(unique_labels2[label])

        for (data,label) in train_loader3:
            combined_data.append(data)
            combined_labels.append(label)
            orginal_label.append(unique_labels3[label]) 
            
        dataset = TensorDataset(torch.cat(combined_data, dim=0),torch.tensor(orginal_label))# torch.cat(combined_labels, dim=0))#,torch.tensor(orginal_label))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader,combined_labels
        

    def models_output_colector(self):

        self._log.log("======== Starting Model Training ========")

        # Load dataset
        self._dataloader,_=self.data_reader()
        
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
            accuracy, probabilities, pr_label, tr_label =self.get_predictions_and_probabilities(model, self._orginal_labels[id],self._dataloader, device='cpu')
            if id==0:
                self._results["True label"]=tr_label
                
            self._results[f"model {id+1} label"]=[self._orginal_labels[id][lb] for lb in pr_label]
            
            self._results[f"model {id+1} prp"]=probabilities
                
            self._log.log(f"Model {id+1} Accuracy: {accuracy}")
        
    
    def get_final_estimation(self):
        
        X = []  # Model outputs (probabilities)
        y = []  # True labels

        # Extract features from all models
        for row in range(len(model._results)):
            pr1 = np.array(model._results.loc[row]["model 1 prp"][:-1])
            pr2 = np.array(model._results.loc[row]["model 2 prp"][:-1])
            pr3 = np.array(model._results.loc[row]["model 3 prp"][:-1])

            # Concatenate model probabilities as features
            X.append(np.concatenate([pr1, pr2, pr3]))

            # Store the true label
            y.append(model._results.loc[row]["True label"])

        # Convert to NumPy arrays
        X = np.array(X)
        y = np.array(y)

        # Split into train & test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train a logistic regression model
        meta_model =RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42)
        #meta_model = LogisticRegression(max_iter=1000)
        meta_model.fit(X_train, y_train)

        # Predict on test set
        detected_labels = meta_model.predict(X_test)

        # Compute accuracy
        accuracy = (detected_labels == y_test).mean()

        self._log.log(f"Meta-Model Accuracy:{accuracy}")  
    
    def get_final_estimation_bymax(self):
        
        total_correct=0
        total_row=0
        data_iter = iter(self._dataloader)
        for row in range(len(self._results)):
            if self._Mixed_Class_Activation:
                pr_row=[]
                for i in range(3):
                    p=np.array(self._results.loc[row][f"model {i+1} prp"])
                    if self._orginal_labels[i][np.argmax(p)]==6:
                        pr1=0 
                    else:
                        pr1=max(p)
                    pr_row.append(pr1)  
            else:  
                pr1 = max(np.array(self._results.loc[row]["model 1 prp"]))
                pr2 = max(np.array(self._results.loc[row]["model 2 prp"]))
                pr3 = max(np.array(self._results.loc[row]["model 3 prp"]))
                pr_row=[pr1,pr2,pr3]       
            elected=np.argmax(pr_row)

            true_labels=self._results.loc[row]["True label"]
            predictedlabel=self._results.loc[row][f"model {elected+1} label"]
            
            
            images, labels = next(data_iter)  # Fetch a batch
            image = images[row % len(images)]
            if row % 100 ==0:
                self.save_result(image,labels,predictedlabel,row)
            
            
            total_row+=1
            if (true_labels==predictedlabel):
                total_correct+=1
        accuracy=total_correct/total_row        
        self._log.log(f"Meta-Model Accuracy:{accuracy}" )

    def combine_beliefs(self,belief1, belief2, unknown_label=6, alpha=0.5):
        """
        Implements Dempster's rule of combination for two mass functions.
        belief1 and belief2 are dictionaries where keys are numeric labels.
        """
        combined = {}
        conflict = 0.0  # Conflict factor
        if 6 in belief1.keys():
            uukn_pr=belief1[6]/10
            del belief1[6]
            for j in range(3):
                for i in self._orginal_labels[j]:
                    if ((i not in belief1.keys()) and (i!=6)):
                        belief1[i]=uukn_pr

        if 6 in belief2.keys():
            uukn_pr=belief2[6]/10
            del belief2[6]
            for j in range(3):
                for i in self._orginal_labels[j]:
                    if ((i not in belief2.keys()) and (i!=6)):
                        belief2[i]=uukn_pr        

        for key1, mass1 in belief1.items():
            temp={}
            for key2, mass2 in belief2.items():
                temp[key2]=mass1 * mass2
                if key1 != key2:  # Matching labels → Normal combination
                    conflict += mass1 * mass2
            combined[key1] = temp


        # Normalize the combined mass function to remove conflict
        final = {key: 0 for key in belief2.keys()}
        if conflict < 1.0:
            for key1, mass1 in combined.items():
                for key2,mass2 in mass1.items():
                   final[key1]+= mass2
                   final[key2]+= mass2
            sum=0       
            for key in final:
                final[key] /= (1 - conflict)    
                sum+=final[key]
            for key in final:
                final[key] /=sum
        else:
            raise ValueError("Total conflict: Dempster's rule fails.")

        return final
    
    def get_final_by_DS(self):
        total_correct=0
        total_row=0
        data_iter = iter(self._dataloader)
        for row in range(len(self._results)):
            # Convert model outputs into mass functions
            fusion=self.combine_beliefs(dict(zip(self._orginal_labels[0],self._results.loc[row]["model 1 prp"])), dict(zip(self._orginal_labels[1],self._results.loc[row]["model 2 prp"])))
            fusion=self.combine_beliefs(fusion, dict(zip(self._orginal_labels[2],self._results.loc[row]["model 3 prp"])))
            predictedlabel = max(fusion, key=fusion.get)
            
            true_labels=self._results.loc[row]["True label"]

            images, labels = next(data_iter)  # Fetch a batch
            image = images[row % len(images)]
            if row % 100 ==0:
                self.save_result(image,labels,predictedlabel,row)
            total_row+=1
            if (true_labels==predictedlabel):
                total_correct+=1
        accuracy=total_correct/total_row        
        self._log.log(f"Meta-Model Accuracy:{accuracy}" )
            
            
model=ModelProcess()
model.models_output_colector()
#model.get_final_estimation()
if model._method=="ByMax":
    model.get_final_estimation_bymax()
if model._method=="DS":
    model.get_final_by_DS()