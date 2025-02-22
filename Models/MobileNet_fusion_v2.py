import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import to_pil_image
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
        self._config1=Config()
        
        # set Directories
        self._data_path = self._config1.directories["data_path"] 
        
        self._model1_data_path=self._config1.directories["model1_data_path"] 
        self._model2_data_path=self._config1.directories["model2_data_path"] 
        self._model3_data_path=self._config1.directories["model3_data_path"] 
        self._group_labels_path=self._config1.directories["group_labels"]
        with open(self._group_labels_path, "r") as file:
            labels = [line.strip() for line in file]  # Convert each line to a float
        self._group_labels=labels
        self._model1_weights_path = self._config1.directories["model1_weights"]  
        self._model2_weights_path = self._config1.directories["model2_weights"]  
        self._model3_weights_path = self._config1.directories["model3_weights"] 
        self._models_weights_path=[self._model1_weights_path,self._model2_weights_path,self._model3_weights_path] 
         
        self._save_log=self._config1.directories["save_log"]
        self._save_graph=self._config1.directories["save_log"]
        
        self._log=Logger(self._save_log,"fusion")
        
        # set hyperparameters
        self._batch_size = self._config1.hyperparameters["batch_size"]
        self._learning_rate = self._config1.hyperparameters["learning_rate"]
        self._epoch = self._config1.hyperparameters["epochs"]
        self._valdata_ratio=self._config1.hyperparameters["valdata_ratio"]
        self._width_transform=self._config1.hyperparameters["width_transform"]
        self._height_transform=self._config1.hyperparameters["height_transform"]
        self._drop_out=self._config1.hyperparameters["drop_out"]
        
        
        # set parameters
        self._num_classes=self._config1.model_parameters["num_classes"]
        self._device=self._config1.model_parameters["device"]
        self._results = pd.DataFrame(columns=["True label"])
        self._orginal_labels=[[0,10,20,30,40],[1,11,21,31,41],[2,12,22,32,42]]

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
                if (orginallabels[pr]==orglabel.tolist()[0]):
                    correct+=1
        accuracy=correct/tot
        return accuracy, probabilities,pr_lable,Tr_lable    

    def data_reader(self):
        
        # model 1
        self._log.log("Loading dataset1...")
        Loader = DataLoad(
            self._model1_data_path, None, None, 
            self._batch_size, self._height_transform, self._width_transform
        )
        train_loader1, _, _ = Loader.DataLoad()
        self._log.log("Dataset1 loaded successfully!")
        
        #model 2
        self._log.log("Loading dataset2...")
        Loader = DataLoad(
            self._model2_data_path, None, None, 
            self._batch_size, self._height_transform, self._width_transform
        )
        train_loader2, _, _ = Loader.DataLoad()
        self._log.log("Dataset2 loaded successfully!")
        
        #model 3
        self._log.log("Loading dataset3...")
        Loader = DataLoad(
            self._model3_data_path, None, None, 
            self._batch_size, self._height_transform, self._width_transform
        )
        train_loader3, _, _ = Loader.DataLoad()
        self._log.log("Dataset3 loaded successfully!")
        
        
        combined_data=[]
        combined_labels=[]
        orginal_label=[]
        for (data,label) in train_loader1:
            combined_data.append(data)
            combined_labels.append(label)
            orginal_label.append(self._orginal_labels[0][label])

        for (data,label) in train_loader2:
            combined_data.append(data)
            combined_labels.append(label)
            orginal_label.append(self._orginal_labels[1][label])

        for (data,label) in train_loader3:
            combined_data.append(data)
            combined_labels.append(label)
            orginal_label.append(self._orginal_labels[2][label]) 
            
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
            pr1 = np.array(model._results.loc[row]["model 1 prp"])
            pr2 = np.array(model._results.loc[row]["model 2 prp"])
            pr3 = np.array(model._results.loc[row]["model 3 prp"])

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
            
model=ModelProcess()
model.models_output_colector()
model.get_final_estimation()
#model.get_final_estimation_bymax()