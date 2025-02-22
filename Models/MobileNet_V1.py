import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from DataLoad import DataLoad
from train import Train
from test import Test
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.mobilenet import MobileNetV2ForCIFAR8M
from Config.config_model1 import Config
from Log import Logger

class ModelProcess:
    def __init__(self):
        self._config1=Config()
        

        # set Directories
        self._train_path = self._config1.directories["train_path"]  
        self._test_path = self._config1.directories["test_path"]
        self._save_path= self._config1.directories["save_path"]
        self._save_graph=self._config1.directories["output_graph"]
        self._save_log=self._config1.directories["save_log"]
        
        self._log=Logger(self._save_log,"model1")
        
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

    def save_result(self,model,tr_ac,val_ac,tr_los,val_los):
        
        #save trained model weight
        save_dir = os.path.dirname(self._save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), self._save_path)
        
        #save validation and train graph
        fig,ax=plt.subplots(ncols=2)
        ax[0].plot(tr_ac, label="Train_ac")
        ax[0].plot(val_ac, label="val_ac")
        ax[0].legend()
        ax[1].plot(tr_los, label="Train_los")
        ax[1].plot(val_los, label="val_los")
        ax[1].legend()
        
        # Save the figure
        plt.savefig(self._save_graph, dpi=300, bbox_inches='tight')  # High-quality save
        
    def main(self):

        self._log.log("======== Starting Model Training ========")

        # Load dataset
        self._log.log("Loading dataset...")
        Loader = DataLoad(
            self._train_path, self._test_path, self._valdata_ratio, 
            self._batch_size, self._height_transform, self._width_transform
        )
        train_loader, val_loader, test_loader = Loader.DataLoad()
        self._log.log("Dataset loaded successfully!")

        # Initialize model
        self._log.log("Initializing the model...")
        model=MobileNetV2ForCIFAR8M(self._num_classes,self._height_transform,self._width_transform,self._drop_out)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_layers = sum(1 for _ in model.modules())
        self._log.log(f"Model initialized with {trainable_params:,} trainable parameters")

        # Log model architecture
        self._log.log(f"Model Architecture: \n{model}")

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self._learning_rate)

        # Train model
        self._log.log(f"Starting training for {self._epoch} epochs...")
        train = Train(model, self._epoch, train_loader, val_loader, criterion, optimizer, self._device,self._log)
        tr_ac, val_ac, tr_los, val_los = train.train_model()

        # Save results
        self._log.log("Saving trained model and training results...")
        self.save_result(model, tr_ac, val_ac, tr_los, val_los)

        # Test model
        self._log.log("Starting model evaluation...")
        test = Test(model, test_loader, criterion, self._device,self._log)
        test.test_model()

        self._log.log("======== Model Training Completed! ========")

model=ModelProcess()
model.main()
