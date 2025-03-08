import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR , ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Train:
    def __init__(self,model,epoch,train_loader,val_loader, criterion, optimizer, device,log,save_path,save_graph):
        self._model=model
        self._train_loader=train_loader
        self._val_loader=val_loader
        self._criterion=criterion
        self._optimizer=optimizer
        self._device=device
        self._epoch=epoch
        self._log=log
        self._save_path=os.path.join(save_path,"model_weights.pth")
        self._save_graph= os.path.join(save_graph,"Train_graph.png")
        
    def save_result(self,tr_ac,val_ac,tr_los,val_los):
        #save trained model weight
        save_dir = os.path.dirname(self._save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), self._save_path)
        #save validation and train graph
        fig,ax=plt.subplots(ncols=2)
        ax[0].plot(tr_ac, label="Train_ac")
        ax[0].plot(val_ac, label="val_ac")
        ax[0].legend()
        ax[1].plot(tr_los, label="Train_los")
        ax[1].plot(val_los, label="val_los")
        ax[1].legend()
        
        # Save the figure
        save_dir = os.path.dirname(self._save_graph)  # Extract directory part

        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        plt.savefig(self._save_graph, dpi=300, bbox_inches='tight')  # High-quality save
                
    def train_model(self):
        self._model.to(self._device)
        tr_los=[]
        tr_ac=[]
        val_los=[]
        val_ac=[]
        for epoch in range(self._epoch):
            self._model.train()
            correct = 0
            total = 0
            running_loss = 0.0
            for inputs, labels in tqdm(self._train_loader, desc=f"Epoch {epoch + 1}/{self._epoch}"):
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                # Zero the parameter gradients
                self._optimizer.zero_grad()
                # Forward pass
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                # Backward pass and optimization
                loss.backward()
                self._optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                scheduler = ReduceLROnPlateau(self._optimizer, mode='min', factor=0.2, patience=2, verbose=True)

            accuracy = 100 * correct / total
            tr_los.append(running_loss/len(self._train_loader))
            tr_ac.append(accuracy)
            self._model.eval()  # Set model to evaluation mode
            val_correct = 0
            val_total = 0
            val_running_loss = 0.0
            with torch.no_grad():  # No gradients needed for validation
                for inputs, labels in self._val_loader:
                    inputs, labels = inputs.to(self._device), labels.to(self._device)
                    outputs = self._model(inputs)
                    loss = self._criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_accuracy = 100 * val_correct / val_total
            val_los.append(val_running_loss/len(self._val_loader))
            val_ac.append(val_accuracy)
            self._log.log(f"Tr_Loss: {running_loss/len(self._train_loader):.4f}, val_loss: {val_running_loss/len(self._val_loader):.4f}, Tr_acc: {accuracy}, val_ac: {val_accuracy}")
        
        self.save_result(tr_ac,val_ac,tr_los,val_los)