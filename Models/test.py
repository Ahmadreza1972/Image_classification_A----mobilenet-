import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Test:
    def __init__(self,model,unique_labels,unique_labels_str, test_loader, criterion, device,log,save_path):
        self._model=model
        self._test_loader=test_loader
        self._criterion=criterion
        self._device=device
        self._log=log
        self._save_path= os.path.join(save_path,"Test_confus.png") 
        self._unique_labels=unique_labels
        self._unique_labels_str=unique_labels_str
    
    def plot_confusion_matrix(self,cm):
        
        """Visualizes the confusion matrix using Seaborn."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self._unique_labels_str, yticklabels=self._unique_labels_str)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Test Phase')      

        # Save the figure
        plt.savefig(self._save_path, dpi=300, bbox_inches='tight')  # High-quality save
        
    def test_model(self):
        self._model.to(self._device)
        self._model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():  # No gradients needed for testing
            for inputs, labels in tqdm(self._test_loader, desc="Testing"):
                inputs, labels = inputs.to(self._device), labels.to(self._device)

                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted)
                all_labels.extend( labels.tolist())
                
        cm = confusion_matrix(torch.tensor(all_labels).cpu().numpy(),  torch.tensor(all_preds).cpu().numpy())
        self.plot_confusion_matrix(cm) 
                       
        accuracy = 100 * correct / total
        self._log.log(f"Test Loss: {running_loss/len(self._test_loader):.4f}")
        self._log.log(f"Test Accuracy: {accuracy:.2f}%")
        return 