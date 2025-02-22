import torch
from tqdm import tqdm

class Test:
    def __init__(self,model, test_loader, criterion, device,log):
        self._model=model
        self._test_loader=test_loader
        self._criterion=criterion
        self._device=device
        self._log=log
        
    def test_model(self):
        self._model.to(self._device)
        self._model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():  # No gradients needed for testing
            for inputs, labels in tqdm(self._test_loader, desc="Testing"):
                inputs, labels = inputs.to(self._device), labels.to(self._device)

                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        self._log.log(f"Test Loss: {running_loss/len(self._test_loader):.4f}")
        self._log.log(f"Test Accuracy: {accuracy:.2f}%")