import os
import torch

class Config:
    def __init__(self):
        self._set_directories()
        self._set_hyperparameters()
        self._set_model_parameters()
        self._set_device()

    def _set_directories(self):
        """Define all directory paths."""
        self._BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._DATA_DIR = os.path.join(self._BASE_DIR, "Data")
        self._OUTPUT_DIR = os.path.join(self._BASE_DIR, "Outputs")
        
        self._MODEL1_WEIGHTS=os.path.join(self._OUTPUT_DIR, "model1\\model1_weights.pth")
        self._MODEL2_WEIGHTS=os.path.join(self._OUTPUT_DIR, "model2\\model2_weights.pth")
        self._MODEL3_WEIGHTS=os.path.join(self._OUTPUT_DIR, "model3\\model3_weights.pth")

        # Model directories
        self._MODEL_DATA = os.path.join(self._DATA_DIR, "fusion/TaskB_fusion_test.pth")
        self._MODEL1_DATA = os.path.join(self._DATA_DIR, "model1/model1_test.pth")
        self._MODEL2_DATA = os.path.join(self._DATA_DIR, "model2/model2_test.pth")
        self._MODEL3_DATA = os.path.join(self._DATA_DIR, "model3/model3_test.pth")
        self._GROUP_LABEL=os.path.join(self._DATA_DIR, "fusion/cifar100_classes.txt")
        self._SAVE_LOG= os.path.join(self._OUTPUT_DIR, "fusion")

    def _set_hyperparameters(self):
        """Define all hyperparameters."""
        self._batch_size = 1
        self._learning_rate = 0.001
        self._epochs = 20
        self._valdata_ratio = 0.2
        self._width_transform=64
        self._height_transform=64

    def _set_model_parameters(self):
        """Define model-specific parameters."""
        self._NUM_CLASSES = 5
        
    def _set_device(self):
        """Check for CUDA (GPU)"""
        self._DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"-------Using device: {self._DEVICE}")    

    @property
    def directories(self):
        """Return a dictionary of directory paths."""
        return {
            "data_path": self._MODEL_DATA,
            "model1_data_path":self._MODEL1_DATA,
            "model2_data_path":self._MODEL2_DATA,
            "model3_data_path":self._MODEL3_DATA,
            "model1_weights": self._MODEL1_WEIGHTS,
            "model2_weights": self._MODEL2_WEIGHTS,
            "model3_weights": self._MODEL3_WEIGHTS,
            "group_labels":self._GROUP_LABEL,
            "save_log":self._SAVE_LOG
        }

    @property
    def hyperparameters(self):
        """Return a dictionary of hyperparameters."""
        return {
            "batch_size": self._batch_size,
            "learning_rate": self._learning_rate,
            "epochs": self._epochs,
            "valdata_ratio": self._valdata_ratio,
            "height_transform": self._height_transform,
            "width_transform": self._width_transform
        }

    @property
    def model_parameters(self):
        """Return a dictionary of model parameters."""
        return {
            "num_classes": self._NUM_CLASSES,
            "device":self._DEVICE
        }
