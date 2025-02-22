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
        self._BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self._DATA_DIR = os.path.join(self._BASE_DIR, "../Data")
        self._OUTPUT_DIR = os.path.join(self._BASE_DIR, "../Outputs")

        # Model 1 directories
        self._MODEL1_TR_DATA = os.path.join(self._DATA_DIR, "model3/model3_train.pth")
        self._MODEL1_VAL_DATA = os.path.join(self._DATA_DIR, "model3/model3_test.pth")
        self._MODEL_OUTPUT = os.path.join(self._OUTPUT_DIR, "model3/model3_weights.pth")
        self._MODEL_OUTPUT_GRAPH = os.path.join(self._OUTPUT_DIR, "model3/training_results.png")
        self._SAVE_LOG= os.path.join(self._OUTPUT_DIR, "model3")

    def _set_hyperparameters(self):
        """Define all hyperparameters."""
        self._batch_size = 128
        self._learning_rate = 0.0005
        self._epochs = 40
        self._valdata_ratio = 0.3
        self._width_transform=32
        self._height_transform=32

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
            "train_path": self._MODEL1_TR_DATA,
            "test_path": self._MODEL1_VAL_DATA,
            "save_path": self._MODEL_OUTPUT,
            "output_graph": self._MODEL_OUTPUT_GRAPH,
            "save_log": self._SAVE_LOG
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
