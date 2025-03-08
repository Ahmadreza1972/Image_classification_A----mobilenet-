import os
import torch

class Config:
    def __init__(self,mixed,supper):
        self._mixed_class_active=mixed
        self._supper_clas_act=supper
        self._set_directories()
        self._set_hyperparameters()
        self._set_model_parameters()
        self._set_device()

    def _set_directories(self):
        """Define all directory paths."""
        self._BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self._DATA_DIR = os.path.join(self._BASE_DIR, "../Data")
        self._OUTPUT_DIR = os.path.join(self._BASE_DIR, "../Outputs")
        
        name=""
        if self._supper_clas_act:
            name=name+"supper"
        else:
            name=name+"normal"
            
        if self._mixed_class_active:
            name=name+"/mixed"
        else:
            name=name+"/normal"
            

        self._MODEL_OUTPUT = os.path.join(self._OUTPUT_DIR, "model2/"+name)
        self._MODEL_OUTPUT_GRAPH = os.path.join(self._OUTPUT_DIR, "model2/"+name)  
        # Model 1 directories
        if self._mixed_class_active:
            if self._supper_clas_act:
                self._MODEL_TR_DATA=os.path.join(self._DATA_DIR, "model2/Mixed_suppermodel2_train.pth")
                self._MODEL_VAL_DATA=os.path.join(self._DATA_DIR, "model2/Mixed_suppermodel2_test.pth")
            else:
                self._MODEL_TR_DATA=os.path.join(self._DATA_DIR, "model2/Mixedmodel2_train.pth")
                self._MODEL_VAL_DATA=os.path.join(self._DATA_DIR, "model2/Mixedmodel2_test.pth")
        else:
            if self._supper_clas_act:
                self._MODEL_TR_DATA=os.path.join(self._DATA_DIR, "model2/model2_train_supercls.pth")
                self._MODEL_VAL_DATA=os.path.join(self._DATA_DIR, "model2/model2_test_supercls.pth")
            else:
                self._MODEL_TR_DATA=os.path.join(self._DATA_DIR, "model2/model2_train.pth")
                self._MODEL_VAL_DATA=os.path.join(self._DATA_DIR, "model2/model2_test.pth")
        self._SAVE_LOG= os.path.join(self._OUTPUT_DIR, "model2")

    def _set_hyperparameters(self):
        """Define all hyperparameters."""
        self._batch_size = 64
        self._learning_rate = 0.00005
        self._epochs = 350
        self._valdata_ratio = 0.3
        self._width_transform=32
        self._height_transform=32
        self._dropout=0.8
        self._weight_decay=0.00001
        self._momentum=0.95
        self._optimizer_type="adam"# "sgd"
        self._label_smoothing=0.4
        if self._mixed_class_active:
            self._epochs = 350
        if self._supper_clas_act:
            self._batch_size = 128
            
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
            "data_dir": os.path.join(self._DATA_DIR, "cifar100_classes.txt"),
            "train_path": self._MODEL_TR_DATA,
            "test_path": self._MODEL_VAL_DATA,
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
            "width_transform": self._width_transform,
            "drop_out":self._dropout,
            "weight_decay":self._weight_decay,
            "optimizer_type":self._optimizer_type,
            "momentum":self._momentum,
            "label_smoothing":self._label_smoothing
        }

    @property
    def model_parameters(self):
        """Return a dictionary of model parameters."""
        return {
            "num_classes": self._NUM_CLASSES,
            "device":self._DEVICE
        }
