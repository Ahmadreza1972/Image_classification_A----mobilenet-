import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Config.config_model2 import Config
from Models.Base_model import BaseModle

class ModelProcess(BaseModle):
    def __init__(self):
        self._supper_class_activation=True
        self._Mixed_Class_Activation=True
        self._config=Config(self._Mixed_Class_Activation,self._supper_class_activation)
        super().__init__(self._config,"model2",Mixed=self._Mixed_Class_Activation,supper=self._supper_class_activation)

model=ModelProcess()
model.main()
