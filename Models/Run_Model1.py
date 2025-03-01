import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Config.config_model1 import Config
from Models.Base_model import BaseModle

class ModelProcess(BaseModle):
    def __init__(self):
        self._config=Config()
        super().__init__(self._config,"model1")
        
model=ModelProcess()
model.main()
