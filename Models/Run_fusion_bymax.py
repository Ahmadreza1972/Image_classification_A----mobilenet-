import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.fus_baseMosel import FusBaseModel

class ByMax(FusBaseModel):
    def __init__(self,Mixed,Supper,method):
        self._mixed=Mixed
        self._supper=Supper
        self._method=method
        super().__init__(self._mixed,self._supper,method)
        
    def get_final_estimation_bymax(self):
        total_correct=0
        total_row=0
        data_iter = iter(self._dataloader)
        for row in range(len(self._results)):
            if self._Mixed_Class_Activation:
                pr_row=[]
                for i in range(3):
                    p=np.array(self._results.loc[row][f"model {i+1} prp"])
                    if self._orginal_labels[i][np.argmax(p)]==6:
                        pr1=0 
                    else:
                        pr1=max(p)
                    pr_row.append(pr1)  
            else:  
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
      
model=ByMax(Mixed=True,Supper=True,method="ByMax")   #["ByMax","DS","TrainMeta"]  --DS just work with mix data
model.models_output_colector()
model.get_final_estimation_bymax()
