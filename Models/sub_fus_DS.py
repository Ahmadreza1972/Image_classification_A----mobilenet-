import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Models.fus_baseMosel import FusBaseModel

class ByDS(FusBaseModel):
    def __init__(self,Mixed,Supper,method):
        self._mixed=Mixed
        self._supper=Supper
        self._method=method
        super().__init__(self._mixed,self._supper,method)
        
    def combine_beliefs(self,belief1, belief2, unknown_label=6, alpha=0.5):
        """
        Implements Dempster's rule of combination for two mass functions.
        belief1 and belief2 are dictionaries where keys are numeric labels.
        """
        combined = {}
        conflict = 0.0  # Conflict factor
        if 6 in belief1.keys():
            uukn_pr=belief1[6]/10
            del belief1[6]
            for j in range(3):
                for i in self._orginal_labels[j]:
                    if ((i not in belief1.keys()) and (i!=6)):
                        belief1[i]=uukn_pr

        if 6 in belief2.keys():
            uukn_pr=belief2[6]/10
            del belief2[6]
            for j in range(3):
                for i in self._orginal_labels[j]:
                    if ((i not in belief2.keys()) and (i!=6)):
                        belief2[i]=uukn_pr        

        for key1, mass1 in belief1.items():
            temp={}
            for key2, mass2 in belief2.items():
                temp[key2]=mass1 * mass2
                if key1 != key2:  # Matching labels → Normal combination
                    conflict += mass1 * mass2
            combined[key1] = temp
        # Normalize the combined mass function to remove conflict
        final = {key: 0 for key in belief2.keys()}
        if conflict < 1.0:
            for key1, mass1 in combined.items():
                for key2,mass2 in mass1.items():
                   final[key1]+= mass2
                   final[key2]+= mass2
            sum=0       
            for key in final:
                final[key] /= (1 - conflict)    
                sum+=final[key]
            for key in final:
                final[key] /=sum
        else:
            raise ValueError("Total conflict: Dempster's rule fails.")
        return final

    def get_final_by_DS(self):
        total_correct=0
        total_row=0
        data_iter = iter(self._dataloader)
        for row in range(len(self._results)):
            # Convert model outputs into mass functions
            fusion=self.combine_beliefs(dict(zip(self._orginal_labels[0],self._results.loc[row]["model 1 prp"])), dict(zip(self._orginal_labels[1],self._results.loc[row]["model 2 prp"])))
            fusion=self.combine_beliefs(fusion, dict(zip(self._orginal_labels[2],self._results.loc[row]["model 3 prp"])))
            predictedlabel = max(fusion, key=fusion.get)
            true_labels=self._results.loc[row]["True label"]
            images, labels = next(data_iter)  # Fetch a batch
            image = images[row % len(images)]
            if row % 100 ==0:
                self.save_result(image,labels,predictedlabel,row)
            total_row+=1
            if (true_labels==predictedlabel):
                total_correct+=1
        accuracy=total_correct/total_row        
        self._log.log(f"Meta-Model Accuracy:{accuracy}" )
      
model=ByDS(Mixed=True,Supper=True,method="DS")# --DS just work with mix data
model.models_output_colector()
model.get_final_by_DS()
