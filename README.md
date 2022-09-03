# DMCL

**Program Description**

DMCL is a cancer subtype framework based on multi-omics cancer datasets. The following is the function of the specific files in the program:
    
    
    processDatasets.py: File for data import and data normalization
    pretrainModel.py: A pre_training file for the model
    networkModels.py: The architecture designed in the paper
    modelLoss.py: The losses in architecture
    DMCL.py: The main file for execution
    
    

**Requirements**

  >= Python 3.7.9

**Usage**

   To execute our algorithm, please load the python file: DMCL.py into your python interpreter and click the 'run' button. Users can also execute the program in command-line mode, the specific command is as follows:```python DMCL.py```
   
   All datasets can be viewed and downloaded from this website: https://github.com/alcs417/CGGA/tree/main/cancer_datasets  

**Parameters**

   There are two parameters in our algorithm which use to balance the loss in the total loss function. gamma is the coefficient of clustering loss and beta is the coefficient of contrastive loss.
   
**Input and Output Directories**

   The datasets used by the program are in the ```data``` folder. The output files are under the ```result``` folder and separated by the name of the dataset. i.e the results of the liver are in ```result/liver/```. The output is Matlab files, where the label variable is the predicted label.
   
**Contact**

   For any questions regarding our work, please feel free to contact us：cwlczt@163.com

