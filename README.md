# DMCL

**Program Description**

DMCL is cancer subtype framwork based on multi-omics cancer datasets.The following is the function of the specific files in the program:
    
    
    processDatasets.py:对数据进行归一化预处理
    pretrainModel.py:预训练部分
    networkModels.py:论文中提出来的架构
    modelLoss.py:架构中需要的损失
    DMCL.py:执行的文件
    
    

**Requirements**

	>= Python 3.7.9

**Usage**

   To execute our algorithm, please load the python file: DMCL.py into your python interpreter and click the 'run' button. Users can also execute the program in command-line mode, the specific command is as follows:```python DMCL.py```
   
   All datasets can be viewed and downloaded from this website: https://github.com/alcs417/CGGA/tree/main/cancer_datasets  

**Parameters**

   There are two parameters in our algorithm which use to balance the loss in total loss fuction.gamma is the coefficient of clustering loss and beta is the coefficient of contrastive loss.
   
**Input and Output Directories**

   The datasets used by the program is in the ```data``` folder.The output files are under the ```result``` folder and separated by the name of the dataset.i.e the results of liver are in ```result/liver/```. The output are matlab files, where the label variable is the predicted label.
   
**Contact**

   For any questions regarding our work, plese feel free to contact us：cwlczt@163.com

