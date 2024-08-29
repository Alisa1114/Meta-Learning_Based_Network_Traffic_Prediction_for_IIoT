# Meta-Learning based Network Traffic Prediction
This is the implementation of our paper **Meta-Learning Based Network Traffic Prediction for IIoT**


# Get Started

### Environment
Use the command:

    pip install -r requirements.txt

to install the required packages.


### Datasets and Data Preparation

Please download the following datasets from:

+ [FedCSIS](https://knowledgepit.ai/fedcsis20-challenge/)

+ [Guangzhou](https://zenodo.org/records/1205229)

+ [Seattle](https://github.com/zhiyongc/Seattle-Loop-Data)

and put the dataset into the folder of the same name. For example, the data of FedCSIS should put into the folder named as "FedCSIS".

### Train

Execute this command at the root directory to train the MAML model: 

    python train.py

### Run Demo / Test with Pretrained Models
Execute the command: 

    python test.py

or you want to compare the MAML model to baseline model each epoch:
    
    python test_and_plot.py



# References

This repo uses [NetworkTrafficPrediction](https://github.com/mliwang/NetworkTrafficPrediction.git). Thanks for their great work!