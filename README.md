<div align="center">

# Decentralized Federated Learning with ROCKET features
Anonymous submission
</div>

# Overview

This repository contains the code to run and reproduce the experiments of DROCKS, a decentralized federated learning method using ROCKET features for time series classification. 

Please cite as:

```bibtex
@inproceedings{,
  author  = {},
  title   = {Decentralized Federated Learning with ROCKET features,
  booktitle    = {},
  year         = {},
  doi          = {},
  pages = {},
  publisher = {},
  url = {}
}
```

# Abstract
Most Federated Learning (FL) solutions are based on an aggregator-client (i.e., master-worker) architecture, which is simple and effective. However, it also exhibits serious robustness and confidentiality issues related to the distinguished role of the master, which is a single point of failure and can observe knowledge extracted from workers. We propose DROCKS, a fully decentralized FL approach for time series classification using ROCKET (RandOm Convolutional KErnel Transform) features. In DROCKS, the global model is trained by sequentially traversing a path across all federation nodes. Each node receives the global model from its predecessor, uses it to compute the local best-performing kernels, and sends the model and the best-performing local kernels to its successor along the path. Results show that DROCKS outperforms state-of-the-art federated approaches on most of the datasets of the UCR archive. Additionally, we demonstrate that DROCKS is less demanding regarding communication and computation than the traditional schema. 

# Method
 <p align = "center"><img src="results/method.pdf" width="600" style = "text-align:center"/></p>

## Usage
- Clone this repo.
- Download and unzip the UCRArchive Time Series Classification dataset: [UCRArchive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/), and place it in the same directory of the repository.
- Install the requirements. 
- Run 'python3 drocks.py'.

## Results
The results directory reports the accuracy and F1-score metrics for all the UCR Archive datasets. This folder contains also the critical difference diagrams, a useful tool for comparing the performance of DROCKS and competitors over all the considered datasets.


## Contributors
* Anonymous submission