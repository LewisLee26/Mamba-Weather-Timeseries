# Mamba Weather Timeseries

## Overview
This repository is for a Mamba model trained to predict weather timeseries data. 

## Dataset 
The model is trained on 8 varaibles from the GFS 0.25 degree dataset. The training dataset has a sample size of 36,160 and the test dataset has a sample size of 9,040. 

**Variables**
- Temperature
- Surface pressure
- V component of wind
- U component of wind
- Specific humidity
- Convective precipitation
- Total precipitation
- Water equivalent of accumulated snow depth

At each time step, data is taken from 200 coordinated. 
The data is normalized to fit within the range of -1 to 1. 

## Models
I trained two models, a [Mamba](https://arxiv.org/abs/2312.00752) and a [LSTM](https://arxiv.org/abs/1402.1128). 
Both models have the save parameters: 
- Hidden dimensions: 512
- Number of layers: 3

## Metrics 
| Variable | Mamba (MSE) | LSTM (MSE) |
|----------|-------------|------------|
|Temperature|1.6630e-05|**1.6136e-05**|
|Surface presure|**5.1565e-05**|7.3468e-05|
|V component of wind|**0.0008**|0.0023|
|U component of wind|**0.0003**|0.0020|
|Specific humidity|**0.0002**|0.0009|
|Convective precipitation|**5.1313e-05**|6.3685e-05|
|Total precipitation|**3.4177e-05**|4.7444e-05|
|Water equivalent of accumulated snow depth|1.4074e-06|**1.1085e-12**|
|Average|**0.00018**|0.00068|

*Lower MSE is better and shown in bold* 
