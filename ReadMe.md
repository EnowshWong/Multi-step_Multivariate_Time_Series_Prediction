## Multi-Step Multivariate Time Series Prediction Using Funds Data
This program using TCN(Time Convolution Networks) Combining the Bayesian Optimization(BO-TCN), LSTM, Seq2seq and Random Forest to solve the MSMTSP question.
We use the hyper-parameters optimization tool optuna to get the best parameters.
Please see in TCN.funds_prediction.find_hyperparameters.py.

## Task Description
Mathematically, given the time series:
x1,x2,...,xt
, we will predict the next t values with
yt+1,yt+2,...,yt+p
and the input x and output y are both D-dimension tensors.
We use direct prediction way rather than iterative way to make
 multi-step time series prediction. Although it may result in 
 losing a fixed length of training data, it prevent the error 
 accumulation and propagation. 

We compare the prediction results with different models and achieve the state of art in BO-TCN mode.

## Requirements
- tensorflow 1.13
- pytorch 1.0
- keras
- optuna
- sk-learn


