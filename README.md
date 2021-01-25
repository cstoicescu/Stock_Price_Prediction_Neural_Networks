# Stock_Price_Prediction_Neural_Networks
Model designed to predict stock prices of any related time series data.  

## ABOUT
Recurrent neural networks (RNN) have proved one of the most powerful models for processing sequential data. Long Short-Term memory is one of the most successful RNNs architectures. LSTM introduces the memory cell, a unit of computation that replaces traditional artificial neurons in the hidden layer of the network. With these memory cells, networks are able to effectively associate memories and input remote in time, hence suit to grasp the structure of data dynamically over time with high prediction capacity.

## Train Data is Provided from Yahoo Finance: 
<img src="https://i.imgur.com/Qo9NiSZ.jpg" />



What we suspect: real TESLA ( TSLA )  Stock price is in red and the predicted stock price is in blue. The predicted stock price is based upon real stock price and then after a split point the RNN is taking the current amount and adding a little amount it percentage, and that's its  basically prediction. For optimizing, i used the adam algorithm with a loss function of mean_squared_error.  


## Results  

<img src="https://i.imgur.com/xZGEDrZ.jpg" />

# Conclusions  

<img src="https://i.imgur.com/OcTpKDY.jpg" />

## Technologies Used  
- Python  

- Pandas  

- Numpy  

- Keras  
