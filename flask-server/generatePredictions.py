import numpy as np
import pandas as pd

import base64
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# pip install matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


from math import log,e
from scipy import stats

from datetime import datetime


import yfinance as yf
class Normalizer():
  def __init__(self):
    self.mu = None #mean
    self.sd = None #standard deviation

  def fit_transform(self,x):
    self.mu = np.mean(x, axis=(0))
    self.sd = np.std(x, axis=(0))
    normalized_x = (x - self.mu)/self.sd
    return normalized_x

  def inverse_transform(self, x):
      return (x*self.sd) + self.mu
  
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2) # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]


def generate_predictions(symbol,average_type):
    def buy_sell(bsm_call,market_call,bsm_put,market_put):
        statements = []
        market_call = np.float64(market_call)
        market_put = np.float64(market_put)
        statements.append("CALL: BSM value greater than market price :D " if bsm_call > market_call else "CALL: BSM value smaller than market price :( ")
        statements.append("PUT: BSM value greater than market price :D " if bsm_put > market_put else "PUT: BSM value smaller than market price :( ")
        return statements
    def get_data(config):
        hist = ticker.history(period='1y')
        data_date = hist.index


        data_close_price = hist['Close']
        num_data_points = len(data_close_price)

        data_date = data_date.strftime('%Y-%m-%d')

        display_date_range = data_date[0] + " to " + data_date[num_data_points-1]


        fig, ax = plt.subplots(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        ax.plot(data_date, data_close_price, color=config["plots"]["color_actual"])

        # Adjusting x-axis ticks
        xticks_interval = config["plots"]["xticks_interval"]
        xticks = data_date[::xticks_interval]  # Select every xticks_interval-th date
        x = np.arange(0, len(xticks) * xticks_interval, xticks_interval)
        ax.set_xticks(x)
        ax.set_xticklabels(xticks, rotation='vertical')

        ax.set_title("Daily close price for " + config["key_info"]["symbol"] + ", " + display_date_range)
        ax.grid(which='major', axis='y', linestyle='--')

        plot_name = 'get_data.png'
        plt.savefig(plot_name)
        plt.close()
        return data_date,data_close_price,num_data_points,display_date_range,plot_name
 

    def prepare_data_x(x, window_size):
    # perform windowing
        x = np.array(x)
        n_row = x.shape[0] - window_size + 1

        output = np.lib.stride_tricks.as_strided(x, shape=(n_row,window_size), strides=(x.strides[0],x.strides[0]))
        return output[:-1], output[-1]

    def prepare_data_y(x, window_size,average_type):
        if average_type == "SMA":
            output = x[window_size:]
            return output
        else:
            alpha = 2 / (window_size + 1)  # EMA smoothing factor
            ema = np.zeros_like(x)
            ema[0] = x[0]
            for i in range(1, len(x)):
                ema[i] = alpha * x[i] + (1 - alpha) * ema[i - 1]
            return ema[window_size:]



    def prepare_data(normalized_data_close_price, config, plot=False):
        data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
        data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"],average_type=average_type)

        # split dataset

        split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
        data_x_train = data_x[:split_index]
        data_x_val = data_x[split_index:]
        data_y_train = data_y[:split_index]
        data_y_val = data_y[split_index:]

        if plot:
            # prepare data for plotting

            to_plot_data_y_train = np.zeros(num_data_points)
            to_plot_data_y_val = np.zeros(num_data_points)

            to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(data_y_train)
            to_plot_data_y_val[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(data_y_val)


            to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
            to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

            ## plots

            fig = figure(figsize=(25, 5), dpi=80)
            fig.patch.set_facecolor((1.0, 1.0, 1.0))
            plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
            plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
            xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
            x = np.arange(0,len(xticks))
            plt.xticks(x, xticks, rotation='vertical')
            plt.title("Daily close prices for " + config["key_info"]["symbol"] + " - showing training and validation data")
            plt.grid(which='major', axis='y', linestyle='--')
            plt.legend()
            
            plot_name = 'split_train_test.png'
            plt.savefig(plot_name)
            plt.close()

        return split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen,plot_name
    config = {
        "key_info": {
        "symbol": symbol,
        "average_type":average_type,
            "outputsize": "full",
            "key_adjusted_close": "5. adjusted close",
        },
        "data": {
            "window_size": 20,
            "train_split_size": 0.80,
        },
        "plots": {
            "show_plots": True,
            "xticks_interval": 12,
            "color_actual": "#001f3f",
            "color_train": "#3D9970",
            "color_val": "#0074D9",
            "color_pred_train": "#3D9970",
            "color_pred_val": "#0074D9",
            "color_pred_test": "#FF4136",
        },
        "model": {
            "input_size": 1, #update based on what kind of techinical indicators user wants, i.e., rsi, Stochastic oscillator, etc,.
            "num_lstm_layers": 2,
            "lstm_size": 32,
            "dropout": 0.2,
        },
        "training": {
            "device": "cpu", # "cuda" or "cpu"
            "batch_size": 64,
            "num_epoch": 100, # from example, looks good enough
            "learning_rate": 0.01,
            "scheduler_step_size": 40,
        }
    }

    def run_epoch(dataloader, is_training=False):
        epoch_loss = 0

        if is_training:
            model.train()
        else:
            model.eval()

        for idx, (x, y) in enumerate(dataloader):
            if is_training:
                optimizer.zero_grad()

            batchsize = x.shape[0]

            x = x.to(config["training"]["device"])
            y = y.to(config["training"]["device"])

            out = model(x)
            loss = criterion(out.contiguous(), y.contiguous())

            if is_training:
                loss.backward()
                optimizer.step()

            epoch_loss += (loss.detach().item() / batchsize)

        lr = scheduler.get_last_lr()[0]

        return epoch_loss, lr
    #black scholes merton
    def black_scholes_merton (stock, strike,rate,time,volatility,dividend = 0.0):
        d1 = (log(stock/strike) +(rate-dividend+volatility**2/2)*time)/(volatility*time**0.5)
        d2 = d1 - volatility*time**0.5

        call = stats.norm.cdf(d1)*stock*e**(-dividend*time)-stats.norm.cdf(d2)*strike*e**(-rate*time)
        put = stats.norm.cdf(-d2)*strike*e**(-rate*time)-stats.norm.cdf(-d1)*stock*e**(-dividend*time)

        return call,put

    ticker = yf.Ticker(config['key_info']['symbol'])
    data_date, data_close_price,num_data_points,display_date_range,get_data_plot = get_data(config)
       

    scaler = Normalizer()
    normalized_data_close_price = scaler.fit_transform(data_close_price)
    split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen,split_plot = prepare_data(normalized_data_close_price, config, plot=config["plots"]["show_plots"])

    dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
    dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

    model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
    model = model.to(config["training"]["device"])


    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

    # define optimizer, scheduler and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

    # begin training
    for epoch in range(config["training"]["num_epoch"]):
        loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
        loss_val, lr_val = run_epoch(val_dataloader)
        scheduler.step()

        #print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
         #       .format(epoch+1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))
        
    # here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date

    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

    model.eval()

    # predict on the training data, to see how well the model managed to learn and memorize

    predicted_train = np.array([])

    for idx, (x, y) in enumerate(train_dataloader):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_train = np.concatenate((predicted_train, out))

# predict on the validation data, to see how the model does

    predicted_val = np.array([])

    for idx, (x, y) in enumerate(val_dataloader):
        x = x.to(config["training"]["device"])
        out = model(x)
        out = out.cpu().detach().numpy()
        predicted_val = np.concatenate((predicted_val, out))

    if config["plots"]["show_plots"]:

        # prepare data for plotting, show predicted prices

        to_plot_data_y_train_pred = np.zeros(num_data_points)
        to_plot_data_y_val_pred = np.zeros(num_data_points)

        to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
        to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

        to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

        # plots

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
        plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
        plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
        plt.title("Compare predicted prices to actual prices")
        xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
        x = np.arange(0,len(xticks))
        plt.xticks(x, xticks, rotation='vertical')
        plt.grid(which='major', axis='y', linestyle='--')
        plt.legend()

        split_data = 'split_data.png'
        plt.savefig(split_data)
        plt.close()

        # prepare data for plotting, zoom in validation

        to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
        to_plot_predicted_val = scaler.inverse_transform(predicted_val)
        to_plot_data_date = data_date[split_index+config["data"]["window_size"]:]

        # plots

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["plots"]["color_actual"])
        plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
        plt.title("Zoom in to examine predicted price on validation data portion")
        xticks = [to_plot_data_date[i] if ((i%int(config["plots"]["xticks_interval"]/5)==0 and (len(to_plot_data_date)-i) > config["plots"]["xticks_interval"]/6) or i==len(to_plot_data_date)-1) else None for i in range(len(to_plot_data_date))] # make x ticks nice

        xs = np.arange(0,len(xticks))
        plt.xticks(xs, xticks, rotation='vertical')
        plt.grid( which='major', axis='y', linestyle='--')
        plt.legend()

        validation_data = 'valid_data.png'
        plt.savefig(validation_data)
        plt.close()

        # predict on the unseen data, tomorrow's price

    model.eval()

    x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
    prediction = model(x)
    prediction = prediction.cpu().detach().numpy()
    prediction = scaler.inverse_transform(prediction)[0]

    next_td = 'next_td.png'
    if config["plots"]["show_plots"]:

        # prepare plots

        plot_range = 10
        to_plot_data_y_val = np.zeros(plot_range)
        to_plot_data_y_val_pred = np.zeros(plot_range)
        to_plot_data_y_test_pred = np.zeros(plot_range)

        to_plot_data_y_val[:plot_range-1] = scaler.inverse_transform(data_y_val)[-plot_range+1:]
        to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

        to_plot_data_y_test_pred[plot_range-1] = prediction

        to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
        to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
        to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

        # plot

        plot_date_test = list(data_date[-plot_range+1:])
        plot_date_test.append("next trading day")

        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
        plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
        plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
        plt.title("Predicted close price of the next trading day")
        plt.grid(which='major', axis='y', linestyle='--')
        plt.legend()

        
        plt.savefig(next_td)
        plt.close()

    #print("Predicted close price of the next trading day:", round(prediction, 2))

        #relevant data for options trading, front month
    def option_chain(ticker):
        exp_dates = ticker.options
        options = ticker.option_chain(exp_dates[4])
        calls = options.calls
        puts = options.puts
        return calls, puts, exp_dates

    options = option_chain(ticker)

    call_options = options[0]
    put_options = options[1]
    exp_dates = options[2]

    #relevant for bsm model
    def days_between(d1, d2):
        d1 = datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.strptime(d2, "%Y-%m-%d")
        return abs((d2 - d1).days)
    days = days_between(datetime.today().strftime('%Y-%m-%d'),exp_dates[4])

    #determining what is otm/atm based on the predicted price from the lstm model
    def atm_check(options,price):
        l = 0
        r = len(options) - 1
        while l < r:
            mid = l + (r - l)//2
            if options.strike.iloc[mid] == price:
            #found the atm option, otm option is then obviously the next call/put (mid + 1)/(mid - 1)
                return mid
            elif options.strike.iloc[mid] > price:
                r = mid
            else:
                l = mid + 1
        return l
    
    model.eval()

    x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
    predictions = []

    for _ in range(days):  # Predict for the next 30 days
        prediction = model(x)
        predictions.append(prediction.item())
        x = torch.cat((x, prediction.unsqueeze(1).unsqueeze(2)), dim=1)  # Adjust concatenation dimension

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    #print(f"Predicted close price in {days} days: {float(predictions[-1]):.2f}")
    close_front_month = f"Predicted close price in {days} days: {float(predictions[-1]):.2f}"
    atm_call = atm_check(call_options,predictions[-1:]*1.05) #range of price is prediction +- 5% - 10%, currently using 5%
    atm_put = atm_check (put_options,predictions[-1:]*0.95)
    #collar
    predicted_price = float(predictions[-1])
    curr_price = ticker.history(period = '1d')['Close'][-1]
    strike_put = put_options.iloc[atm_put]  # what the strike should be, determine option using this
    strike_call = call_options.iloc[atm_call]
    interest_rate = yf.Ticker('^TNX').get_info()['previousClose']*0.01
    time_to_expiration = days/365

    #note that i am assuming ALL ticker dividends are 0
    call = black_scholes_merton(curr_price, strike_call.strike, interest_rate, time_to_expiration, strike_call.impliedVolatility)[0]
    put = black_scholes_merton(curr_price, strike_put.strike, interest_rate, time_to_expiration, strike_put.impliedVolatility)[1]

    #print(buy_sell(call,strike_call.iloc[atm_call],put,strike_put.iloc[atm_put]))
    collar_bsm =buy_sell(call,strike_call.lastPrice,put,strike_put.lastPrice)
    net_premium = (call- put)
    collar_print= f"Buying options that expire in {days} days and calls and puts at strike price: ${strike_call.strike}"
    prices = np.arange(curr_price*0.8,curr_price*1.2,0.1)
    collar_list = []
    #for chart
    for stock in prices:
        if stock >strike_call.strike:
            collar_list.append((strike_call.strike+net_premium)*100)
        elif stock < strike_put.strike:
            collar_list.append((strike_put.strike+net_premium)*100)
        else:
            collar_list.append((stock+net_premium)*100)

        if predicted_price > strike_call.strike:
            # If the predicted price is above the strike price of the call option
            collar_pl = (strike_call.strike + net_premium) - predicted_price
        elif predicted_price < strike_put.strike:
            # If the predicted price is below the strike price of the put option
            collar_pl = (strike_put.strike + net_premium) - predicted_price
        else:
            # If the predicted price is between the strike prices of the call and put options
            collar_pl = (predicted_price + net_premium)  - predicted_price

    collar_print_2 = f"If options are exercised, predicted profit/loss: ${collar_pl:.2f} per share." 
    #print(f"If options are exercised, predicted profit/loss: ${collar_pl:.2f} per share." )
    plt.plot(np.arange(curr_price*0.8,curr_price*1.2,0.1),collar_list,label='Collar Strategy',lw=3)
    plt.plot(prices,prices*100,label='Stock Position',lw=5)
    plt.xticks(np.arange(curr_price*0.8,curr_price*1.2,20))
    plt.xlabel('Stock Price')
    plt.ylabel('Position Valuation (assuming you hold 100 shares)')
    plt.title('At the end of the following month (Collar Strategy)')
    plt.grid()
    plt.legend()

    collar_plot = 'collar_plot.png'
    plt.savefig(collar_plot)
    plt.close()


        #this strat is to be long the straddle (i.e., same expiration and same strike) buy it at the current price NOT prediction price
    straddle_call = atm_check(call_options,curr_price)
    straddle_put = atm_check(put_options,curr_price)

    #print(buy_sell(call,call_options.iloc[straddle_call].lastPrice,put,put_options.iloc[straddle_put].lastPrice))
    straddle_bsm = buy_sell(call,call_options.iloc[straddle_call].lastPrice,put,put_options.iloc[straddle_put].lastPrice)
    underlying_prices = np.arange(curr_price*0.8,curr_price*1.2)
    straddle_profit = []
    straddle_print=f'Purchasing a call AND put at strike: ${call_options.iloc[straddle_call].strike} that expire in {days} days.'
    #print(f'Purchasing a call AND put at strike: {call_options.iloc[straddle_call].strike}')
    #for chart
    for price in underlying_prices:
        call_payoff = max(0, price - curr_price) - call_options.iloc[straddle_call].lastPrice
        put_payoff = max(0, curr_price - price) - put_options.iloc[straddle_put].lastPrice
        total_payoff = call_payoff + put_payoff
        straddle_profit.append(total_payoff)

    call_payoff = max(0, predicted_price - call_options.iloc[straddle_call].strike) - call_options.iloc[straddle_call].lastPrice
    put_payoff = max(0, put_options.iloc[straddle_put].strike - predicted_price) - put_options.iloc[straddle_put].lastPrice
    straddle_pl = call_payoff + put_payoff
    #print(f'If options are exercised, predicted profit/loss: ${straddle_pl:.2f} per share')
    straddle_print_2 = f'If options are exercised, predicted profit/loss: ${straddle_pl:.2f} per share'
    # Plot the profit/loss diagram
    plt.plot(underlying_prices, straddle_profit)
    plt.xlabel('Underlying Asset Price')
    plt.ylabel('Profit/Loss')
    plt.title('Straddle Profit/Loss Diagram')
    plt.grid(True)
    straddle_plot = 'straddle_plot.png'
    plt.savefig(straddle_plot)
    plt.close()

        #strangle
    strangle_call = atm_check(call_options,curr_price*1.05)

    strangle_put = atm_check(put_options,curr_price*0.95)

    #print(buy_sell(call,call_options.iloc[strangle_call].lastPrice,put,put_options.iloc[strangle_put].lastPrice))
    strangle_bsm =buy_sell(call,call_options.iloc[strangle_call].lastPrice,put,put_options.iloc[strangle_put].lastPrice)
    strangle_profit = []
    #print(f'Purchasing a call at: {call_options.iloc[strangle_call].strike} and put at : {put_options.iloc[strangle_put].strike}')
    strangle_print = f'Purchasing a call at: ${call_options.iloc[strangle_call].strike} and put at: ${put_options.iloc[strangle_put].strike} that both expire in {days} days'
    #for chart
    for price in underlying_prices:
        call_payoff = max(0, price -  call_options.iloc[strangle_call].strike) - call_options.iloc[strangle_call].lastPrice
        put_payoff = max(0, put_options.iloc[strangle_put].strike - price) - put_options.iloc[strangle_put].lastPrice
        total_payoff = call_payoff + put_payoff
        strangle_profit.append(total_payoff)



    call_payoff = max(0, predicted_price - call_options.iloc[strangle_call].strike) - call_options.iloc[strangle_call].lastPrice
    put_payoff = max(0, put_options.iloc[strangle_put].strike - predicted_price) - put_options.iloc[strangle_put].lastPrice
    strangle_pl = call_payoff + put_payoff

    #print(f'If options are exercised, predicted profit/loss: ${strangle_pl:.2f} per share')
    strangle_print_2 = f'If options are exercised, predicted profit/loss: ${strangle_pl:.2f} per share'
    # Plot the profit/loss diagram
    plt.plot(underlying_prices, strangle_profit)
    plt.xlabel('Underlying Asset Price')
    plt.ylabel('Profit/Loss')
    plt.title('Strangle Profit/Loss Diagram')
    plt.grid(True)
    strangle_plot = 'strangle_plot.png'
    plt.savefig(strangle_plot)
    plt.close()

    #the only things this returns is the graphs, and the statements that go along with the graphs, graphs first statements second
    return get_data_plot,split_plot,split_data,validation_data,next_td,collar_plot,straddle_plot,strangle_plot,close_front_month,collar_bsm,collar_print,collar_print_2,strangle_bsm,strangle_print,strangle_print_2,straddle_bsm,straddle_print,straddle_print_2
