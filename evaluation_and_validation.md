---
title: Evaluation and Validation
notebook:
nav_include: 4
---

## Overview

Each policy will be compared with several benchmark portfolios constructed using traditional portfolio optimization techniques.  The portfolio benchmarks that we will use are: mean-variance optimization, equal risk contribution and hierarchical risk parity. [1] 

### Performance Metrics

To evaluate our model, we will split the dataset in train and test sets. After training our models, we will backtest the strategy using the test set. The performance metrics that we will use for the evaluation will be the following:

#### Sharpe Ratio

The Sharpe Ratio is a measure that indicates the average return minus the risk-free return divided by the standard deviation of return on an investment, also known as volatility. We have selected to use the Sharpe Ratio as one of our performance metrics since it measures the performance of our portfolio compared to the risk-free asset, after adjusting for its risk. The Sharpe Ratio characterizes how well the return of an asset compensates the investor for the risk taken.

Sharpe Ratio Formula:

<img src="https://render.githubusercontent.com/render/math?math=S_{a} = \frac{E[R_{a} - R_{b}]}{\sigma_{a}} = \frac{E[R_{a} - R_{b}]}{\sqrt{var[R_{a} - R_{b}]}}">

where ***R*<sub>*a*</sub>** is the asset return, ***R*<sub>*b*</sub>** is the risk-free return.

#### Sortino Ratio

The Sortino Ratio measures the risk-adjusted return of the portfolio. The Sortino Ratio is similar to the Sharpe Ratio, but instead factors in a penalization for those returns falling below a target rate of return.  Generally, the Sortino ratio is used as a way to compare the risk-adjusted performance of strategies with differing risk and return characteristics. 

Sortino Ratio Formula:

<img src="https://render.githubusercontent.com/render/math?math=S = \frac{R - T}{DR}">

where ***R*** is the portfolio average realized return, ***T*** is the target rate of return for the investment strategy, ***DR*** is the square root of the target semivariance (also known as downside deviation). 


#### Calmar Ratio
Calmar Ratio is a function of the fund’s average compounded annual rate of return versus its maximum drawdown.  The benefits of the Calmar Ratio is that it changes gradually and serves to smooth out the over achievement and underachievement periods.

Calmar Ratio Formula:

<img src="https://render.githubusercontent.com/render/math?math=\text{Calmar Ratio} = \frac{R_{p} - R_{f}}{Maximum Drawdown}">

where ***R*<sub>*p*</sub>** is the portfolio return, ***R*<sub>*f*</sub>** is the risk-free return, and (***R*<sub>*p*</sub> − *R*<sub>*f*</sub>**) is also known as the average annual rate of return. 

#### Draw-Downs

Draw-Downs are a measure of peak-to-trough decline during a specific period of time for an investment. Draw-Downs are important for measuring the historical risk of different investments.

Draw-Downs Formula:

<img src="https://render.githubusercontent.com/render/math?math=D(T) = max[\max_{t\varepsilon (0, T)}X(t) - X(T), 0] \equiv [\max_{t\varepsilon (0, T)}X(t) - X(T)]">

where ***T*** is time and ***X*(*T*)** is the value of the asset at time ***T***.  

#### Volatility

Volatility refers to the amount of uncertainty or risk related to the size of changes in a security's value. A higher volatility means that a security's value can potentially be spread out over a larger range of values. This means that the price of the security can change dramatically over a short time period in either direction. A lower volatility means that a security's value does not fluctuate dramatically, and tends to be more steady.

Volatility Formula:

<img src="https://render.githubusercontent.com/render/math?math=\sigma_T = \sigma * \sqrt{T}">

where ***T*** is time.

### Training and Testing Data Set

To estimate the performance of our reinforcement learning algorithms, we will create training and testing datasets. These datasets will be used to validate the performance of our models.

We begin by dividing the Real Dataset ETF Price History's into two datasets, the training set which includes ETF price history from January 2017 to March 2020, and a test dataset with ETF price history from April 2020 to November 2020.


[1] We will use the public library PyPortfolioOpt. More details can be found in here https://pyportfolioopt.readthedocs.io/en/latest/
