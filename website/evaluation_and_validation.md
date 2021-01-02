---
title: Evaluation and Validation
notebook:
nav_include: 4
---

## Overview

Each policy will be compared with several benchmark portfolios constructed using traditional portfolio optimization techniques. The portfolio benchmarks that we will use are: maximum return, maximum Sharpe ratio, and minimum volatility. [1]

### Performance Metrics

The performance metrics are computed using the Critical Line Algorithm (CLA). The CLA is a robust alternative to the quadratic solver used to find mean-variance optimal portfolios, that is especially advantageous when we apply linear inequalities.

To evaluate our model, we will split the dataset in train and test sets. After training our models, we will backtest the strategy using the test set. The performance metrics that we will use for the evaluation will be the following:

#### Maximum Return

The maximum return is a simple performance metric which is based on the maximum return the portfolio. We will utilize the CLA algorithm and compute the entire efficient frontier and extract the maximum return weights.

#### Sharpe Ratio

The Sharpe Ratio is a measure that indicates the average return minus the risk-free return divided by the standard deviation of return on an investment, also known as volatility. We have selected to use the Sharpe Ratio as one of our performance metrics since it measures the performance of our portfolio compared to the risk-free asset, after adjusting for its risk. The Sharpe Ratio characterizes how well the return of an asset compensates the investor for the risk taken.

Sharpe Ratio Formula:

<img src="https://render.githubusercontent.com/render/math?math=S_{a} = \frac{E[R_{a} - R_{b}]}{\sigma_{a}} = \frac{E[R_{a} - R_{b}]}{\sqrt{var[R_{a} - R_{b}]}}">

where ***R*<sub>*a*</sub>** is the asset return, ***R*<sub>*b*</sub>** is the risk-free return.

#### Volatility

Volatility refers to the amount of uncertainty or risk related to the size of changes in a security's value. A higher volatility means that a security's value can potentially be spread out over a larger range of values. This means that the price of the security can change dramatically over a short time period in either direction. A lower volatility means that a security's value does not fluctuate dramatically, and tends to be more steady.

Volatility Formula:

<img src="https://render.githubusercontent.com/render/math?math=\sigma_T = \sigma * \sqrt{T}">

where ***T*** is time.

### Observed Metrics

To evaluate the performance of our models and the performance of our raw data, we will observe the following statistics in addition to the performance metrics.

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

### Training and Testing Data Set

To estimate the performance of our reinforcement learning algorithms, we will create training and testing datasets. These datasets will be used to validate the performance of our models.

The Real-World Dataset ETF consists of data from January 2017 to November 2020. We begin by dividing the Real-World Dataset ETF closing price history into two datasets, the training set which includes ETF closing price history from February 2019 to February 2020, and a test dataset with ETF closing price history from March 2020 to November 2020. We selected the training and testing data sets due to the significant draw-downs observed in December 2018 to January 2019 and February 2020 to March 2020.

Figure 9 shows the Real-World Dataset ETF returns for the training dataset. Figure 10 shows the Real-World Dataset ETF returns for the testing dataset. Due to the significant draw-downs observed in February 2020 and the rapid market bounce starting March 2020, our testing dataset observed significant levels of returns and volatilities. Moving forward to our results, our performance metric benchmarks will be based on the training dataset, while the backtests performed will be based on the testing dataset.

![Figure 2](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_9.png)

![Figure 2](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_10.png)


[1] We will use the public library PyPortfolioOpt. More details can be found in here <https://pyportfolioopt.readthedocs.io/en/latest/>
