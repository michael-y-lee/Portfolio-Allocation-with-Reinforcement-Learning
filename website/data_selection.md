---
title: Data Selection
notebook:
nav_include: 2
---

## Overview
For the project, we will utilize two different sets of data: A control dataset and a real-world dataset. The usage of the control dataset is to measure the efficiency of the frameworks in finding optimal actions to a known solution. The real-world dataset will be used to train the reinforcement learning models.

## Control Dataset
For the control dataset, we simulated different assets using a classical geometric Brownian motion process for each of the assets i.e.

<img src="https://render.githubusercontent.com/render/math?math=dS_t=\mu S_tdt %2B \sqrt{\sigma}S_tdB_t">

where **B**<sub>*t*</sub> is a Brownian motion.

### Control Data Set Generation

To Generate different simulations of assets we build a class method to each of the environments called

> build_environment_from_simulated_assets

The description of the method can be found in the docstrings.

## Real-World Data

We divided the real data set in two sub-sets[1]. One set will be use to select features and allocate weights in the portfolio and the other one will be used only to choose features. The portfolio set contains the following ETFs[2]:

- iShares MSCI USA Value Factor ETF (**VLUE**)
- iShares MSCI USA Quality Factor (**QUAL**)
- iShares MSCI USA Momentum Factor (**MTUM**)
- iShares US Size Factor (**SIZE**)
- iShares SP500 Minimum Volatility ETF (**USMV**)
- iShares MSCI EM Minimum Volatility (**EEMV**)
- iShares MSCI EAFE Minimum Volatility (**EFAV**)

We choose this set of ETFs to have a well known, investable set of simple systematic strategies. This set contains all the factor strategies (value, size, momentum, volatility) from iShares[2] with more than 5 years of daily data. The second subset contains the assets that we will use only to extract features:

- iShares 7-10 years US Treasuries (**IEF**)
- iShares Gold Trust (**IAU**)
- SPDR SP 500 ETF Trust (**SPY**)
- DB US Dollar Index (**UUP**)
- Investment Grade Corp Bond iShares Iboxx (**LQD**)

### Real-World Data Input to Model

In terms of data requirements for our model, the assets used for our model should consist of daily time bars of closing price and all assets should have the same start and end time range. This data should saved as a separate parquet file for each asset and stored in the data_env directory in our Github repository. When preparing our data, the time series are de-meaned. We built a class method called

> build_environment_from_dirs_and_transform

to load the asset price data from the data_env directory into our environment and persist data transformations for further training.  Please refer to our User Guide for more details of the data requirements and an example of how the real-world data is loaded into our models.

### Processing Data As Features

Once the real-world data is loaded into our models, we process the data before added them as features to our model.  Since time series data are nonstationary, we process the data so that they become stationary.  First, we detrend each time series using a Double Exponential Smoothing model (this is equivalent to a Holt-Winters smoothing model without the seasonal component) as seen in Figure 2 (Ng, 2019).  The trend and level coefficients of the double exponential smoothing model are scaled and added as features.  From the detrended time series, we then compute the residuals and demean the returns add the scaled data as features to our model as well. The QQ Plot in Figure 3 shows that although the residuals of a detrended MTUM ETF time series has a left skew, the majority of the residuals in the [-1, 1] theoretical quantile range are normally distributed. Likewise, the QQ Plots of the other ETFs in the portfolio (please refer to Appendix B, Section 8.1) also show normally distributed residuals in the [-1, 1] theoretical quantile range.  This gives us confidence that our model can properly use the real-world time series data.  

![Figure 2](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figures_2_3.png)


We also plotted an ACF (autocorrelation function) and PACF (partial autocorrelation function) plot of the residuals of the detrended MTUM time series, as seen in Figures 4 and 5 respectively.  The ACF plot shows the correlation of a time series with all of its lagged values while the PACF plot shows the correlation of the residuals with its next lag. If there are any high correlations in the PACF plot, it suggests that there could be hidden information in the residual that can be modeled by the following lag and we should keep that lag as a feature.[3].  Based on the PACF plot in Figure 5 as well as the PACF plots of the other ETFs in Appendix B, Section 8.1, there is only a significant spike at the lag-0 and lag-1 autocorrelation positions (some ETFs may show a statically significant autocorrelation at other lags but only lag-0 and lag-1 are statistically significant across all 7 ETFs).  As a result, we only include one lag to the features in our model.  

![Figure 4](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figures_4_5.png)


[1] Data provided by barchart.com <https://barchart.com>

[2] <https://www.ishares.com/us/strategies/smart-beta-investing>

[3] Please refer to <https://towardsdatascience.com/significance-of-acf-and-pacf-plots-in-time-series-analysis-2fa11a5d10a8> for more discussion about the significance of ACF and PACF plots for time series data
