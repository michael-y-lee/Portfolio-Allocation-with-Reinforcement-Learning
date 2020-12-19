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

## Real Dataset

We divided the real data set in two sub-sets[2]. One set will be use to select features and allocate weights in the portfolio and the other one will be used only to choose features. The portfolio set contains the following ETFs[3]:

- iShares MSCI USA Value Factor ETF (**VLUE**)
- iShares MSCI USA Quality Factor (**QUAL**)
- iShares MSCI USA Momentum Factor (**MTUM**)
- iShares US Size Factor (**SIZE**)
- iShares SP500 Minimum Volatility ETF (**USMV**)
- iShares MSCI EM Minimum Volatility (**EEMV**)
- iShares MSCI EAFE Minimum Volatility (**EFAV**)

We choose this set of ETFs to have a well known, investable set of simple systematic strategies. This set contains all the factor strategies (value, size, momentum, volatility) from iShares[3] with more than 5 years of daily data. The second subset contains the assets that we will use only to extract features:

- iShares 7-10 years US Treasuries (**IEF**)
- iShares Gold Trust (**IAU**)
- SPDR SP 500 ETF Trust (**SPY**)
- DB US Dollar Index (**UUP**)
- Investment Grade Corp Bond iShares Iboxx (**LQD**)

[2] Data provided by barchart.com

[3] https://www.ishares.com/us/strategies/smart-beta-investing
