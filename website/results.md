---
title: Results
notebook:
nav_include: 5
---

## Overview
[Control Dataset](#control-dataset)

[Real Dataset - Two Asset Portfolio](#real-dataset---two-asset-portfolio)

[Real Dataset - Full Portfolio](#real-dataset---full-portfolio)

[Statistics](#statistics)


## Control Dataset

We tested all our models on a simulated 2-asset environment with a one-step reward defined as follows: on each observation the agent receives a reward ***R_t***. [1]

<img src="https://render.githubusercontent.com/render/math?math=R_t =  \Delta\Pi_t -\lambda a_t^T\Sigma a_t">

Where:

<img src="https://render.githubusercontent.com/render/math?math=\Delta\Pi_t="> Portfolio return between ***t*** and ***t+1*** given action at ***a_t*** time ***t***

<img src="https://render.githubusercontent.com/render/math?math=a_t="> action at time ***t*** which is the vector of weights

<img src="https://render.githubusercontent.com/render/math?math=\Sigma="> asset covariance matrix at time ***t***

<img src="https://render.githubusercontent.com/render/math?math=\lambda="> risk aversion parameter


The risk aversion parameter can be adjusted in order to account for the trade-off between risk and reward.  Setting ***λ = 0*** would represent in a minimum volatility reward function, while setting ***λ = 1*** would represent a maximum return reward function.
 
The results were consistent across different levels of ***μ*** and ***σ***. Below is an example of the model convergence using a predefined choice of parameters and for each of the previously discussed reward functions.


![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/model_convergence_reward_functions.png "model_convergence_reward_functions")

Below we present the average weights per epoch at different levels of risk aversion ***λ***. As expected, we can see that when  ***λ = 0*** (minimum volatility case) the weights converge to 100% investment in the asset with less volatility (Asset 0) and for the cases when and when ***λ = 1*** (maximum return case) the weights converge to 100% in the asset with higher return (Asset 1).


#### Models Convergence

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figures_11_12.png)

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figures_13_16.png)



## Real Dataset
Now that we have successfully demonstrated the operation of the Policy Gradient Methods with our control dataset, we then test the following Policy Gradient Methods on real-world data: REINFORCE, REINFORCE with Baseline, Actor-Critic, and Actor-Critic with Eligibility traces.

## Real Dataset - Two Asset Portfolio

We begin our evaluation of the real-world data by testing the PG Methods on a two asset portfolio containing MTUM (the higher return asset) and EFAV (the lower volatility asset) to check for model convergence and correct operation.  For each algorithm, we train our models for five different cases of ***λ***: 0, 0.2, 0.5, 0.8, and 1 to see how these algorithms select a portfolio which balances between different levels of risk and reward.  We discuss our findings for the ***λ = 0*** (minimum volatility) and ***λ = 1*** (maximum return) cases in this section.  Please refer to Appendix B, Section 8.2 for the results of the ***λ*** = 0.2, 0.5, and 0.8 cases.

#### REINFORCE 

##### ***λ = 0*** Case (Minimum Volatility)

Figure 17 shows how the REINFORCE algorithm selects a portfolio with low volatility (identified in blue) for a risk aversion parameter ***λ = 0***.  This is expected since ***λ = 0*** represents the minimum volatility portfolio. The stable evolution of backtests over the epoch training demonstrate that the model has converged on an optimal solution.  Comparing our training backtest to the benchmark portfolios, the backtest shows that the portfolio selected by REINFORCE has a volatility (3.53%) which is lower than the minimum volatility benchmark's volatility (3.95%).

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_17.png)

Figure 18 shows the contribution of the reward and volatility components of the reward function.  The components of the reward function are normalized so that they both have approximately the same magnitude at the start of the model training. Since ***λ = 0***, the reward function in our case is represented as <img src="https://render.githubusercontent.com/render/math?math=R_t=-a_t^T{\sigma}a_t"> (negative magnitude of the volatility component).

Figure 19 shows the asset weight distribution over the model training.  The model converges on the correct solution for a minimum volatility reward function with ***λ = 0***, consistently allocating 100% of the portfolio in EFAV (the lower volatility asset) after 2500 epochs of training.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_18_19.png)

Using the asset weights from the training period, we perform a backtest on the test dataset and compare the backtest return to the benchmarks as seen in Figure 20.  We see that the backtest has a volatility (4.1%) lower than the benchmark minimum volatility case (4.23%).  This demonstrates that our REINFORCE model is performing as expected to find the minimum volatility portfolio for a two asset portfolio.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_20.png)

##### ***λ = 1*** Case (Maximum Return)

In Figure 21, we demonstrate the operation of the REINFORCE algorithm using a risk aversion parameter of ***λ = 1***.  This represents the maximum return reward function, and the backtest of the training data shows that the return we obtained is the maximum return (24.03%), exceeding the returns of maximum return benchmark portfolio (18.21% return).  We also note that the model has converged during model training, as evidenced by the consistent backtest results across the last few sets of training epochs.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_21.png)

Figure 22 shows the contribution of the reward and volatility components of the reward function. Since ***λ = 1***, the reward function in our case is represented as <img src="https://render.githubusercontent.com/render/math?math=R_t=\Delta\Pi_t"> (the return component) without any contribution from the volatility component.

Figure 23 shows the asset weight distribution over the model training.  The model converges on the correct solution, consistently allocating 100% of the portfolio in MTUM (the higher return asset) after approximately 1300 epochs of training. 

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_22_23.png)

In Figure 24, we perform a backtest on the test dataset and note that it has a higher return (54.08%) then the three benchmarks.  This demonstrates that our REINFORCE model can return a maximum return portfolio with two assets on a test dataset it has not been trained on.  We also note that the volatility of our test backtest is 13.06%, so the maximum return does come at the expense of a higher volatility.  

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_24.png)


## Real Dataset - Full Portfolio
## Statistics

## Real Dataset


We begin by dividing the ETF price history into two datasets, the training set which includes ETF price history from January 2017 to March 2020, and a test dataset with ETF price history from April 2020 to November 2020. The series are de-meaned and we run two cases at different levels of risk aversion ***λ***, 0 and 10. In Figures 11-14, we see that with a risk aversion ***λ***=0, we did not reach convergence in any of the PG Methods. This is further evidenced by having a relatively equal asset weight distribution as seen in Figures 15-18. 

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/rewards_11_12.png)

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figures_13_18.png)

Using the policies obtained in the Policy Gradient Methods training, we perform a backtest on each of the PG Methods using the test set data and compare the backtest against the benchmark return.  The benchmark return is obtained using PyPortfolioOpt's mean-variance optimization with hierarchical risk parity weights.  Figure 19 shows that all four PG Methods have backtest returns which are similar to each other.  The backtest returns appear to be similar in behavior to the benchmark return, but at a lower magnitude of return.

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/risk_aversion_0.png)

In Figures 20-23, we run each of the Policy Gradient Methods using a risk aversion ***λ*** parameter of 10 for 10,000 epochs and note that the the PG Methods do not converge in these cases as well. Figures 24-27 show that in each of the PG Methods, the distribution appears to be fairly equal amongst all assets in the portfolio.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figures_20_25.png)

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figures_26_27.png)

In Figure 28, we compare the backtest returns for each of the PG Methods when ***λ***=10 to the benchmark return and observe that the backtest returns are similar to the results when ***λ***=0; they have a lower magnitude of return as compared to the benchmark return. 

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/risk_aversion_10.png)


## Statistics

Now that we have shown the results of our dataset, we will be evaluating the statistical components of our dataset. The statistical components will be based on the performance metrics described earlier. To generate our statistical components we will be using the public library of QuantStats.[2]


#### Real World Dataset Statistics
In the following figure, we evaluate each ETF's statistical components from January 2017 to November 2020. 

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/Real_Dataset_Statistics.png)

### Backtest Statistics

#### Backtest with Risk Aversion Factor 0
![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/BackTest_RiskAversion0_Statistics.png)

#### Backtest with Risk Aversion Factor 10
![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/BackTest_RiskAversion1000_Statistics.png)

### Rolling Sharpe Ratio

In the following figures, we can observe the rolling Sharpe Ratios of each respective model. The rolling Sharpe Ratio is useful to analyze the historical performance of a fund since it gives investors insights to the performance of the strategy.

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/rolling_sharpe_reinforce.png)
![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/rolling_sharpe_reinforce_baseline.png)
![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/rolling_sharpe_ac_no_trace.png)
![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/rolling_sharpe_ac_trace.png)










[1] Reward function could also be parametrized as <img src="https://render.githubusercontent.com/render/math?math=R_t = \lambda\Delta\Pi_t -(1-\lambda) a_t^T\Sigma a_t">. In this case caution needs to be taken in scaling the covariance matrix.

[2] QuantStats performs portfolio profiling, which allows portfolio managers to understand their performance better by providing them with in-depth analytics and risk metrics. <https://github.com/ranaroussi/quantstats>
