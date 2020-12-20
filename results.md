---
title: Results
notebook:
nav_include: 5
---

## Control Dataset

We tested all our models on a simulation of a 2-asset environment. The results were consistent across different levels of **μ** and **σ**. Below is an example of the model convergence using a predefined choice of parameters and for each of the previously discussed reward functions 

We tested all our models on a simulated 2-asset environment with a one-step reward defined as follows: on each observation the agent receives a reward ***R_t***. [5]

<img src="https://render.githubusercontent.com/render/math?math=R_t =  \Delta\Pi_t -\lambda a_t^T\Sigma a_t">

Where:

<img src="https://render.githubusercontent.com/render/math?math=\Delta\Pi_t="> Portfolio return between ***t*** and ***t+1*** given action at ***a_t*** time ***t***

<img src="https://render.githubusercontent.com/render/math?math=a_t="> action at time ***t*** which is the vector of weights

<img src="https://render.githubusercontent.com/render/math?math=\Sigma="> asset covariance matrix at time ***t***

<img src="https://render.githubusercontent.com/render/math?math=\lambda="> risk aversion parameter

The results were consistent across different levels of ***μ*** and ***σ***. Below is an example of the model convergence using a predefined choice of parameters and for each of the previously discussed reward functions.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/model_convergence_reward_functions.png "model_convergence_reward_functions")


Below we present the average weights per epoch at different levels of risk aversion ***λ***. As expected, we can see that when ***lim<sub>*λ* → ∞</sub>*** the weights converge to 100% investment in the asset with less volatility (Asset 0) and for the cases when and when ***λ*** = 0 the weights converge to 100% in the asset with higher return (Asset 1).


#### Models Convergence

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/model_convergence_1.png)

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/model_convergence_2.png)


## Real Dataset

Now that we have successfully demonstrated the operation of the PQM algorithms with our control dataset, we then test four of the PQM algorithms on real-world data: REINFORCE, REINFORCE with Baseline, Actor-Critic, and Actor-Critic with Eligibility traces. 

We begin by dividing the ETF price history into two datasets, the training set which includes ETF price history from January 2017 to March 2020, and a test dataset with ETF price history from April 2020 to November 2020. The series are de-meaned and we run two cases at different levels of risk aversion ***λ***, 0 and 10. In Figures 13-16, we see that with a risk aversion ***λ***=0, we did not reach convergence in any of the PG Methods. This is further evidenced by having a relatively equal asset weight distribution as seen in Figures 17-20. 

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/risk0_rewards.png)

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/risk0_weights.png)

Using the policies obtained in the Policy Gradient Methods training, we perform a backtest on each of the PG Methods using the test set data and compare the backtest against the benchmark return.  The benchmark return is obtained using PyPortfolioOpt's mean-variance optimization with hierarchical risk parity weights.  Figure 21 shows that all four PG Methods have backtest returns which are similar to each other.  The backtest returns appear to be similar in behavior to the benchmark return, but at a lower magnitude of return.

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/plot_backtest_0.png)

In Figures 22-25, we run each of the Policy Gradient Methods using a risk aversion ***λ*** parameter of 10 for 10,000 epochs and note that the the PGM models do not converge in these cases as well. Figures 26-29 show that in each of the PGM models, the distribution appears to be fairly equal amongst all assets in the portfolio. 

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/risk10_rewards.png)

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/risk10_weights1.png)
![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/risk10_weights2.png)

In Figure 30, we compare the backtest returns for each of the PG Methods when ***λ***=10 to the benchmark return and observe that the backtest returns are similar to the results when ***λ***=0; they have a lower magnitude of return as compared to the benchmark return. 

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/plot_backtest_10.png)


## Statistics

Now that we have shown the results of our dataset, we will be evaluating the statistical components of our dataset. The statistical components will be based on the performance metrics described earlier. To generate our statistical components we will be using the public library of QuantStats.[6]


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










[5] Reward function could also be parametrized as <img src="https://render.githubusercontent.com/render/math?math=R_t = \lambda\Delta\Pi_t -(1-\lambda) a_t^T\Sigma a_t">. In this case caution needs to be taken in scaling the covariance matrix.
[6] QuantStats performs portfolio profiling, which allows portfolio managers to understand their performance better by providing them with in-depth analytics and risk metrics. \url{https://github.com/ranaroussi/quantstats}
