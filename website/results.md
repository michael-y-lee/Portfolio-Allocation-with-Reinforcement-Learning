---
title: Results
notebook:
nav_include: 5
---

## Overview
[Control Dataset](#control-dataset)

[Real Dataset - Two Asset Portfolio](#real-dataset---two-asset-portfolio)
- [REINFORCE](#two-asset-portfolio-reinforce)
- [REINFORCE with Baseline](#two-asset-portfolio-reinforce-with-baseline)
- [Actor Critic](#two-asset-portfolio-actor-critic)
- [Actor Critic with Eligibility Traces](#two-asset-portfolio-actor-critic-with-eligibility-traces)


[Real Dataset - Full Portfolio](#real-dataset---full-portfolio)
- [REINFORCE](#full-portfolio-reinforce)
- [REINFORCE with Baseline](#full-portfolio-reinforce-with-baseline)
- [Actor Critic](#full-portfolio-actor-critic)
- [Actor Critic with Eligibility Traces](#full-portfolio-actor-critic-with-eligibility-traces)


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

#### Two Asset Portfolio REINFORCE 

##### ***λ = 0*** Case (Minimum Volatility)

Figure 17 shows how the REINFORCE algorithm selects a portfolio with low volatility (identified in blue) for a risk aversion parameter ***λ = 0***.  This is expected since ***λ = 0*** represents the minimum volatility portfolio. The stable evolution of backtests over the epoch training demonstrate that the model has converged on an optimal solution.  Comparing our training backtest to the benchmark portfolios, the backtest shows that the portfolio selected by REINFORCE has a volatility (3.53%) which is lower than the minimum volatility benchmark's volatility (3.95%).

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_17.png)

Figure 18 shows the contribution of the reward and volatility components of the reward function.  The components of the reward function are normalized so that they both have approximately the same magnitude at the start of the model training. Since ***λ = 0***, the reward function in our case is represented as <img src="https://render.githubusercontent.com/render/math?math=R_t=-a_t^T\Sigma a_t"> (negative magnitude of the volatility component).

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

#### Two Asset Portfolio REINFORCE with Baseline 

##### ***λ = 0*** Case (Minimum Volatility)

In Figure 25, the REINFORCE with Baseline algorithm selects a low volatility portfolio (3.53%) for a risk aversion parameter ***λ = 0*** during model training.  Although there is some variation in the backtests at the start of the model training (as evidenced by the light blue backtest returns in the plot), the subsequent backtests with the dark blue backtests are fairly stable and consistent.  

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_25.png)

Figure 26 shows that the backtest of the test dataset has a volatility of 4.11%, which is below the benchmark minimum volatility portfolio's volatility of 4.23%.  Please refer to Appendix B, Section 8.2 for the reward function and the asset weight plots.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_26.png)

##### ***λ = 1*** Case (Maximum Return)

Figure 27 shows that with a maximum return reward function ***λ = 1***, the REINFORCE with Baseline algorithm selects the highest return portfolio (24.03%) for the training data set, with a higher return than the maximum return benchmark (18.21%).  This conclusion is also shown with the test dataset in Figure 28, where the test benchmark has a higher return (54.34%) than the maximum return benchmark (32.67%).  Please refer to Appendix B, Section 8.2 for the reward function and the asset weight plots. 

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_27.png)

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_28.png)

#### Two Asset Portfolio Actor Critic 

##### ***λ = 0*** Case (Minimum Volatility)

In Figure 29, the Actor-Critic algorithm converges on a portfolio with a volatility (3.53%) lower than the minimum volatility benchmark portfolio's volatility (3.95%). Figure 30 shows that the Actor-Critic algorithm also selects the minimum volatility portfolio for the test dataset, with a backtest volatility of 4.12% as compared to the minimum volatility benchmark's volatility of 4.23% in the test dataset. Please refer to Appendix B, Section 8.2 for the reward function and the asset weight plots.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_29_30.png)

##### ***λ = 1*** Case (Maximum Return)

Figure 31 shows that with a maximum return reward function ***λ = 1***, the Actor-Critic algorithm selects a maximum return portfolio (selected portfolio's return is 23.79%, while the maximum return benchmark's return is 18.21%). When we apply the weights learned from the training dataset to the test dataset in Figure 32 and perform a backtest, the backtest has a return of 52.87%, which is higher than the test dataset maximum return benchmark's return of 32.67%.  Please refer to Appendix B, Section 8.2 for the reward function and the asset weight plots. 

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_31_32.png)

#### Two Asset Portfolio Actor Critic with Eligibility Traces

##### ***λ = 0*** Case (Minimum Volatility)

In Figure 33, the Actor-Critic with Eligibility Traces algorithm converges on a minimum volatility portfolio for a risk aversion parameter ***λ = 0*** during model training.  The training backtest has volatility of 3.53% across the majority of the training epochs, and this is lower than the minimum volatility benchmark of 3.95%. 

Figure 34 shows that the Actor-Critic with Eligibility Traces algorithm selects the minimum volatility portfolio for the test dataset (volatility of 4.1% while the minimum volatility benchmark has a volatility of 4.23%)  Please refer to Appendix B, Section 8.2 for the reward function and the asset weight plots.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_33_34.png)

##### ***λ = 1*** Case (Maximum Return)

Figure 35 shows that with a max return reward function ***λ = 1***, the Actor-Critic with Eligibility Traces algorithm selects a portfolio with a return of 23.79%, which is higher than the maximum return benchmark's return (18.21%). Figure 36 shows that a backtest of the test dataset has a return of 53.81% while the maximum return benchmark has a return of 32.67%. Please refer to Appendix B, Section 8.2 for the reward function and the asset weight plots.   

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_35_36.png)


## Real Dataset - Full Portfolio

Now that we have tested our Policy Gradient Methods on a portfolio of two real assets, we now test our Policy Gradient Methods on our full portfolio of ETFs.  For each algorithm, we train our models for five different cases of ***λ***: 0, 0.2, 0.5, 0.8, and 1 to see how these algorithms select a portfolio which balances between different levels of risk and reward.  We discuss our findings for the ***λ = 0*** (minimum volatility), ***λ = 0.5***, and ***λ = 1*** (maximum return) cases in this section.  Please refer to Appendix B, Section 8.3 for the results of the ***λ*** = 0.2 and 0.8 cases.

#### Full Portfolio REINFORCE

##### ***λ = 0*** Case (Minimum Volatility)

Figure 37 shows how the REINFORCE algorithm selects a portfolio with low volatility (identified in blue) for a risk aversion parameter ***λ = 0***.  This is expected since ***λ = 0*** represents the minimum volatility portfolio. The stable evolution of backtests over the epoch training demonstrate that the model has converged on an optimal solution.  Comparing our training backtest to the benchmark portfolios, the backtest shows that the portfolio selected by REINFORCE has a volatility (3.52%) lower than the benchmark minimum volatility portfolio (4.62%).

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_37.png)

Figure 38 shows the contribution of the reward and volatility components of the reward function.  As previously discussed, the components of the reward function are normalized so that they both have approximately the same magnitude at the start of the model training.  Since ***λ = 0***, the reward function in this case is represented as the negative magnitude of the volatility component.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_38.png)

Figure 39 shows the evolution of the asset weight distribution over the model training.  Although the asset weights evolve during the model training the algorithm generally chooses to place 100% of the portfolio in EFAV, which is one of the lowest volatility assets in our portfolio. As shown in Figure 37, the backtest results are stable during the model training.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_39.png)

Using the asset weights from the training period, we perform a backtest on the test dataset and compare the backtest return to the benchmarks as seen in Figure 40.  We see that the backtest has a volatility (4.27%) lower than the benchmark minimum volatility case (4.39%).  This demonstrates that our REINFORCE model is performing as expected to find the minimum volatility portfolio for our full portfolio dataset.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_40.png)

##### ***λ = 0.5*** Case

Figure 41 shows how the REINFORCE algorithm selects a portfolio with low volatility (identified in blue) for a risk aversion parameter ***λ = 0.5***.  This parameter represents a tradeoff between the the minimum volatility portfolio (***λ = 0***) and maximum return portfolio (***λ = 1***), where both components of the reward function (reward and volatility) have an equal weight. The stable evolution of backtests over the epoch training demonstrate that the model has converged on an optimal solution.  Comparing our training backtest to the benchmark portfolios, the backtest shows that the portfolio selected by REINFORCE in the training data appears to have a volatility and return close to the benchmark maximum return portfolio (the backtest has a return of 22.33% and a volatility of 6.06%, while the benchmark has a return of 22.82% and a volatility of 6.35%).

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_41.png)

Figure 42 shows the contribution of the reward and volatility components of the reward function.  The components of the reward function are normalized so that they both have approximately the same magnitude at the start of the model training.  Since ***λ = 0.5***, the reward function in our case is represented as **<img src="https://render.githubusercontent.com/render/math?math=R_t= 0.5\Delta\Pi_t - 0.5a_t^T\Delta\a_t">** (the reward function depends on both the reward component and the volatility component equally).

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_42.png)

Figure 43 shows the asset weight distribution over the model training.  In general, the model allocates 100% of the portfolio in USMV, which is one of the higher return assets in our portfolio which is designed to have a minimum volatility.  Although the asset weights evolve during the model training, the training backtest in Figure 41 is fairly stable over the course of the model training.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_43.png)

Using the asset weights from the training period, we perform a backtest on the test dataset and compare the backtest return to the benchmarks as seen in Figure 44.  We see that the backtest has a volatility (5.55%) higher than the benchmark minimum volatility case (4.39%) but lower than the benchmark maximum return case (6.55%). The backtest also has a return of 30.47%, which is higher thn the benchmark maximum return case of 20.9%. This demonstrates that in the ***λ = 0.5*** case, the REINFORCE algorithms finds a portfolio which tries to maximize the return while controlling the volatility.  

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_44.png)

##### ***λ = 1*** Case (Maximum Return)

In Figure 45, we demonstrate the operation of the REINFORCE algorithm using a risk aversion parameter of ***λ = 1***.  This represents the maximum return reward function.  The backtest of the training data is fairly stable after the first 500 epochs, which indicates model convergence and the return is 22.36% (this is approximately the same as the maximum return benchmark, which has a return of 22.82%).

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_45.png)

Figure 46 shows the contribution of the reward and volatility components of the reward function. Since ***λ = 1***, the reward function in our case is represented by return component without any contribution from the volatility component.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_46.png)

Figure 47 shows the asset weight distribution over the model training.  The model allocates 100% of the assets to USMV, which is one of the higher return ETFs in our portfolio. 

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_47.png)

In Figure 48, we perform a backtest on the test dataset and note that it has a higher return (30.84%) then the maximum return benchmark portfolio (20.9% return).  This demonstrates that our REINFORCE model can return a maximum return portfolio given a portfolio of ETFs.  

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_48.png)

#### Full Portfolio REINFORCE with Baseline

##### ***λ = 0*** Case (Minimum Volatility)

In Figure 49, the REINFORCE with Baseline algorithm selects a low volatility portfolio (3.52%) for a risk aversion parameter ***λ = 0*** during model training (this is a lower volatility then the minimum volatility benchmark portfolio's volatility).  With the exception of one epoch training result, the other backtest returns are fairly stable.  Figure 50 shows that the REINFORCE with baseline algorithm selects the minimum volatility portfolio for the test dataset (the backtest has a volatility of 4.29%, which is below the benchmark minimum volatility portfolio's volatility of 4.39%).  Please refer to Appendix B, Section 8.3 for the reward function and the asset weight plots.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_49_50.png)

##### ***λ = 0.5*** Case

In Figure 51, the REINFORCE with Baseline algorithm tries to balance risk versus reward and selects a portfolio which appears to be similar to the maximum return or maximum Sharpe benchmarks.  The backtest of the training set has a return of 22.16% and a volatility of 6.06%.  In Figure 52, we apply the asset weights learned in the model training to the test dataset. The backtest has a return of 29.86% and a volatility of 5.52%; the return is higher than the benchmark returns but the volatility is within the range of the three benchmark volatilities.  Please refer to Appendix B, Section 8.3 for the reward function and the asset weight plots.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_51_52.png)

##### ***λ = 1*** Case (Maximum Return)

Figure 53 shows that with a maximum return reward function ***λ = 1***, the REINFORCE with Baseline algorithm selects a portfolio which maximizes return for the training data set.  The training backtest's return (22.0%) is slightly lower than the maximum return benchmark's return (22.82%) and similar to the training data backtest return in the ***λ = 0.5*** case.  Figure 54 shows that the test benchmark has a higher return (30.03%) than the maximum return benchmark's return (20.9%).  Please refer to Appendix B, Section 8.3 for the reward function and the asset weight plots. 

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_53_54.png)

#### Full Portfolio Actor Critic

##### ***λ = 0*** Case (Minimum Volatility)

In Figure 55, the Actor-Critic algorithm converges on a portfolio with a volatility of 3.52%. This is a lower volatility than the minimum volatility benchmark portfolio's volatility (4.62%). Figure 56 shows that the Actor-Critic algorithm also selects the minimum volatility portfolio for the test dataset, with a backtest volatility of 4.28% as compared to the minimum volatility benchmark of 4.39% in the test dataset. Please refer to Appendix B, Section 8.3 for the reward function and the asset weight plots.  

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_55_56.png)

##### ***λ = 0.5*** Case

In Figure 57, the Actor-Critic algorithm seeks to balance reward versus risk by selecting a portfolio which has a return of 21.96% and a volatility of 6.05%.  This is close to the maximum return and maximum Sharpe benchmarks.  Although there is some variation in the backtests over the model run, the backtests across different epochs are similar to each other in terms of return and volatility.  In Figure 58, the backtest on the test dataset has a return of 22.16% and a volatility of 4.28%.  This is a higher return than the maximum return benchmark and the volatility is lower than the minimum volatility benchmark. Please refer to Appendix B, Section 8.3 for the reward function and the asset weight plots. 

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_57_58.png)


##### ***λ = 1*** Case (Maximum Return)

Figure 59 shows that the Actor-Critic algorithm selects a portfolio with a return of 21.83%. This is slightly lower than the returns of the maximum return and the maximum Sharpe benchmarks.  The backtests are fairly stable over the course of the model training.  When we perform a backtest on the test dataset in Figure 60, the portfolio has a return of 29.23%, which is higher than the maximum return benchmark of 20.9%.  Please refer to Appendix B, Section 8.3 for the reward function and the asset weight plots. 

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_59_60.png)

#### Full Portfolio Actor Critic with Eligibility Traces

##### ***λ = 0*** Case (Minimum Volatility)

In Figure 61, the Actor-Critic with Eligibility Traces algorithm selects a portfolio which has a training backtest volatility of 3.52% across the majority of the training epochs; this is lower than the minimum volatility benchmark of 4.62%.  Figure 62 shows that the Actor-Critic with Eligibility Traces algorithm selects the minimum volatility portfolio for the test dataset (volatility of 4.4% is similar to the volatility of the the minimum volatility benchmark, 4.39%)  Please refer to Appendix B, Section 8.3 for the reward function and the asset weight plots.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_61_62.png)

##### ***λ = 0.5*** Case

In Figure 63, the Actor-Critic with Eligibility Traces algorithm tries to balance risk versus reward and selects a portfolio which appears to be similar to the maximum return or maximum Sharpe benchmarks.  The backtest of the training set has a return of 22.26% and a volatility of 6.06%.  In Figure 64, we perform a backtest on the test dataset and the return is 30.56% and the volatility is 5.57%; the return is higher than the benchmark returns and the volatility is within the range of the three benchmark volatilities.  Please refer to Appendix B, Section 8.3 for the reward function and the asset weight plots.

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_63_64.png)

##### ***λ = 1*** Case (Maximum Return)

Figure 65 shows that with a max return reward function ***λ = 1***, the Actor-Critic with Eligibility Traces algorithm selects a portfolio with a return of 21.83% and a volatility of 6.05%.  This return is slightly lower than but close to the maximum return benchmark. The backtest evolution over the course of the model training is stable.  In Figure 66, the backtest on the test dataset has a return (30.15\%) higher than the maximum return benchmark (20.9%). Please refer to Appendix B, Section 8.3 for the reward function and the asset weight plots.   

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/figure_65_66.png)

## Statistics




[1] Reward function could also be parametrized as <img src="https://render.githubusercontent.com/render/math?math=R_t = \lambda\Delta\Pi_t -(1-\lambda) a_t^T\Sigma a_t">. In this case caution needs to be taken in scaling the covariance matrix.

[2] QuantStats performs portfolio profiling, which allows portfolio managers to understand their performance better by providing them with in-depth analytics and risk metrics. <https://github.com/ranaroussi/quantstats>
