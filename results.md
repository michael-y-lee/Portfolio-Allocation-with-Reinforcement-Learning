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






[5] Reward function could also be parametrized as <img src="https://render.githubusercontent.com/render/math?math=R_t = \lambda\Delta\Pi_t -(1-\lambda) a_t^T\Sigma a_t">. In this case caution needs to be taken in scaling the covariance matrix.
