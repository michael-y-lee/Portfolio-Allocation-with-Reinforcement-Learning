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

REINFORCE ***λ=0***

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/reinforce_l_0.png "REINFORCE \lambda=0")

REINFORCE ***λ=1e5***

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/reinforce_l_inf.png "REINFORCE \lambda=1e5")

REINFORCE with Baseline ***λ=0***

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/reinforce_baseline_max_return_sim.png "REINFORCE with Baseline \lambda=0")

REINFORCE with Baseline ***λ=1e5***

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/reinforce_baseline_min_var_sim.png "REINFORCE with Baseline \lambda=1e5")

Actor Critic ***λ=0***

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/ac_l_0.png "Actor Critic \lambda=0")

Actor Critic ***λ=1e5***

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/ac_l_inf.png "Actor Critic \lambda=1e5")

Soft Actor Critic ***λ=0***

![Figure 5](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/sac_max_ret.png "Soft Actor Critic \lambda=0")

Soft Actor Critic ***λ=1e5***

![Figure 6](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/sac_min_var.png "Soft Actor Critic \lambda=1e5")



## Real Dataset








[5] Reward function could also be parametrized as <img src="https://render.githubusercontent.com/render/math?math=R_t = \lambda\Delta\Pi_t -(1-\lambda) a_t^T\Sigma a_t">. In this case caution needs to be taken in scaling the covariance matrix.
