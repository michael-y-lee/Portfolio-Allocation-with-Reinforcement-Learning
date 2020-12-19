---
title: Tested Models
notebook:
nav_include: 3
---

## Overview

As mentioned in the introduction, the objective of the capstone is to build a full machine learning model portfolio of quantitative strategies. To achieve this task we will use Reinforcement Learning, and within Reinforcement Learning we will use Policy Gradient Methods. Policy Gradient Methods focus on estimating directly the Policy **π**(**a**\|**s**) rather than the action value function, usually referred in Reinforcement Learning as **Q**(**s**, **a**). An advantage of Policy Gradient methods is that it only requires that the policy be parametric in any way as long as **π**(**a**\|**s**, **θ**) is differential with respect to its parameters. This is an interesting feature as it is not making any restriction on the action or state space. In our case, this is a necessary condition given that we want a full machine learning approach and our action space consists of all the possible combinations of weights **w** in a portfolio. This is a continuous infinite space which can’t be directly estimated using Action-Value methods.

## Policy Gradient Methods

The Policy Gradient Methods that we will explore in this capstone work by defining a performance function **J**(**θ**) and then maximizing it using gradient ascent. By the Policy Gradient theorem, the gradient of **J**(**θ**) is proportional to:

<img src="https://render.githubusercontent.com/render/math?math=\nabla J(\theta)\propto \sum_s\mu(s)\sum_aq_\pi(s,a)\nabla \pi(a|s,\theta)">

Several algorithms can be used to solve this problem. For this capstone project, we will explore three different Policy Gradient methods: REINFORCE, REINFORCE with Baseline, and Actor-Critic Methods.

### REINFORCE

Our first model will be REINFORCE as a baseline and the simplest PG model. The Policy Gradient theorem gives an exact expression proportional to the gradient; all that is needed is some way of sampling whose expectation equals or approximates this expression. Notice that the right-hand side of the Policy Gradient theorem is a sum over states weighted by how often the states occur under the target policy **π**; if **π** is followed, then states will be encountered in these proportions. Thus:

<img src="https://render.githubusercontent.com/render/math?math=\nabla J(\theta)\propto \sum_s\mu(s)\sum_aq_\pi(s,a)\nabla\pi(a|s,\theta)">
<img src="https://render.githubusercontent.com/render/math?math==E_\pi[\sum_a q_\pi(S_t,a)\nabla\pi(a|S_t,\theta)]">

We could stop here and instantiate our stochastic gradient-ascent algorithm where q̂  is some learned approximation to *q*<sub>*π*</sub>. This algorithm has been called an all-actions method because all actions are involved in the update. In the case of REINFORCE, the update at time ***t*** involves just ***A*<sub>*t*</sub>**, the one action was actually taken at time ***t***.

<img src="https://render.githubusercontent.com/render/math?math=\nabla J(\theta)=E_\pi[\sum_a \pi(a|S_t,\theta)q_\pi(S_t,a)\frac{\nabla\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}]">

<img src="https://render.githubusercontent.com/render/math?math= = E_\pi[q_\pi(S_t,A_t)\frac{\nabla\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}]">

<img src="https://render.githubusercontent.com/render/math?math= = E_\pi[G_t\frac{\nabla\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}]]">

Where we replaced *a* by a sample *A*<sub>*t*</sub>.

The final expression in brackets is exactly what is needed, a quantity that can be sampled on each time step whose expectation is equal to the gradient. Using this sample to instantiate our generic stochastic gradient ascent algorithm yields the REINFORCE update:

<img src="https://render.githubusercontent.com/render/math?math= \theta_{t+1}=\theta_t+\alpha G_t\frac{\nabla\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}">



### REINFORCE with Baseline
### Actor-Critic Methods
### One Step Actor-Critic
### Actor-Critic with Eligibility Traces
### Policy Parameterization
