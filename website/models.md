---
title: Tested Models
notebook:
nav_include: 3
---

## Overview

As mentioned in the introduction, the objective of the capstone is to build a full machine learning model portfolio of quantitative strategies. To achieve this task we will use Reinforcement Learning, and within Reinforcement Learning we will use Policy Gradient Methods. Policy Gradient Methods focus on estimating directly the Policy **π**(**a**\|**s**) rather than the action value function, usually referred in Reinforcement Learning as **Q**(**s**, **a**). An advantage of Policy Gradient Methods is that it only requires that the policy be parametric in any way as long as **π**(**a**\|**s**, **θ**) is differential with respect to its parameters. This is an interesting feature as it is not making any restriction on the action or state space. In our case, this is a necessary condition given that we want a full machine learning approach and our action space consists of all the possible combinations of weights **w** in a portfolio. This is a continuous infinite space which can’t be directly estimated using Action-Value Methods.

## Policy Gradient Methods

The Policy Gradient Methods that we will explore in this capstone work by defining a performance function **J**(**θ**) and then maximizing it using gradient ascent. By the Policy Gradient theorem, the gradient of **J**(**θ**) is proportional to:

<img src="https://render.githubusercontent.com/render/math?math=\nabla J(\theta)\propto \sum_s\mu(s)\sum_aq_\pi(s,a)\nabla \pi(a|s,\theta)">

Several algorithms can be used to solve this problem. For this capstone project, we will explore three different Policy Gradient Methods: REINFORCE, REINFORCE with Baseline, and Actor-Critic Methods.

### REINFORCE

Our first model will be REINFORCE as a baseline and the simplest PG model. The Policy Gradient theorem gives an exact expression proportional to the gradient; all that is needed is some way of sampling whose expectation equals or approximates this expression. Notice that the right-hand side of the Policy Gradient theorem is a sum over states weighted by how often the states occur under the target policy **π**; if **π** is followed, then states will be encountered in these proportions. Thus:

<img src="https://render.githubusercontent.com/render/math?math=\nabla J(\theta)\propto \sum_s\mu(s)\sum_aq_\pi(s,a)\nabla\pi(a|s,\theta)">
<img src="https://render.githubusercontent.com/render/math?math==E_\pi[\sum_a q_\pi(S_t,a)\nabla\pi(a|S_t,\theta)]">

We could stop here and instantiate our stochastic gradient-ascent algorithm where q̂  is some learned approximation to *q*<sub>*π*</sub>. This algorithm has been called an all-actions method because all actions are involved in the update. In the case of REINFORCE, the update at time ***t*** involves just ***A*<sub>*t*</sub>**, the one action was actually taken at time ***t***.

<img src="https://render.githubusercontent.com/render/math?math=\nabla J(\theta)=E_\pi[\sum_a \pi(a|S_t,\theta)q_\pi(S_t,a)\frac{\nabla\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}]">

<img src="https://render.githubusercontent.com/render/math?math==E_\pi[q_\pi(S_t,A_t)\frac{\nabla\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}]">

<img src="https://render.githubusercontent.com/render/math?math==E_\pi[G_t\frac{\nabla\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}]">


Where we replaced **a** by a sample ***A*<sub>*t*</sub>**.

The final expression in brackets is exactly what is needed, a quantity that can be sampled on each time step whose expectation is equal to the gradient. Using this sample to instantiate our generic stochastic gradient ascent algorithm yields the REINFORCE update:

<img src="https://render.githubusercontent.com/render/math?math=\theta_{t%2B1}=\theta_t%2B\alpha G_t\frac{\nabla\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}">


### REINFORCE with Baseline

As a stochastic gradient method, REINFORCE has good theoretical convergence properties. By construction, the expected update over an episode is in the same direction as the performance gradient. This assures an improvement in expected performance for sufficiently small **α**, and convergence to a local optimum under standard stochastic approximation conditions for decreasing **α**. However, as a Monte Carlo method, REINFORCE may be of high variance and thus produce slow learning. To improve REINFORCE, the addition of a baseline function can help speed the convergence of the algorithm. The Policy Gradient theorem (Sutton and Barto, Equation 13.5) can be generalized to include a comparison of the action value to an arbitrary baseline ***b*(*s*)**:

<img src="https://render.githubusercontent.com/render/math?math=\nabla J(\theta)\propto \sum_s\mu(s)\sum_a(q_\pi(s,a)-b(s))\nabla \pi(a|s,\theta)">

The baseline can be any function, including a random variable as long as it does not vary with ***a***. One natural choice for the baseline is an estimate of the state value ***v̂*(*S*<sub>*t*</sub>, *w*)**; with this baseline the update rule in the gradient ascent will be:

<img src="https://render.githubusercontent.com/render/math?math=\theta_t%2B1=\theta%2B\alpha(G_t-b(S_t))\frac{\nabla\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}">

### Actor-Critic Methods

In REINFORCE with baseline, the learned state-value function estimates the value of only the first state of each state transition. This estimate sets a baseline for the subsequent return, but is made prior to the transition’s action and thus cannot be used to assess that action. In actor-critic methods, on the other hand, the state-value function is applied also to the second state of the transition. The estimated value of the second state, when discounted and added to the reward, constitutes the one-step return, ***G*<sub>*t* : *t* + 1</sub>** which is a useful estimate of the actual return and thus is a way of assessing the action.

When the state-value function is used to assess actions in this way it is called a critic, and the overall policy-gradient method is termed an actor-critic method. Note that the bias in the gradient estimate is not due to bootstrapping as such; the actor would be biased even if the critic was learned by a Monte Carlo method.

![Figure 2](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/actor_critic.png "From Deep Reinforcement Learning Hands-On (Lapan, 2018)")

The Actor-Critic Methods which we tested are the following:

### One Step Actor-Critic

One-step actor–critic methods replace the full return with the one-step return (and use a learned state-value function as the baseline) as follows:

<img src="https://render.githubusercontent.com/render/math?math=\theta_{t%2B1}=\theta_t %2B \alpha(G_{t:t%2B1}-\hat{\nu}(S_t,{w})\frac{\nabla\pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)}">

<img src="https://render.githubusercontent.com/render/math?math=\theta_{t%2B1}=\theta_t %2B \alpha(R_{t+1}\gamma\hat{\nu}(S_{t%2B1},{w})-\hat{\nu}(S_t,{w})\frac{\nabla\pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)}">

As Sutton and Barto (2018) notes, one of the advantages of one-step methods is that they are fully online and incremental, but they avoid the complexities of eligibility traces since they are a special case of the eligibility trace methods that is easier to understand.  Pseudocode for one-step actor-critic is listed below [1]:

![Figure 3](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/ac_1_step.png "One-Step Actor-Critic Pseudocode (Sutton and Barto, 2018)")

### Actor-Critic with Eligibility Traces

The forward view of n-step methods can be generalized by replacing the one-step return with ***G*<sub>*t* : *t* + *n*</sub>** and the forward view of a ***λ***-return algorithm is replaced by ***G*<sub>*t*</sub><sup>*λ*</sup>**. The backward view of the ***λ***-return algorithm uses separate eligibility traces for the actor and critic. Pseudocode for actor-critic with eligibility traces is listed below [1]:

![Figure 4](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/ac_l_return.png "Actor-Critic with Eligibility Traces pseudocode (Sutton and Barto, 2018)")

### Policy Parameterization

So far we have discussed PG Methods that depend on a parameterization of the action space. For our particular case, we need a continuous parameterization. Our baseline parameterization will be a multivariate Gaussian policy defined as:

<img src="https://render.githubusercontent.com/render/math?math=\pi(a|s,\theta)=\frac{exp[-\frac{1}{2}(a-\mu(s,\theta))^T\Sigma^{-1}_{(s,\theta)}(a-\mu(s,\theta))]}{\sqrt{2\pi^k|\Sigma_{(s,\theta)}|}}">

Where ***k*** is the number of assets in the portfolio.

For the parameterization of the mean and variance, we will try several approaches. First, we will use a linear parameterization where the feature space ***x(s)*** will be the same realized volatilites that we are using to build an Equal Risk Contribution (ERC) portfolio. The reason for choosing this initial parameterization is to have a sensitivity of how the algorithm performs when it gets the same information as the benchmark. 

<img src="https://render.githubusercontent.com/render/math?math=\mu(s,\theta)=\theta_\mu^Tx(s)">

<img src="https://render.githubusercontent.com/render/math?math=\sigma_{i,j}(s,\theta)=\theta_\sigma_{i,j}^Tx(s)">

[1] Sutton, Richard S., and Andrew G. Barto. Reinforcement Learning: an Introduction. The MIT Press, 2018. <http://incompleteideas.net/book/the-book.html> 


