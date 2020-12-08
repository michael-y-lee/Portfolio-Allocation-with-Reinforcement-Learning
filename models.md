---
title: Tested Models
notebook:
nav_include: 3
---

## Overview of Reinforcement Learning

As mentioned in the introduction, the objective of the capstone is to build a full machine learning model portfolio of quantitative strategies. To achieve this task we will use Reinforcement Learning, and within Reinforcement Learning we will use Policy Gradient methods. Policy Gradient methods focus on estimating directly the Policy **π**(**a**\|**s**) rather than the action-value function, usually referred to in Reinforcement Learning as **Q**(**s**, **a**). An advantage of Policy Gradient methods is that it only requires that the policy be parametric in any way as long as **π**(**a**\|**s**, **θ**) is differential with respect to its parameters. This is an interesting feature as it is not making any restriction on the action or state space. In our case, this is a necessary condition given that we want a full machine learning approach and our action space consists of all the possible combinations of weights **w** in a portfolio. This is a continuous infinite space that can’t be directly estimated using action value methods. 

## Policy Gradient Methods

### REINFORCE
### REINFORCE with Baseline
### Actor-Critic Methods
### One Step Actor-Critic
### Actor-Critic with Eligibility Traces
### Soft Actor-Critic
### Policy Parameterization
