---
title: Introduction
notebook:
nav_include: 1
---

Trading strategies in quantitative finance are based on financial fundamentals using a mathematical model . **F** is a function that maps information, **I**, into **R**. **R** is usually is a price forecast which can also be a trading signal **(i.e., a buy or a sell indicator)**. Once the mathematical model produces **R**, the quantitative strategy needs to transform **R** into an action. From a portfolio management perspective, this is known as an allocation of capital. Taking recommendations and price forecasts and converting them into actions requires an additional layer of logic. If this layer of logic is executed manually, then the machine learning model is not considered to be automated and therefore is not extensible or adaptable. If we were to automate this layer by machine learning, then we would be able to factor in transaction costs, overlapping transactions against alternative strategies, controlling for draw-downs, sharing signal information between strategies, etc. Therefore, the allocation between quantitative strategies is a problem of interest to financial investors who leverages quantitative strategies for portfolio management. In this capstone project, we plan to address this problem by building a machine learning algorithm for portfolio allocation using reinforcement learning.  

Reinforcement learning (RL) is an area of machine learning, in which learning of an agent occurs as a result of its own actions and interaction with the environment; the agent has to both exploit is current knowledge to get a better reward. Every action that the agent takes on every state can be described as **policy** *π*(*a*\|*s*) = *P*(*A*<sub>*t*</sub> = *a*\|*s*). Most of the Reinforcement Learning algorithms can be classified in two categories:  

* Methods that try to find the optimal policy *π*(*a*\|*s*) by estimating the value of each action at each state i.e. *Q*(*a*, *s*).

* Methods that try to find the optimal policy *π*(*a*\|*s*) by directly estimating it without necessary estimating the state-value action *Q*(*a*\|*s*), this methods are known as Policy Gradient (PG) methods. One of the main advantages when using this type of methods is that they can work with environments with a lot of actions, or in extreme, with a continuous action space.

Using reinforcement learning terminology, the portfolio manager is considered the agent. The actions are all the possible allocation between strategies, the environment is the stock market and the cumulative reward can be customized to the investor’s goals. For this reason we will focus in this project in Policy Gradient methods. Development of reinforcement learning models would allow quantitative investors to improve their portfolio allocation between different quantitative strategies. Applying artificial intelligence to portfolio management is gaining importance as seen in papers by Moody and Saffell (2001), Dempster and Leemans (2006), Cumming (2015), Deng (2017), Jiang, Xu and Liang (2017), and Honchar (2020) .The motivation and primary goal of our project is to use Reinforcement Learning Policy Gradient methods to build a portfolio of quantitative strategies. 

![Figure 1](https://raw.githubusercontent.com/nikatpatel/epsilon-greedy-quants/main/_assets/reinforcement_learning.png "Figure 1 - Portfolio Manager Perspective of Reinforcement Learning")

For this project we will narrow portfolio construction of quantitative strategies to **Smart Beta** strategies, one of the most popular quantitative strategies available to financial investors[1]. Smart Beta emphasizes capturing investment factors or market inefficiencies in a rules-based and transparent way. However, there are two main problems in building a portfolio based on Smart Beta strategies. The first problem is that Smart Beta returns are hard to forecast and the second problem is that underlying assets of the strategies can change over time. Since Smart Beta captures a statistical risk premium over time, there is no need to time the strategies and allocations that only focus on risk factors. Therefore, the theoretical solution to this problem is to build an equal risk contribution portfolio. 

The reality is that Smart Beta strategies represent a correlation with other risk factors, which in many cases are not taken into account when the financial strategy is constructed. Therefore, having a "full" machine learning algorithm for portfolio construction of Smart Beta strategies will bring an immediate benefit to the investor community by providing a better allocation between Smart Beta strategies under different market scenarios. For the purpose of this project, a "full" machine learning algorithm combines the trading signals with the portfolio allocation. 

[1] As of 2018, 91 percent of asset owners have a Smart Beta strategy in their portfolios. Source: https://www.investopedia.com/news/survey-confirms-smart-beta-growth-trajectory/ 


