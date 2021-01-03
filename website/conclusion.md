---
title: Conclusion
notebook:
nav_include: 6
---

Applying Reinforcement Learning to portfolio management was the objective of this capstone project.  We determined that this could be accomplished by using Policy Gradient Methods such as REINFORCE, REINFORCE with baseline, Actor-Critic, and Actor-Critic with Eligibility Traces to build a model-free agent which selects portfolio weights.

We used a reward function which can be balanced between risk and reward by using a risk aversion parameter, ***λ***. Using a control dataset of simulated data, we were able to demonstrate model convergence for the four Policy Gradient Methods that we tried.  After detrending our real-world data and demeaning the returns, we were also able to demonstrate convergence for all four Policy Gradient methods in a portfolio of two real-world assets.  Finally, we tested our Policy Gradient Methods on a full portfolio of seven ETFs and the Policy Gradient Methods were able to pick between a minimum volatility portfolio or a maximum return portfolio, depending on the ***λ*** parameter specified.

In order to validate our models, we split our dataset into a training dataset and testing dataset. In all four Policy Gradient Methods with various risk aversion parameters, the testing dataset resulted in either meeting the portfolio benchmarks of the training dataset or exceeding the benchmarks. 

Our statistical evaluation of our real-world dataset gave us insight to the performance of the ETF’s in our portfolio by providing in-depth analytics and risk metrics. Our evaluation consisted of both two-real world assets model statistics and the full real world portfolio model statistics. Based on the Sharpe Ratio, Sortino Ratio, Calmar Ratio, and Max Drawdown, the ETFs used in our real-world dataset provided us a diverse distribution of assets to be used for our models and therefore provided us with an opportunity to explore Reinforcement Learning in portfolio management.
