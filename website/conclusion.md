---
title: Conclusion
notebook:
nav_include: 6
---

Applying Reinforcement Learning to portfolio management was the objective of this capstone project. We determined that this could be accomplished by using Policy Gradient Methods such as REINFORCE, REINFORCE with baseline, Actor-Critic, and Actor-Critic with Eligibility Traces to build a model-free agent which selects portfolio weights.

By using a control dataset of simulated data, we were able to demonstrate model convergence for the four Policy Gradient Methods we tried. However, when we tried to apply these Policy Gradient models to real-world data, the results were not as evident. The Policy Gradient Methods did not converge with the real-world dataset and the policy action weights were distributed relatively evenly across all the assets.

Our statistical evaluation of our real-world dataset gave us insight to the performance of the ETF’s in our portfolio by providing in-depth analytics and risk metrics. Based on the Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, and Volatility metrics, the ETFs used in our real-world dataset provided us a diverse distribution of assets to be used for our models. Alongside our real-world dataset, we conducted a statistical evaluation of our de-meaned backtests with risk aversion factors of 0 and 10.
