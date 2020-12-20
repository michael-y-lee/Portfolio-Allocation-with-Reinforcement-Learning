---
title: User Guide
notebook:
nav_include: 7
---

This is a user guide for how to load real-world data into our model and how to perform Policy-Gradient Methods such as REINFORCE, REINFORCE with Baseline, Actor-Critic, and Actor-Critic with Eligibility Traces.


```python
from environments.e_greedy import DeepTradingEnvironment, LinearAgent

import datetime
import numpy as np
import pandas as pd
import os
from utils import train_val_test
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
from pypfopt.cla import CLA
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import quantstats as qs
qs.extend_pandas()
```


```python
root = os.getcwd()
data_env = root+"/data_env/"
```

# Utility Functions


```python
def _retrieve_asset_dict():
    # obtain close prices from parquet files of ETF price history
    root = os.getcwd()
    data_env = root+"/data_env/"
    files = [_ for _ in os.listdir(data_env) if "parquet" in _]
    assets_dict = {file: pd.read_parquet(data_env + "/" + file) for file in files}
    counter=0
    for key, value in assets_dict.items():
        if counter==0:
            main_index=value.index
    else:
        main_index=main_index.join(value.index,how="inner")
        
    for key, value in assets_dict.items():
        tmp_df=value.reindex(main_index)
        tmp_df=tmp_df.fillna(method='ffill')
        assets_dict[key]=tmp_df['close']
    return assets_dict

def build_portfolio_df(asset_dict):
    portfolio_df = pd.DataFrame()
    
    for key, value in assets_dict.items():
        key = key.split(".")[0]
        tmp_df = pd.DataFrame(data=value)
        tmp_df.columns=[key]
        portfolio_df = pd.concat([portfolio_df, tmp_df], axis=1)
        
    portfolio_df.index = pd.to_datetime(portfolio_df.index, errors='coerce')
    return portfolio_df
```


```python
def plot_backtest(linear_agent_train, env_test, test_input, model):
    ## Create plot of backtest returns
    if not "backtest" in locals():
        backtest=None
    backtest=linear_agent_train.backtest_policy(epoch=1,backtest=backtest, env_test=env_test, test_input=test_input)
    plt.figure(figsize=(8,4))
    plt.plot(backtest,color="blue")
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Date", fontsize = 10)
    plt.ylabel("Backtest", fontsize = 10)
    plt.title("Backtest on Test Data: "+ model,fontsize = 16)
    plt.savefig(root+'/temp_persisted_data/backtest_'+model+'.png')
    plt.show()
    return backtest
```

# Reviewing Real-World Data

The real-world data that we use for our models is located in the "data_env" folder.  If you have other data, please save it in the "data_env" folder as a parquet file.  Below is an example of what the ETF data history file should contain.   


```python
# read a sample ETF
pd.read_parquet(data_env+'EEMV.parquet').head()
```

To view what a portfolio of ETFs looks like, we create a portolio with the ETF files which are in the "data_env" folder.  We can then split these datasets into train and test sets for testing our model performances.


```python
# create a portfolio
assets_dict = _retrieve_asset_dict()
portfolio_df = build_portfolio_df(assets_dict)
```


```python
# create a train dataset and de-mean the time series

portfolio_df_train = portfolio_df[portfolio_df.index <= '2020-04-01']
portfolio_df_train.sub(portfolio_df_train.mean())

portfolio_df_train.head()
```


```python
# create a test dataset consisting of 6 months of data and de-mean the time series

portfolio_df_test = portfolio_df[portfolio_df.index >= '2020-04-16']
portfolio_df_test = portfolio_df_test[portfolio_df_test.index <= '2020-11-16']
portfolio_df_test.sub(portfolio_df_test.mean())

portfolio_df_test.head()
```

# Set Up Environment

Here, we set up the environment to load ETFs from the "data_env" folder and persist features.  We also specify the meta parameters and objective parameters we want to specify for the environment.


```python
# parameters related to the transformation of data, this parameters govern an step before the algorithm
out_reward_window=datetime.timedelta(days=7)

meta_parameters = {"in_bars_count": 14,
                   "out_reward_window":out_reward_window ,
                   "state_type":"in_window_out_window",
                   "risk_aversion":0,
                   "include_previous_weights":False}

# parameters that are related to the objective/reward function construction
objective_parameters = {"percent_commission": .001}

print("===Meta Parameters===")
print(meta_parameters)
print("===Objective Parameters===")
print(objective_parameters)

# create an environment and build features based on Real-World Dataset located in the "data_env" folder 
env = DeepTradingEnvironment.build_environment_from_dirs_and_transform(meta_parameters, objective_parameters,data_hash="real_data", data_dir="data_env")

number_of_assets = env.number_of_assets
```

#### Split Features and Forward Returns into Training and Test sets

For performing backtest return analysis, we split the features and forward returns that were generated by the "build_environment_from_dirs_and_transform" function into a training set and a test set.  These features and forward returns are then de-meaned and will be used to create two separate environments for training and test.


```python
features = pd.read_parquet("temp_persisted_data/only_features_real_data")

features_train = features[features.index <= '2020-04-01']
features_train.sub(features_train.mean()) 

features_test = features[features.index >= '2020-04-16']
features_test = features_test[features_test.index <= '2020-11-16']
features_test.sub(features_test.mean())

features_test.head()
```


```python
forward_return_dates = pd.read_parquet("temp_persisted_data/forward_return_dates_real_data")

forward_return_dates_train = forward_return_dates[forward_return_dates.index <= '2020-04-01']

forward_return_dates_test = forward_return_dates[forward_return_dates.index > '2020-04-16']
forward_return_dates_test = forward_return_dates_test[forward_return_dates_test.index <= '2020-11-16']

forward_return_dates_test.head()
```


```python
forward_returns = pd.read_parquet("temp_persisted_data/only_forward_returns_real_data")

forward_returns_train = forward_returns[forward_returns.index <= '2020-04-01']
forward_returns_train.sub(forward_returns_train.mean()) 

forward_returns_test = forward_returns[forward_returns.index >= '2020-04-16']
forward_returns_test = forward_returns_test[forward_returns_test.index <= '2020-11-16']
forward_returns_test.sub(forward_returns_test.mean()) 

forward_returns_test.head()
```

# Run Policy Gradient Method Algorithms on Real-World Data

Now that we have our features and forward returns divded into training and test set, we can perform our policy gradient method algorithms on the real-world data which we loaded.  

- For each PGM, we first create a train and test environment based on the features and forward_returns of that data.  

- We then instantiate a Linear Agent based on the train environment and a specified reward function (we are currently using "return_with_variance_risk" but one could also use "cum_return", "max_sharpe", "min_vol", or "min_variance").

- Once the Linear Agent is instantiated, we then call REINFORCE_fit() or ACTOR_CRITIC_fit().  We can choose whether to include baseline for REINFORCE or eligibility traces for Actor-Critic by specifying the appropriate flag when calling the function).

- As the model runs, the progress status is listed and status plots are displayed based on the interval of plotting the user specified.  To turn off the print display, set verbose=False when calling the model function.

- Once the model has completed training, view the backtest returns of the test dataset by calling the plot_backtest() function.  Backtest results can be saved to a CSV file.


```python
max_iter = 10000
model_run = "demeaned_return_reward_variance_risk_0_"
sample_observations = 4
```

### REINFORCE


```python
# create environment and run REINFORCE

env_reinforce_train=DeepTradingEnvironment(features_train, forward_returns_train, forward_return_dates_train, objective_parameters,
                 meta_parameters)
env_reinforce_test = DeepTradingEnvironment(features_test, forward_returns_test, forward_return_dates_test, objective_parameters,
                 meta_parameters)

linear_agent_reinforce = LinearAgent(environment=env_reinforce_train,out_reward_window_td=out_reward_window, reward_function="return_with_variance_risk",sample_observations=sample_observations)
linear_agent_reinforce.REINFORCE_fit(max_iterations=max_iter, add_baseline=False, verbose=True)
```


```python
# perform backtest
backtest_reinforce = plot_backtest(linear_agent_reinforce, env_reinforce_test, portfolio_df_test, model="REINFORCE")
```

### REINFORCE with Baseline


```python
# create environment and run REINFORCE with baseline
env_reinforce_baseline_train = DeepTradingEnvironment(features_train, forward_returns_train, forward_return_dates_train, objective_parameters,
                 meta_parameters)
env_reinforce_baseline_test = DeepTradingEnvironment(features_test, forward_returns_test, forward_return_dates_test, objective_parameters,
                 meta_parameters)

linear_agent_reinforce_baseline = LinearAgent(environment=env_reinforce_baseline_train,out_reward_window_td=out_reward_window, reward_function="return_with_variance_risk",sample_observations=sample_observations)
linear_agent_reinforce_baseline.REINFORCE_fit(max_iterations=max_iter, add_baseline=True, verbose=True)
```


```python
# perform backtest
backtest_reinforce_baseline = plot_backtest(linear_agent_reinforce_baseline, env_reinforce_baseline_test, portfolio_df_test, model="REINFORCE with Baseline")
```

### Actor-Critic


```python
# create environment and run Actor-Critic 

env_actor_critic_no_trace_train = DeepTradingEnvironment(features_train, forward_returns_train, forward_return_dates_train, objective_parameters,
                 meta_parameters)
env_actor_critic_no_trace_test = DeepTradingEnvironment(features_test, forward_returns_test, forward_return_dates_test, objective_parameters,
                 meta_parameters)

linear_agent_actor_critic_no_trace = LinearAgent(environment=env_actor_critic_no_trace_train,out_reward_window_td=out_reward_window, reward_function="return_with_variance_risk",sample_observations=sample_observations)
linear_agent_actor_critic_no_trace.ACTOR_CRITIC_FIT(use_traces=False,max_iterations=max_iter, verbose=True)
```


```python
# perform backtest 
backtest_actor_critic_no_trace = plot_backtest(linear_agent_actor_critic_no_trace, env_actor_critic_no_trace_test,  portfolio_df_test, model="Actor-Critic without Eligibility Traces")
```

### Actor-Critic with Eligibility Traces


```python
# create environment and run Actor-Critic with Eligibility Traces 
env_actor_critic_trace_train = DeepTradingEnvironment(features_train, forward_returns_train, forward_return_dates_train, objective_parameters,
                 meta_parameters)
env_actor_critic_trace_test = DeepTradingEnvironment(features_test, forward_returns_test, forward_return_dates_test, objective_parameters,
                 meta_parameters)

linear_agent_actor_critic_trace = LinearAgent(environment=env_actor_critic_trace_train,out_reward_window_td=out_reward_window, reward_function="return_with_variance_risk",sample_observations=sample_observations)
linear_agent_actor_critic_trace.ACTOR_CRITIC_FIT(use_traces=True,max_iterations=max_iter, verbose=True)
```


```python
# perform backtest
backtest_actor_critic_trace = plot_backtest(linear_agent_actor_critic_trace, env_actor_critic_trace_test,  portfolio_df_test, model="Actor-Critic with Eligibility Traces")
```
