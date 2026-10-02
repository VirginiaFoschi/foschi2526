# Reinforcement Learning for Electric Vehicles Charging
 
This repository contains the implementation, experimental results, and the final report for the Multi-Agent Systems course projectwork (3 CFU), academic year 2025/2026.

Author: Foschi Virginia

## Abstract
This project proposes the development of a Reinforcement Learning agent capable of scheduling
the overnight charging of an electric vehicle (EV) in order to minimise the electricity cost paid on
the wholesale day-ahead spot market. The environment is based on real hourly wholesale electricity
price data for Germany, sourced from the Ember Energy European Wholesale Electricity Price
Dataset, which provides average hourly day-ahead prices for European countries over multiple
years.
The scenario simulates a realistic home-charging situation: an EV arrives home at a random
time in the evening (between 17:00 and 20:00), and the driver needs the battery sufficiently charged
by departure the following morning (07:00). During this window, an agent decides each hour
whether to charge at full power, charge at half power, or do nothing. The goal is to minimise total
charging cost while always ensuring the car is ready on time.

## Repository structure
 
```
foschi2526/
  ├── agent/
  │   ├── baseline.py
  │   ├── q_learning_agent.py
  │   ├── rollout_env.py
  │   ├── run_baseline.py
  │   └── utils.py         
  ├── data/
  │   ├── data_exploration.ipynb
  │   ├── Germany.csv
  │   └── price_loader.py            
  ├── environment/
  │   ├── discrete_env.py
  │   └── model_env.py              
  ├── figures/                     
  ├── q_agent_chekpoint/          
  ├── plot_agent.py
  ├── result_analysis.ipynb
  ├── train_agent.py        
  ├── environment.yml     
  ├── .gitignore
  └── README.md
```

## Overview
- **Scenario:** an EV arrives home at a random hour between 17:00 and 20:00 with a partially depleted battery and must reach 80% state of charge (SoC) by 07:00 the next morning.
- **Decision:** at every hour, the agent chooses to charge at full power (11 kW), half power (5.5 kW), or stay idle.
- **Goal:** minimise the total cost of electricity drawn over the night, without missing the departure target.
- **Data:** hourly German day-ahead prices from the Ember Energy European Wholesale Electricity Price Dataset, years 2021–2025.
- **Method:** tabular Q-learning on a custom Gymnasium environment (`EVChargingEnvDiscrete`).

Each night is treated as an independent episode. Prices from 2021–2023 are used for training (1094 nights) and prices from 2024–2025 for validation (730 nights)

## How It Works

### Environment (MDP)

| Component    | Description |
|--------------|-------------|
| **State**    | `(soc_bin, hours_until_departure, price_bin)`: SoC in 10 equal-width bins, the integer countdown to 07:00, and the current price in 3 percentile-based bins (low / medium / high, fitted on the training data only). At most 10 × 14 × 3 = 420 decision states |
| **Actions**  | `0` = idle (0 kW), `1` = half power (5.5 kW), `2` = full power (11 kW) |
| **Dynamics** | 60 kWh battery, 90% charging efficiency, 1-hour steps |
| **Reward**   | Negative electricity cost. At the last step a shortfall penalty of 2 EUR per kWh missing below the 80% SoC target is subtracted. Negative prices can yield positive rewards |
| **Episodes** | Arrival hour sampled uniformly from {17, 18, 19, 20}, arrival SoC from [0.15, 0.45], fixed departure at 07:00, so each episode lasts 11–14 steps |

### Policies

- **`NaiveImmediateChargePolicy`:** ignores prices, charges at full power until the target SoC is reached, then stays idle
- **`TrainedQPolicy`:** acts greedily with respect to the learned Q-table (no exploration at evaluation time).

### Q-learning hyperparameters

| Parameter                   | Value |
|-----------------------------|-------|
| Training episodes           | 30,000 |
| Discount factor γ           | 0.95 |
| Learning rate α             | 0.10 → 0.02 |
| Exploration ε (ε-greedy)    | 1.00 → 0.05 |
| Q-table initialisation      | 0.0 |
| Random seed                 | 0 |
| SoC bins / price bins       | 10 / 3 |


## Key Results

| Split      | Policy     | Nights | Mean cost / night (EUR) | Total cost (EUR) |
|------------|------------|-------:|------------------------:|-----------------:|
| Train      | Q-learning | 1094   | 4.482                   | 4903.2           |
| Train      | Naive      | 1094   | 6.740                   | 7373.3           |
| Validation | Q-learning | 730    | **2.730**               | 1992.8           |
| Validation | Naive      | 730    | 4.380                   | 3197.3           |

## How to reproduce results
### 1. Environment and data preparation
 
1. Dowload the Ember Energy European Wholesale Electricity Price
   Dataset from the following link (https://ember-energy.org/data/european-wholesale-electricity-price-data/)
   and extract the files
2. Create the data directory and place the Ember Energy Germany price CSV Germany.csv into it.
3. Create the required Conda environment from the supplied environment
   file (`environment.yml`), which specifies all required Python
   packages and their dependencies:
```bash
   conda env create -f environment.yml
   conda activate environment
```

### 2. Train the agent

```bash
python -m train_agent
```

This runs the training loop with periodic validation on the validation years and prints the final comparison between the Q-learning and naive policies. The trained Q-table is saved to `q_agent_checkpoint/agent_params.pkl`.

### 3. Analyse the results

Open and run the `results_analysis.ipynb` notebook. It loads the saved Q-table, evaluates the learned policy together with the naive baseline, and reproduces the plots used in the report:

### 4. Evaluate the baseline only (optional)

```bash
python -m agent.run_baseline
```

This evaluates the naive immediate-charging policy independently 
