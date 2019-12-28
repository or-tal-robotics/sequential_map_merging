#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 13:45:54 2019

@author: or
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_de = pd.read_csv("MonteCarloStatistics_de.csv")
data_de = data_de.values
data_pf = pd.read_csv("MonteCarloStatistics_pf.csv")
data_pf = data_pf.values
avg_de = np.mean(data_de, axis=0)
avg_pf = np.mean(data_pf, axis=0)
best_run_pf = np.where(data_pf == np.min(data_pf[:,-1]))[0]
best_run_de = np.where(data_de == np.min(data_de[:,-1]))[0]
plt.figure(1)
plt.plot(avg_de, color = 'r', label = "DE")
plt.plot(avg_pf, color = 'b', label = "PF")
plt.legend()
plt.xlabel("t")
plt.ylabel("MSE")
plt.title("100 Monte Carlo runs")

plt.figure(2)
plt.plot(data_pf[best_run_pf,:].reshape(-1), color = 'k', label = "PF")
plt.plot(data_de[best_run_de,:].reshape(-1), color = 'y', label = "DE")
plt.legend()
plt.xlabel("t")
plt.ylabel("MSE")
plt.title("Best runs")

plt.show()


