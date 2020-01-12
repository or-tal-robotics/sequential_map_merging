#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 13:45:54 2019

@author: or
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib._color_data as mcd

data_de = pd.read_csv("csv/MonteCarloStatistics_de_map10V3Disap.csv", sep='\t')
data_de = data_de.values
data_pf = pd.read_csv("csv/MonteCarloStatistics_pf_map10V3Disap.csv",sep='\t')
data_pf = data_pf.values
avg_de = np.mean(data_de, axis=0)
avg_pf = np.mean(data_pf, axis=0)
best_run_pf = np.where(data_pf == np.min(data_pf[:,-1]))[0]
best_run_de = np.where(data_de == np.min(data_de[:,-1]))[0]

m = []
for ii in range(data_de.shape[0]):
    m.append(np.mean(data_de[ii]))
m = np.array(m)
best_run_de_overall = np.where(m == np.min(m))[0]
best_run_de_overall = data_de[best_run_de_overall].reshape((-1))

m = []
for ii in range(data_pf.shape[0]):
    m.append(np.mean(data_pf[ii]))
m = np.array(m)
best_run_pf_overall = np.where(m == np.min(m))[0]
best_run_pf_overall = data_pf[best_run_pf_overall].reshape((-1))


best_de_line = data_de[best_run_de,-1]*np.ones_like(avg_pf)

plt.figure(1)
plt.plot(avg_de, color = 'r', label = "DENDT (average)", linewidth = 1.7)
plt.plot(avg_pf, color = 'b', label = "PF (average)" , linewidth = 1.7)
plt.plot(best_run_de_overall, color = 'r', label = "DENDT (best one)", ls='dotted')
plt.plot(best_run_pf_overall, color = 'b', label = "PF (best one)", ls='dotted')
plt.xlim(1,150)
#plt.legend()
plt.xlabel("k")
plt.ylabel("MSE")

#plt.title("100 Monte Carlo runs")

#plt.figure(2)
#plt.plot(data_pf[best_run_pf,:].reshape(-1), color = 'k', label = "PF")
#plt.plot(data_de[best_run_de,:].reshape(-1), color = 'y', label = "DE")
#plt.legend()
#plt.xlabel("t")
#plt.ylabel("MSE")
#plt.title("Best runs")

plt.show()


