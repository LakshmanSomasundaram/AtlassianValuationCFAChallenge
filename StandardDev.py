# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:20:51 2026

@author: lsoma
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
# %%
TEAM_raw = pd.read_excel('PriceHistory.xlsx',skiprows=2)
SPX_raw = pd.read_excel('SPX.xlsx',skiprows=15)

TEAM = TEAM_raw[['Date','Price']].rename(columns={'Price':'TEAM'})
SPX = SPX_raw[['Date','Price']].rename(columns={'Price':'SP'})

df = pd.merge(SPX, TEAM, on='Date', how='inner')
df = df.sort_values('Date').reset_index(drop=True)

print(df.head())
print(df.tail())

df['SPX_ret'] = df['SP'].pct_change()
df['TEAM_ret'] = df['TEAM'].pct_change()

df_ret = df.dropna(subset=['SPX_ret', 'TEAM_ret'])
# %% WACC CAPM
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df_ret['SPX_ret'],
    df_ret['TEAM_ret']
    )
print(f"Beta: {slope:.4f}")
print(f"Std Error of Beta: {std_err:.4f}")
print(f"R-squared: {r_value**2:.4f}")
print(f"t-stat: {slope/std_err:.2f}")

MRP = 0.0446
We =  0.9692 

wacc_std_dev = We * MRP * std_err
print(f"WACC Std Dev: {wacc_std_dev:.4%}")
# %%
historical_growth = np.array([0.29424,0.34165,0.26108,0.23311,0.19655,0.26020])
years = np.arange(len(historical_growth))
print ("Time index:", years)
slope, intercept, r_value, p_value, std_err = stats.linregress(years,historical_growth)
print(f"Trend slope: {slope:.4f}")
print(f"Trend intercept: {intercept:.4f}")
predicted = slope * years + intercept
print("Predicted (trend) values:", predicted)
residuals = historical_growth - predicted
print("Residuals:", residuals)
growth_std_dev = np.std(residuals, ddof=2)
print(f"\nSales growth std dev (detrended): {growth_std_dev:.4%}")
print(f"Raw std dev (no detrend):     {np.std(historical_growth, ddof=1):.4%}")
# %%
historical_fcf_margin = np.array([-0.12134,-0.24280,-0.14263,-0.04935,0.06796,0.00360,-0.05490])
years_hist = np.arange(len(historical_fcf_margin))

slope_hist, intercept_hist, r_hist, p_hist, se_hist = stats.linregress(years_hist, historical_fcf_margin)
predicted_hist = slope_hist * years_hist + intercept_hist        
residuals_hist = historical_fcf_margin - predicted_hist

fcf_margin_std = np.std(historical_fcf_margin,ddof=1)
print(f"Trend slope: {slope_hist:.4%} per year   (p = {p_hist:.4f}, r2 = {r_hist**2:.4f})")
print(f"Detrended std dev: {np.std(residuals_hist, ddof=2):.4%}")
print(f"Raw std dev:       {fcf_margin_std:.4%}")
# %%
FRED = pd.read_excel('GDP_PCH.xlsx', sheet_name='Annual')
FRED['observation_date'] = pd.to_datetime(FRED['observation_date'])
FRED_clean = FRED.dropna(subset=['GDP_PCH'])
recent = FRED_clean[FRED_clean['observation_date'] >= '1986-01-01']
gdp_growth = recent['GDP_PCH'].to_numpy()
print(gdp_growth)
print(len(gdp_growth))
gdp_growth = gdp_growth / 100
gdp_growth_std_dev = np.std(gdp_growth, ddof=1)
gdp_growth_mean = np.mean(gdp_growth)
print(f"GDP growth std dev (last 40y): {gdp_growth_std_dev:.4%}")
print(f"GDP growth mean (last 40y): {gdp_growth_mean:.4%}")
# %%
n = len(gdp_growth)
se_mean = gdp_growth_std_dev / np.sqrt(n)
print(f"M1- SE of Mean: {se_mean:.4%}")
n_blocks = n // 10
blocks = gdp_growth[:n_blocks*10].reshape(n_blocks, 10)
block_means = blocks.mean(axis=1)
years_arr = recent['observation_date'].dt.year.to_numpy()[:n_blocks*10].reshape(n_blocks, 10)
for i in range(n_blocks):
    print(f"  {years_arr[i][0]}-{years_arr[i][-1]}: {block_means[i]:.4%}")
gL_std_decade = np.std(block_means, ddof=1)
print(f"M2 - std across decade means: {gL_std_decade:.4%}")

     
# %%

df['SPX_ret'] = df['SP'].pct_change()
df['TEAM_ret'] = df['TEAM'].pct_change()

# Rename to match the labels exactly, and reorder
df_export = df.rename(columns={
    'SP': 'S&P 500',
    'TEAM': 'TEAM',
    'SPX_ret': 'S&P 500 Return',
    'TEAM_ret': 'TEAM Return'
})

df_export = df_export[['Date', 'S&P 500', 'TEAM', 'S&P 500 Return', 'TEAM Return']]

df_export.to_excel(r'c:\users\lsoma\onedrive\desktop\spyder_practice\TEAM_SPX_data.xlsx', index=False)
print("Saved successfully")     