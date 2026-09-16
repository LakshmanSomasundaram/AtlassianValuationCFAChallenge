# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:37:19 2026

@author: lsoma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
# %%
#Fixed inputs
sales_base = 6572.31
cash = 1239.51
debt = 1232.91
net_debt = debt - cash
diluted_shares = 260.16
n_forecast_years = 6
current_share_price = 173.72
DCF_point_estimate = 111.14
# %%
# Distribution parameters
sales_growth_mean=np.array([ 0.147292,0.211080,0.206258,0.178051,0.130314,0.101458])
sales_growth_std = 0.041039

fcf_margin_mean = np.array([ 0.06620,0.08091,0.09201,0.11256,0.12872,0.14996 ])
fcf_margin_std = 0.081759
rho = 0.7
sigma_ln = np.sqrt(np.log(1+(fcf_margin_std/fcf_margin_mean)**2))
mu_ln = np.log(fcf_margin_mean) - 0.5*sigma_ln**2

wacc_mean = 0.0965
wacc_std =  0.020759817

gL_mean = 0.035
gL_std =  0.010689

# %%
#Regular Gordon Growth Model
def run_one_simulation():
    gL_draw = np.random.normal(gL_mean,gL_std)
    lower = gL_draw + 0.02
    a = (lower - wacc_mean) / wacc_std
    wacc_draw = truncnorm.rvs(a,np.inf,loc=wacc_mean,scale=wacc_std)
    
    sales_growth_draw = np.random.normal(sales_growth_mean, sales_growth_std)
    z_common = np.random.normal()
    z_specific = np.random.normal(size = n_forecast_years)
    z = np.sqrt(rho)*z_common + np.sqrt(1-rho)*z_specific
    fcf_margin_draw = np.exp(mu_ln + sigma_ln*z)

    sales = np.zeros(n_forecast_years)
    sales[0] = sales_base * (1+sales_growth_draw[0])
    for t in range(1,n_forecast_years):
        sales[t] = sales[t-1] * (1+sales_growth_draw[t])
        
    fcf = sales * fcf_margin_draw
    
    t = np.arange(1, n_forecast_years + 1)
    discount_factors = 1/(1+wacc_draw)**t
    pv_fcf = fcf*discount_factors
    pv_fcf_total = np.sum(pv_fcf)
    
    fcf_final_year = fcf[-1]
    
    terminal_value = fcf_final_year * (1 + gL_draw)/(wacc_draw -gL_draw)
    pv_terminal_value = terminal_value / (1 + wacc_draw) ** (n_forecast_years)  
    
    enterprise_value = pv_fcf_total + pv_terminal_value
    equity_value = enterprise_value - net_debt
    share_price = equity_value / diluted_shares
    
    return share_price
    
np.random.seed(42)
results = np.array([run_one_simulation() for _ in range(50000)])

print("Sample size:", len(results))
print("Mean:", np.mean(results))
print("Median:", np.median(results))
print("Std dev:", np.std(results, ddof=1))
print("Min:", np.min(results), " Max:", np.max(results))
   
n_negative = np.sum(results< 0)
print(f"Negative share prices: {n_negative} ({n_negative/50000:.2%})")
# %%
#Percentiles
percentiles = [5,10,25,50,75,90,95]
percentile_values = np.percentile(results,percentiles)

for p,v in zip(percentiles, percentile_values):
    print(f"{p}th percentile: ${v:.2f}")

for trial in range(5):
    trial_results = np.array([run_one_simulation() for _ in range(10000)])
    print(f"Trial {trial+1} — 5th percentile: ${np.percentile(trial_results, 5):.2f}, Median: ${np.median(trial_results):.2f}")
# %%
for label, val in [("DCF point estimate", 111.14),
                   ("Monte Carlo median", np.median(results)),
                   ("Monte Carlo mean", np.mean(results)),
                   ("Current market price", current_share_price)]:
    pct = (results < val).mean()
    print(f"{label:22s} ${val:7.2f}  {pct:5.1%} percentile")
# %%
    
prob_above_current = np.mean(results > current_share_price)
prob_below_current = np.mean(results < current_share_price)
print(f"Probability stock is undervalued (implied price > ${current_share_price}): {prob_above_current:.2%}")
print(f"Probability stock is overvalued (implied price < ${current_share_price}): {prob_below_current:.2%}")

pct_overvalued_by_10plus = np.mean(results < current_share_price *0.90)
pct_overvalued_by_20plus = np.mean(results < current_share_price*0.80)
print (f"Probability overvalued by 10%+: {pct_overvalued_by_10plus:.2%}")
print (f"Probability overvalued by 20%+: {pct_overvalued_by_20plus:.2%}")
# %%
#Histogram
results_display = results[results > 0]
plt.figure(figsize=(10,6))
plt.hist(results_display, bins=300, color='#0067ED', edgecolor = 'white', alpha=0.85)
plt.axvline(current_share_price, color = 'red', linestyle ='--', linewidth=2, label=f'Current Price (${current_share_price:.2f})')
plt.axvline(np.median(results), color='black', linestyle ='-', linewidth = 2, label =f'Median (${np.median(results):.2f})')
plt.axvline(np.mean(results), color='orange', linestyle ='-.', linewidth = 2, label =f'Mean (${np.mean(results):.2f})')
plt.axvline(DCF_point_estimate, color = 'fuchsia', linestyle = '--', linewidth = 2, label = f'DCF Valuation (${DCF_point_estimate:.2f})')

plt.xlabel('Implied Share Price ($)', fontsize=11)
plt.ylabel ('Number of Simulations', fontsize=11)
plt.title ('Atlassian (TEAM) - Monte Carlo DCF Simulation\nDistribution of Implied Share Price (n=50,000)', fontsize = 13)
plt.xlim(0,500)
plt.legend()
plt.tight_layout()
plt.savefig('atlassian_monte_carlo_histogram.png', dpi=300, bbox_inches='tight')
plt.show()
# %%
#VaR 95%, 99%
var_5 = current_share_price - np.percentile(results, 5)
var_1 = current_share_price - np.percentile(results, 1)

print(f"VaR (95% confidence): ${var_5:.2f} ({var_5/current_share_price:.2%} of current price)")
print(f"VaR (99% confidence): ${var_1:.2f} ({var_1/current_share_price:.2%} of current price)")
# %%
#Conditional VaR
implied_returns = (results - current_share_price)/ current_share_price

var_95_r = np.percentile(implied_returns, 5)
var_99_r = np.percentile(implied_returns,1)

ES_95_r = implied_returns[implied_returns <= var_95_r].mean()
ES_99_r = implied_returns[implied_returns <= var_99_r].mean()

n_tail_95 = np.sum(implied_returns <= var_95_r)
n_tail_99 = np.sum(implied_returns <=var_99_r)

ES_95_P = (ES_95_r * current_share_price)
ES_99_P = (ES_99_r * current_share_price)


print(f"VaR (95%): {var_95_r:.2%}   ES (95%): {ES_95_r:.2%}   [n={n_tail_95}]")
print(f"VaR (99%): {var_99_r:.2%}   ES (99%): {ES_99_r:.2%}   [n={n_tail_99}]")
print(f"ES95($): {ES_95_P:.2f}    ES99($): {ES_99_P:.2f}")
# %%

import matplotlib.ticker

results_display = results[results > 0]

# quartile boundaries from the FULL results array
q1, q2, q3 = np.percentile(results, [25, 50, 75])
quartile_colors = ['#FBC828', '#AF59E1', '#0067ED', '#B3D1FA']

fig, ax = plt.subplots(figsize=(10,6))

# borders: bottom only, thin black
for name, spine in ax.spines.items():
    spine.set_visible(name == 'bottom')
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(0.8)
ax.tick_params(axis='y', length=0, labelsize=13)
ax.tick_params(axis='x', length=14, width=2.5, color='black', pad=10, labelsize=13)

# histogram, weighted to % of simulations
weights = np.ones_like(results_display) / len(results) * 100
n, bins, patches = ax.hist(results_display, bins=500, weights=weights,
                           edgecolor='white', linewidth=0.2)

# recolour each bar by which quartile its centre falls in
bin_width = bins[1] - bins[0]
for patch, left_edge in zip(patches, bins[:-1]):
    centre = left_edge + bin_width/2
    if centre < q1:
        patch.set_facecolor(quartile_colors[0])
    elif centre < q2:
        patch.set_facecolor(quartile_colors[1])
    elif centre < q3:
        patch.set_facecolor(quartile_colors[2])
    else:
        patch.set_facecolor(quartile_colors[3])

ax.set_ylabel('% of Simulations', fontsize=11)
ax.set_xlim(0, 500)

# major ticks = the two price markers (big ticks, word labels)
ax.set_xticks([DCF_point_estimate, current_share_price])
ax.set_xticklabels(['Target Price', 'Current Price'])
ax.get_xticklabels()[0].set_ha('right')
ax.get_xticklabels()[1].set_ha('left')

ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=1))

plt.tight_layout()
plt.savefig('atlassian_monte_carlo_histogram.png', dpi=300,
            bbox_inches='tight', transparent=True)
plt.show()
