This repository contains the primary valuation analysis used to conduct an intrinsic valuation of Atlassian (NASDAQ:TEAM). Included is my excel DCF file, fama-french calculations, standard deviation calculations for my Monte Carlo simulation, and finally my Monte Carlo Simulation.

My excel file contains a forecasted three-statement DCF model, comparable peers valuation, sensitivity analysis, reverse DCF, CAPM calculation, as well as two discount for 
lack of marketability models (Chaffe & Finnerty) which I constructed in an attempt to value the cash-equivalent cost of TEAM's stock-based compensation, accounting for the 
loss of the ability to sell by valuing the loss as a put, and subtracting it from the value of the SBC. However, I found that this method was redundant, as accounting
for the future value of dilution cancelled out the additional cash-equivalent cash flow. 

My Fama-French python file contains calculations for TEAM's Fama-French cost-of-equity (CoE) models, both 3-Factor and 5-Factor. I elected to proceed with the Fama-French
3-Factor model, as it had a higher R^2 value (0.175) than the CAPM (0.136), while the 5-Factor model was rendered unusable due to producing a negative CoE (-1.62%). This
negative CoE was driven by negative RMW & CMA coefficents (-2.6660 & -1.2589 respectively). A negative RMW for TEAM is expected as this is commonplace in companies with
heavy R&D and negative EBIT, both of which are native to TEAM current financial profile. A negative CMA can be explained by TEAM's aggressive expansion over the last few years,
typical for high-growth software companies as they expand operations. 

My standard deviation python file is critical for my Monte Carlo simulation, as it is the backbone for simulation spreads. It covers standard deviation for FCF margins,
sales growth (calculated from historical margins), WACC (calculated from Fama-French 3-Factor regression), and terminal growth (calculated from 40-year US nominal GDP 1986-2026).

Finally, my Monte Carlo simulation (n = 50,000) ran four key variables, sales growth, FCF margin, WACC & terminal growth. These variables were slected based upon independent 
review and determination of the most impactful variables on potential share price. Forecast margins were dervied from my DCF model, thus my Monte Carlo simulation mean & median
both returned values close to the derived DCF price.
