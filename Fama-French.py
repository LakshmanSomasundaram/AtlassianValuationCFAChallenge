# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:11:04 2026

@author: lsoma
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
import statsmodels.api as sm
# %%
#Data import
TEAM_SPX = pd.read_excel('TEAM_SPX_data.xlsx')
ThreeFactor = pd.read_csv('ThreeFactor.csv',skiprows=4)
FiveFactor = pd.read_csv('FiveFactor.csv',skiprows=4)
# %%
ThreeFactor = ThreeFactor[ThreeFactor['Date'].astype(str).str.match(r'^\d{8}$')]
ThreeFactor.columns = ['Date', 'Mkt-RF','SMB','HML','RF']
ThreeFactor['Date'] = pd.to_datetime(ThreeFactor['Date'], format='%Y%m%d')
ThreeFactor[['Mkt-RF', 'SMB','HML','RF']] = ThreeFactor[['Mkt-RF','SMB','HML','RF']]/100

print(ThreeFactor.head())
print(ThreeFactor.tail())
print(ThreeFactor['Date'].dt.day_name().unique())
# %%
FiveFactor.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW','CMA','RF']
FiveFactor = FiveFactor[FiveFactor['Date'].astype(str).str.match(r'^\d{8}$')]
FiveFactor['Date'] = pd.to_datetime(FiveFactor['Date'], format='%Y%m%d')
FiveFactor[['Mkt-RF', 'SMB', 'HML','RMW','CMA','RF']] = FiveFactor[['Mkt-RF', 'SMB', 'HML','RMW','CMA','RF']]/100

print(FiveFactor.head())
print(FiveFactor.tail())

FiveFactor_indexed = FiveFactor.set_index('Date')
FiveFactor_weekly = (1+FiveFactor_indexed).resample('W-FRI').prod()-1
FiveFactor_weekly = FiveFactor_weekly.reset_index()

print(FiveFactor_weekly.head())
print(FiveFactor_weekly.tail())
# %%
TEAM_SPX_sorted = TEAM_SPX.sort_values('Date').reset_index(drop=True)
ThreeFactor_sorted = ThreeFactor.sort_values('Date').reset_index(drop=True)
FiveFactor_weekly_sorted = FiveFactor_weekly.sort_values('Date').reset_index(drop=True)

merged_3f = pd.merge_asof(TEAM_SPX_sorted,ThreeFactor_sorted, on='Date',direction ='nearest', tolerance=pd.Timedelta('3D'))

print(merged_3f.head())
print(merged_3f.tail())
print("Rows with missing factor data:",merged_3f['Mkt-RF'].isna().sum())

merged_5f = pd.merge_asof(TEAM_SPX_sorted, FiveFactor_weekly_sorted, on='Date', direction='nearest', tolerance=pd.Timedelta('3D'))

print(merged_5f.head())
print(merged_5f.tail())
print("Rows with missing factor data:", merged_5f['Mkt-RF'].isna().sum())
# %%
merged_3f_clean = merged_3f.dropna(subset=['Mkt-RF', 'SMB', 'HML'])
merged_5f_clean = merged_5f.dropna(subset=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])

print("3-Factor regression sample size:", len(merged_3f_clean))
print("5-Factor regression sample size:", len(merged_5f_clean))
# %%
merged_3f_clean['TEAM_excess'] = merged_3f_clean['TEAM Return'] - merged_3f_clean['RF']
merged_5f_clean['TEAM_excess'] = merged_5f_clean['TEAM Return'] - merged_5f_clean['RF']

merged_3f_clean = merged_3f.dropna(subset=['Mkt-RF', 'SMB', 'HML', 'TEAM Return']).copy()
merged_3f_clean['TEAM_excess'] = merged_3f_clean['TEAM Return'] - merged_3f_clean['RF']

print("Rows remaining:", len(merged_3f_clean))
print("Any NaN in TEAM_excess:", merged_3f_clean['TEAM_excess'].isna().sum())


X_3f = merged_3f_clean[['Mkt-RF','SMB','HML']]
X_3f = sm.add_constant(X_3f)
y_3f = merged_3f_clean['TEAM_excess']

model_3f = sm.OLS(y_3f,X_3f).fit()
print(model_3f.summary())

merged_5f_clean = merged_5f.dropna(subset=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'TEAM Return']).copy()
merged_5f_clean['TEAM_excess'] = merged_5f_clean['TEAM Return'] - merged_5f_clean['RF']

X_5f = merged_5f_clean[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
X_5f = sm.add_constant(X_5f)
y_5f = merged_5f_clean['TEAM_excess']

model_5f = sm.OLS(y_5f, X_5f).fit()
print(model_5f.summary())
# %%
#Long run average weekly premium
smb_premium_3f = ThreeFactor['SMB'].mean() * 52
hml_premium_3f = ThreeFactor['HML'].mean() * 52

rmw_premium_5f = FiveFactor['RMW'].mean() * 252   # daily data, so annualize with 252 trading days
cma_premium_5f = FiveFactor['CMA'].mean() * 252
smb_premium_5f = FiveFactor['SMB'].mean() * 252
hml_premium_5f = FiveFactor['HML'].mean() * 252

print(f"SMB premium (3F dataset, annualized): {smb_premium_3f:.4%}")
print(f"HML premium (3F dataset, annualized): {hml_premium_3f:.4%}")
print(f"RMW premium (5F dataset, annualized): {rmw_premium_5f:.4%}")
print(f"CMA premium (5F dataset, annualized): {cma_premium_5f:.4%}")

smb_premium_5f = FiveFactor['SMB'].mean() * 252
hml_premium_5f = FiveFactor['HML'].mean() * 252

print(f"SMB premium (5F dataset, annualized): {smb_premium_5f:.4%}")
print(f"HML premium (5F dataset, annualized): {hml_premium_5f:.4%}")
# %%
We =  0.9685 
MRP = 0.0446

q3 = np.array([0, MRP, smb_premium_3f, hml_premium_3f])
q5 = np.array([0, MRP, smb_premium_5f, hml_premium_5f, rmw_premium_5f, cma_premium_5f])

for name, model, q in [('FF3', model_3f, q3), ('FF5',model_5f,q5)]:
    V = model.cov_params().values
    sigma_Ke = np.sqrt(q @ V @ q)
    print(f'{name}: sigma_Ke {sigma_Ke:.4%} sigma_WACC {We*sigma_Ke:.4%}')
# %%


#Excel extraction

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Cost of Equity Analysis"

# ---------- Styles ----------
title_font = Font(name="Arial", size=14, bold=True, color="FFFFFF")
subtitle_font = Font(name="Arial", size=10, italic=True, color="595959")
header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
section_font = Font(name="Arial", size=11, bold=True, color="1F4E78")
label_font = Font(name="Arial", size=10)
result_font = Font(name="Arial", size=10, bold=True)
warn_font = Font(name="Arial", size=10, bold=True, color="C00000")
note_font = Font(name="Arial", size=9, italic=True, color="808080")

title_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
header_fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
summary_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
warn_fill = PatternFill(start_color="FCE4E4", end_color="FCE4E4", fill_type="solid")
alt_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")

thin = Side(style="thin", color="BFBFBF")
border = Border(left=thin, right=thin, top=thin, bottom=thin)

def cell(r, c, value, font=label_font, fill=None, numfmt=None, align="left", indent=0, wrap=False):
    cl = ws.cell(row=r, column=c, value=value)
    cl.font = font
    if fill: cl.fill = fill
    if numfmt: cl.number_format = numfmt
    cl.alignment = Alignment(horizontal=align, indent=indent, wrap_text=wrap, vertical="center")
    cl.border = border
    return cl

# ---------- Title ----------
ws.merge_cells("A1:H1")
cell(1, 1, "Atlassian (TEAM) — Cost of Equity: CAPM vs. Fama-French Models", title_font, title_fill, align="center")
ws.row_dimensions[1].height = 26

ws.merge_cells("A2:H2")
cell(2, 1, "Full regression outputs, premiums, and headline comparison", subtitle_font, align="center")

row = 4

# ---------- Headline comparison ----------
ws.merge_cells(f"A{row}:E{row}")
cell(row, 1, "Headline Comparison", header_font, header_fill, align="center")
row += 1
for i, h in enumerate(["Model", "Cost of Equity", "R²", "vs. CAPM (bps)", "Reliability"]):
    cell(row, 1+i, h, header_font, header_fill, align="center")
row += 1

headline = [
    ("CAPM (S&P 500 Beta)", 0.1084, 0.1358, 0, "Reliable"),
    ("Fama-French 3-Factor", 0.0966, 0.175, (0.0966-0.1084)*10000, "Reliable"),
    ("Fama-French 5-Factor", -0.0191, 0.325, (-0.0191-0.1084)*10000, "Unreliable — see note"),
]
for i, (name, ke, rsq, diff, rel) in enumerate(headline):
    bad = "Unreliable" in rel
    fill = warn_fill if bad else (summary_fill if i == 0 else alt_fill)
    cell(row, 1, name, result_font if i == 0 else label_font, fill, align="left", indent=1)
    cell(row, 2, ke, warn_font if bad else result_font, fill, numfmt="0.00%", align="center")
    cell(row, 3, rsq, label_font, fill, numfmt="0.000", align="center")
    cell(row, 4, diff, label_font, fill, numfmt="+0;-0", align="center")
    cell(row, 5, rel, warn_font if bad else label_font, fill, align="center")
    row += 1
row += 2

# ---------- CAPM ----------
ws.merge_cells(f"A{row}:E{row}")
cell(row, 1, "CAPM Inputs", header_font, header_fill, align="center")
row += 1
capm_rf_row = row
cell(row, 1, "Risk-Free Rate (Rf)", label_font, alt_fill, align="left", indent=1)
cell(row, 2, 0.0467, label_font, alt_fill, numfmt="0.00%", align="center")
row += 1
capm_mrp_row = row
cell(row, 1, "Market Risk Premium (MRP)", label_font, None, align="left", indent=1)
cell(row, 2, 0.0446, label_font, None, numfmt="0.00%", align="center")
row += 1
capm_beta_row = row
cell(row, 1, "Beta (own regression vs. S&P 500)", label_font, alt_fill, align="left", indent=1)
cell(row, 2, 1.3828, label_font, alt_fill, numfmt="0.0000", align="center")
row += 1
capm_ke_row = row
cell(row, 1, "Cost of Equity (CAPM)", result_font, summary_fill, align="left", indent=1)
cell(row, 2, f"=B{capm_rf_row}+B{capm_beta_row}*B{capm_mrp_row}", result_font, summary_fill, numfmt="0.00%", align="center")
row += 2

# ---------- FF3 regression (LINEST) ----------
ws.merge_cells(f"A{row}:H{row}")
cell(row, 1, "Fama-French 3-Factor — Live Regression (LINEST)", header_font, header_fill, align="center")
row += 1

n3 = len(merged_3f_clean)
k3 = 3
factor_cols_3 = ["Mkt-RF", "SMB", "HML"]

# Reserve rows for stats block, then place data table right after
stats_rows_3 = 2*k3 + 3  # intercept + k coefs + stderr-intercept + k stderrs + Rsq
data_header_3 = row + stats_rows_3 + 3
data_start_3 = data_header_3 + 1
data_end_3 = data_start_3 + n3 - 1

x_range_3 = f"E{data_start_3}:G{data_end_3}"
y_range_3 = f"D{data_start_3}:D{data_end_3}"

r = row
cell(r, 1, "Intercept (alpha)", label_font, summary_fill, align="left", indent=1)
cell(r, 2, f"=INDEX(LINEST({y_range_3},{x_range_3},TRUE,TRUE),1,{k3+1})", result_font, summary_fill, numfmt="0.0000", align="center")
r += 1
ff3_coef_rows = {}
for i, f in enumerate(factor_cols_3):
    idx = k3 - i
    cell(r, 1, f"{f} (coefficient)", label_font, align="left", indent=1)
    cell(r, 2, f"=INDEX(LINEST({y_range_3},{x_range_3},TRUE,TRUE),1,{idx})", result_font, numfmt="0.0000", align="center")
    ff3_coef_rows[f] = r
    r += 1
cell(r, 1, "Std Error (Intercept)", label_font, align="left", indent=1)
cell(r, 2, f"=INDEX(LINEST({y_range_3},{x_range_3},TRUE,TRUE),2,{k3+1})", label_font, numfmt="0.0000", align="center")
r += 1
for i, f in enumerate(factor_cols_3):
    idx = k3 - i
    cell(r, 1, f"Std Error ({f})", label_font, align="left", indent=1)
    cell(r, 2, f"=INDEX(LINEST({y_range_3},{x_range_3},TRUE,TRUE),2,{idx})", label_font, numfmt="0.0000", align="center")
    r += 1
cell(r, 1, "R-Squared", result_font, summary_fill, align="left", indent=1)
cell(r, 2, f"=INDEX(LINEST({y_range_3},{x_range_3},TRUE,TRUE),3,1)", result_font, summary_fill, numfmt="0.0000", align="center")
r += 2

# FF3 Premiums + Ke
cell(r, 1, "Fama-French 3-Factor — Premiums & Cost of Equity", header_font, header_fill, align="center")
ws.merge_cells(f"A{r}:E{r}")
r += 1
ff3_mrp_row = r
cell(r, 1, "Market Risk Premium (MRP)", label_font, alt_fill, align="left", indent=1)
cell(r, 2, f"=B{capm_mrp_row}", label_font, alt_fill, numfmt="0.00%", align="center")
r += 1
ff3_smb_prem_row = r
cell(r, 1, "SMB Premium (annualized, ×52)", label_font, align="left", indent=1)
cell(r, 2, 0.013976, label_font, numfmt="0.00%", align="center")
r += 1
ff3_hml_prem_row = r
cell(r, 1, "HML Premium (annualized, ×52)", label_font, alt_fill, align="left", indent=1)
cell(r, 2, 0.040145, label_font, alt_fill, numfmt="0.00%", align="center")
r += 1
ff3_ke_row = r
ke3_formula = (f"=B{capm_rf_row}+B{ff3_coef_rows['Mkt-RF']}*B{ff3_mrp_row}"
               f"+B{ff3_coef_rows['SMB']}*B{ff3_smb_prem_row}"
               f"+B{ff3_coef_rows['HML']}*B{ff3_hml_prem_row}")
cell(r, 1, "Cost of Equity (FF3)", result_font, summary_fill, align="left", indent=1)
cell(r, 2, ke3_formula, result_font, summary_fill, numfmt="0.00%", align="center")
r += 2

row_after_ff3_summary = r

# FF3 raw data table
for i, h in enumerate(["Date", "TEAM Return", "RF", "TEAM Excess Return"] + factor_cols_3):
    cell(data_header_3, 1+i, h, header_font, header_fill, align="center")
for i, rd in merged_3f_clean.reset_index(drop=True).iterrows():
    rr = data_start_3 + i
    vals = [pd.to_datetime(rd['Date']).date(), rd['TEAM Return'], rd['RF'], rd['TEAM_excess']] + [rd[f] for f in factor_cols_3]
    for j, v in enumerate(vals):
        c = ws.cell(row=rr, column=1+j, value=v)
        c.border = border
        c.font = Font(name="Arial", size=9)
        c.number_format = "mm/dd/yyyy" if j == 0 else "0.0000"
        if i % 2 == 0:
            c.fill = alt_fill

row = data_end_3 + 3

# ---------- FF5 regression (LINEST) ----------
ws.merge_cells(f"A{row}:H{row}")
cell(row, 1, "Fama-French 5-Factor — Live Regression (LINEST)", header_font, header_fill, align="center")
row += 1

n5 = len(merged_5f_clean)
k5 = 5
factor_cols_5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

stats_rows_5 = 2*k5 + 3
data_header_5 = row + stats_rows_5 + 3
data_start_5 = data_header_5 + 1
data_end_5 = data_start_5 + n5 - 1

x_range_5 = f"E{data_start_5}:I{data_end_5}"
y_range_5 = f"D{data_start_5}:D{data_end_5}"

r = row
cell(r, 1, "Intercept (alpha)", label_font, summary_fill, align="left", indent=1)
cell(r, 2, f"=INDEX(LINEST({y_range_5},{x_range_5},TRUE,TRUE),1,{k5+1})", result_font, summary_fill, numfmt="0.0000", align="center")
r += 1
ff5_coef_rows = {}
for i, f in enumerate(factor_cols_5):
    idx = k5 - i
    extreme = f in ("RMW", "CMA")
    cell(r, 1, f"{f} (coefficient)", warn_font if extreme else label_font, warn_fill if extreme else None, align="left", indent=1)
    cell(r, 2, f"=INDEX(LINEST({y_range_5},{x_range_5},TRUE,TRUE),1,{idx})", warn_font if extreme else result_font, warn_fill if extreme else None, numfmt="0.0000", align="center")
    ff5_coef_rows[f] = r
    r += 1
cell(r, 1, "Std Error (Intercept)", label_font, align="left", indent=1)
cell(r, 2, f"=INDEX(LINEST({y_range_5},{x_range_5},TRUE,TRUE),2,{k5+1})", label_font, numfmt="0.0000", align="center")
r += 1
for i, f in enumerate(factor_cols_5):
    idx = k5 - i
    cell(r, 1, f"Std Error ({f})", label_font, align="left", indent=1)
    cell(r, 2, f"=INDEX(LINEST({y_range_5},{x_range_5},TRUE,TRUE),2,{idx})", label_font, numfmt="0.0000", align="center")
    r += 1
cell(r, 1, "R-Squared", result_font, summary_fill, align="left", indent=1)
cell(r, 2, f"=INDEX(LINEST({y_range_5},{x_range_5},TRUE,TRUE),3,1)", result_font, summary_fill, numfmt="0.0000", align="center")
r += 2

# FF5 Premiums + Ke
cell(r, 1, "Fama-French 5-Factor — Premiums & Cost of Equity", header_font, header_fill, align="center")
ws.merge_cells(f"A{r}:E{r}")
r += 1
ff5_mrp_row = r
cell(r, 1, "Market Risk Premium (MRP)", label_font, alt_fill, align="left", indent=1)
cell(r, 2, f"=B{capm_mrp_row}", label_font, alt_fill, numfmt="0.00%", align="center")
r += 1
ff5_prem_rows = {}
premiums_5 = {"SMB": 0.015095, "HML": 0.037097, "RMW": 0.030199, "CMA": 0.029486}
for i, (fname, val) in enumerate(premiums_5.items()):
    cell(r, 1, f"{fname} Premium (annualized, ×252)", label_font, alt_fill if i % 2 == 0 else None, align="left", indent=1)
    cell(r, 2, val, label_font, alt_fill if i % 2 == 0 else None, numfmt="0.00%", align="center")
    ff5_prem_rows[fname] = r
    r += 1
ff5_ke_row = r
ke5_formula = (f"=B{capm_rf_row}+B{ff5_coef_rows['Mkt-RF']}*B{ff5_mrp_row}"
               f"+B{ff5_coef_rows['SMB']}*B{ff5_prem_rows['SMB']}"
               f"+B{ff5_coef_rows['HML']}*B{ff5_prem_rows['HML']}"
               f"+B{ff5_coef_rows['RMW']}*B{ff5_prem_rows['RMW']}"
               f"+B{ff5_coef_rows['CMA']}*B{ff5_prem_rows['CMA']}")
cell(r, 1, "Cost of Equity (FF5)", warn_font, warn_fill, align="left", indent=1)
cell(r, 2, ke5_formula, warn_font, warn_fill, numfmt="0.00%", align="center")
r += 2

ws.merge_cells(f"A{r}:H{r+3}")
note_text = ("Note: The 5-Factor model produces an economically implausible negative cost of equity, driven by an "
             "extreme RMW coefficient combined with a positive RMW premium — likely multicollinearity among "
             "profitability/investment factors relative to sample size (n=149). Reported for transparency; not "
             "used in the primary WACC/DCF. CAPM and FF3 are broadly consistent and support the WACC assumptions used.")
c = ws.cell(row=r, column=1, value=note_text)
c.font = Font(name="Arial", size=9, italic=True)
c.alignment = Alignment(wrap_text=True, vertical="top", indent=1)
r += 5

# FF5 raw data table
stats_rows_5 = 2*k5 + 3  # intercept + k coefs + stderr-intercept + k stderrs + Rsq
data_header_5 = row + stats_rows_5 + 20   # ← bumped from +3 to +20, a safe buffer
data_start_5 = data_header_5 + 1
data_end_5 = data_start_5 + n5 - 1
for i, h in enumerate(["Date", "TEAM Return", "RF", "TEAM Excess Return"] + factor_cols_5):
    cell(data_header_5, 1+i, h, header_font, header_fill, align="center")
for i, rd in merged_5f_clean.reset_index(drop=True).iterrows():
    rr = data_start_5 + i
    vals = [pd.to_datetime(rd['Date']).date(), rd['TEAM Return'], rd['RF'], rd['TEAM_excess']] + [rd[f] for f in factor_cols_5]
    for j, v in enumerate(vals):
        c = ws.cell(row=rr, column=1+j, value=v)
        c.border = border
        c.font = Font(name="Arial", size=9)
        c.number_format = "mm/dd/yyyy" if j == 0 else "0.0000"
        if i % 2 == 0:
            c.fill = alt_fill

# Column widths
widths = {"A": 32, "B": 15, "C": 13, "D": 15, "E": 13, "F": 13, "G": 13, "H": 13, "I": 13}
for col, w in widths.items():
    ws.column_dimensions[col].width = w

ws.sheet_view.showGridLines = False

wb.save("TEAM_CostOfEquity_Full.xlsx")
print("Saved successfully")







