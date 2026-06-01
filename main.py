"""Seasonality Was a Lie - is the season real, or just a pricing habit?"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

rng = np.random.default_rng(17)
days = pd.date_range("2024-01-01", periods=365)
t = np.arange(365)

# the team discounts every quarter end, which LOOKS like seasonality
discount = ((days.day > 22) & (days.month % 3 == 0)).astype(float) * 0.25
price = 20 * (1 - discount)
demand = 300 - 6 * price + rng.normal(0, 8, 365)

apparent_season = pd.Series(demand, index=days).groupby(days.month).mean()

X = sm.add_constant(price)
model = sm.OLS(demand, X).fit()
resid_season = pd.Series(model.resid, index=days).groupby(days.month).mean()

print(f"Apparent monthly swing: {apparent_season.max() - apparent_season.min():.0f} units")
print(f"After controlling for price: {resid_season.max() - resid_season.min():.0f} units")
print(f"Price coefficient: {model.params[1]:.1f} units per dollar")

os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(9, 5))
plt.plot(apparent_season.index, apparent_season.values, "-o", color="#ff6a3d", label="apparent season")
plt.plot(resid_season.index, resid_season.values + apparent_season.mean(), "--o", color="#999999", label="after price control")
plt.xlabel("month")
plt.ylabel("demand")
plt.legend()
plt.title("The season that was really a pricing habit")
plt.tight_layout()
plt.savefig("outputs/false_seasonality.png", dpi=120)
print("Saved outputs/false_seasonality.png")
