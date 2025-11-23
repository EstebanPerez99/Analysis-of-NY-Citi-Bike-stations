"""
Time Series Validation Script
Run this to get validation results before adding to notebook
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Load the demand_hourly dataset (you need to have this from your notebook)
print("=" * 80)
print("TIME SERIES VALIDATION SCRIPT")
print("=" * 80)
print("\n⚠️  NOTE: This script assumes you have 'demand_hourly' saved.")
print("   Run this in your notebook after creating all features:\n")
print("   demand_hourly.to_csv('data/demand_hourly_temp.csv', index=False)")
print("\n" + "=" * 80)

try:
    demand_hourly = pd.read_csv('data/demand_hourly_temp.csv', parse_dates=['date'])
    print(f"\n✅ Loaded demand_hourly: {demand_hourly.shape}")
except FileNotFoundError:
    print("\n❌ ERROR: Could not find 'data/demand_hourly_temp.csv'")
    print("   Please save your demand_hourly dataframe first:")
    print("   demand_hourly.to_csv('data/demand_hourly_temp.csv', index=False)")
    exit(1)

# ============================================================================
# 1. STATIONARITY CHECK
# ============================================================================
print("\n" + "=" * 80)
print("1. STATIONARITY CHECK (ADF TEST)")
print("=" * 80)

# Test for pickups
pickups_series = demand_hourly.groupby('date')['pickups'].sum()
result_pickups = adfuller(pickups_series)

print("\n📊 Augmented Dickey-Fuller Test for PICKUPS:")
print(f"   ADF Statistic: {result_pickups[0]:.6f}")
print(f"   p-value: {result_pickups[1]:.6f}")
print(f"   Critical Values:")
for key, value in result_pickups[4].items():
    print(f"      {key}: {value:.3f}")

if result_pickups[1] < 0.05:
    print(f"   ✅ RESULT: Series is STATIONARY (p-value < 0.05)")
    print(f"      → No differencing needed (d=0 in ARIMA)")
else:
    print(f"   ⚠️  RESULT: Series is NON-STATIONARY (p-value >= 0.05)")
    print(f"      → May need differencing (d=1 in ARIMA)")

# Test for dropoffs
dropoffs_series = demand_hourly.groupby('date')['dropoffs'].sum()
result_dropoffs = adfuller(dropoffs_series)

print("\n📊 Augmented Dickey-Fuller Test for DROPOFFS:")
print(f"   ADF Statistic: {result_dropoffs[0]:.6f}")
print(f"   p-value: {result_dropoffs[1]:.6f}")
print(f"   Critical Values:")
for key, value in result_dropoffs[4].items():
    print(f"      {key}: {value:.3f}")

if result_dropoffs[1] < 0.05:
    print(f"   ✅ RESULT: Series is STATIONARY (p-value < 0.05)")
else:
    print(f"   ⚠️  RESULT: Series is NON-STATIONARY (p-value >= 0.05)")

# ============================================================================
# 2. FEATURE CLEANUP
# ============================================================================
print("\n" + "=" * 80)
print("2. FEATURE CLEANUP")
print("=" * 80)

features_to_remove = [
    'season',  # String version, we have dummies
    'pickups_percentile',  # Intermediate calculation
    'dropoffs_percentile',  # Intermediate calculation
    'cumulative_net_flow',  # Daily-specific, resets each day
    'daily_net_flow',  # Future information leakage
]

print(f"\n🗑️  Features to remove ({len(features_to_remove)}):")
for feat in features_to_remove:
    if feat in demand_hourly.columns:
        print(f"   ✓ {feat}")
    else:
        print(f"   ✗ {feat} (not found)")

# ============================================================================
# 3. MULTICOLLINEARITY CHECK
# ============================================================================
print("\n" + "=" * 80)
print("3. MULTICOLLINEARITY CHECK")
print("=" * 80)

continuous_cols = [col for col in demand_hourly.columns if demand_hourly[col].dtype in ['float64', 'int64']]
continuous_cols = [col for col in continuous_cols if col not in ['pickups', 'dropoffs', 'cluster']]

print(f"\nChecking {len(continuous_cols)} continuous features...")

corr_matrix = demand_hourly[continuous_cols].corr().abs()

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print(f"\n⚠️  Found {len(high_corr_pairs)} highly correlated pairs (>0.95):")
    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"   {feat1} <-> {feat2}: {corr_val:.3f}")
else:
    print(f"\n✅ No highly correlated features found (all < 0.95)")

# ============================================================================
# 4. DATA QUALITY CHECK
# ============================================================================
print("\n" + "=" * 80)
print("4. DATA QUALITY CHECK")
print("=" * 80)

# Missing values
print("\n📊 Missing Values:")
missing_counts = demand_hourly.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]
if len(missing_counts) > 0:
    for col, count in missing_counts.items():
        pct = (count / len(demand_hourly)) * 100
        print(f"   {col}: {count:,} ({pct:.2f}%)")
else:
    print(f"   ✅ No missing values")

# Target statistics
print("\n📊 Target Variable Statistics:")
print(f"\nPickups:")
print(f"   Mean: {demand_hourly['pickups'].mean():.2f}")
print(f"   Median: {demand_hourly['pickups'].median():.2f}")
print(f"   Std: {demand_hourly['pickups'].std():.2f}")
print(f"   Min: {demand_hourly['pickups'].min():.0f}, Max: {demand_hourly['pickups'].max():.0f}")
print(f"   Zeros: {(demand_hourly['pickups'] == 0).sum():,} ({(demand_hourly['pickups'] == 0).mean():.1%})")

print(f"\nDropoffs:")
print(f"   Mean: {demand_hourly['dropoffs'].mean():.2f}")
print(f"   Median: {demand_hourly['dropoffs'].median():.2f}")
print(f"   Std: {demand_hourly['dropoffs'].std():.2f}")
print(f"   Min: {demand_hourly['dropoffs'].min():.0f}, Max: {demand_hourly['dropoffs'].max():.0f}")
print(f"   Zeros: {(demand_hourly['dropoffs'] == 0).sum():,} ({(demand_hourly['dropoffs'] == 0).mean():.1%})")

# Dataset summary
print("\n📊 Final Dataset Summary:")
print(f"   Shape: {demand_hourly.shape}")
print(f"   Date range: {demand_hourly['date'].min()} to {demand_hourly['date'].max()}")
print(f"   Total days: {demand_hourly['date'].nunique()}")
print(f"   Total clusters: {demand_hourly['cluster'].nunique()}")

print("\n" + "=" * 80)
print("✅ VALIDATION COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Review the results above")
print("2. Copy the code cells from 'time_series_validation_code.md' to your notebook")
print("3. Run them in your notebook to generate plots")
print("4. Proceed with train/test split")

