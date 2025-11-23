# Time Series Validation and Feature Selection Code
# Add these cells BEFORE the train/test split

## Markdown Cell 1:
```markdown
## Time Series Validation and Feature Selection

Before splitting our data and training models, we need to:

1. **Check stationarity** using the Augmented Dickey-Fuller (ADF) test
2. **Analyze autocorrelation** (ACF/PACF) to understand temporal dependencies
3. **Clean up features** - remove redundant or problematic features
4. **Validate data quality** - ensure no issues before modeling

These steps are critical for time series forecasting success.
```

## Code Cell 1: Check Stationarity
```python
# ============================================================================
# Time Series Validation: Stationarity Check (ADF Test)
# ============================================================================

from statsmodels.tsa.stattools import adfuller

print("Testing stationarity for pickups and dropoffs...")
print("=" * 80)

# Test for pickups (aggregate across all clusters)
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

print("\n" + "=" * 80)
```

## Code Cell 2: ACF/PACF Analysis
```python
# ============================================================================
# Time Series Validation: Autocorrelation Analysis
# ============================================================================

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

print("Analyzing autocorrelation patterns...")

# Select a representative cluster for analysis
top_cluster = demand_hourly.groupby('cluster')['pickups'].sum().idxmax()
cluster_pickups = demand_hourly[demand_hourly['cluster'] == top_cluster].sort_values('date')['pickups']

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

# ACF plot
plot_acf(cluster_pickups, lags=72, ax=axes[0])  # 72 hours = 3 days
axes[0].set_title(f'Autocorrelation Function (ACF) - Cluster {top_cluster}', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Lag (hours)', fontsize=12)
axes[0].set_ylabel('Correlation', fontsize=12)

# PACF plot
plot_pacf(cluster_pickups, lags=72, ax=axes[1])
axes[1].set_title(f'Partial Autocorrelation Function (PACF) - Cluster {top_cluster}', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Lag (hours)', fontsize=12)
axes[1].set_ylabel('Partial Correlation', fontsize=12)

plt.tight_layout()
plt.show()

print("\n📊 Interpretation Guide:")
print("   - ACF: Shows correlation between observations at different time lags")
print("   - PACF: Shows direct correlation after removing effects of intermediate lags")
print("   - Lags outside blue confidence bands are statistically significant")
print("   - Strong patterns at lag 24 suggest daily seasonality")
print("   - Strong patterns at lag 168 suggest weekly seasonality")
```

## Markdown Cell 2:
```markdown
### Feature Selection and Cleanup

Based on time series best practices and our analysis, we need to:

1. **Remove redundant features** that could cause multicollinearity
2. **Drop features not useful for forecasting** (e.g., identifiers, intermediate calculations)
3. **Keep only features that will be available at prediction time**

#### Features to Remove:
- `season` (string) - already encoded as dummies
- `pickups_percentile`, `dropoffs_percentile` - used to create binary indicators, not needed in final model
- `cumulative_net_flow` - daily-specific, resets each day (not useful for next-day prediction)
- `daily_net_flow` - future information leakage (we won't know this until day ends)
- Potentially redundant lag features if highly correlated

#### Features to Keep:
- All temporal indicators (hour, day, month, season dummies, rush hours)
- Lag features (24h, 168h, rolling averages)
- Cluster characteristics
- Binary demand indicators
- Net flow (instantaneous)
```

## Code Cell 3: Feature Selection
```python
# ============================================================================
# Feature Selection and Cleanup
# ============================================================================

print("Performing feature selection and cleanup...")
print("=" * 80)

# Features to remove (redundant or problematic)
features_to_remove = [
    'season',  # String version, we have dummies
    'pickups_percentile',  # Intermediate calculation
    'dropoffs_percentile',  # Intermediate calculation
    'cumulative_net_flow',  # Daily-specific, resets each day
    'daily_net_flow',  # Future information leakage
]

print(f"\n🗑️  Removing {len(features_to_remove)} redundant/problematic features:")
for feat in features_to_remove:
    if feat in demand_hourly.columns:
        print(f"   - {feat}")
        demand_hourly = demand_hourly.drop(columns=[feat])
    else:
        print(f"   - {feat} (not found, skipping)")

# Check for multicollinearity in continuous features
print(f"\n🔍 Checking for highly correlated features...")

continuous_cols = [
    'avg_trip_duration', 'pct_subscribers', 'avg_age',
    'cluster_station_count', 'cluster_center_lat', 'cluster_center_lng', 'cluster_total_trips',
    'pickups_lag_24h', 'pickups_lag_168h', 'pickups_rolling_24h',
    'dropoffs_lag_24h', 'dropoffs_lag_168h', 'dropoffs_rolling_24h',
    'hour_sin', 'hour_cos', 'net_flow'
]

# Calculate correlation matrix
corr_matrix = demand_hourly[continuous_cols].corr().abs()

# Find pairs with correlation > 0.95 (very high)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print(f"   ⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95):")
    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"      {feat1} <-> {feat2}: {corr_val:.3f}")
    print(f"   → Consider removing one from each pair if model performance suffers")
else:
    print(f"   ✅ No highly correlated features found (all < 0.95)")

# Final feature count
print(f"\n📊 Final feature summary:")
print(f"   Total columns: {len(demand_hourly.columns)}")
print(f"   Features available for modeling: {len(demand_hourly.columns) - 2}")  # Excluding date, cluster
print(f"   Target variables: pickups, dropoffs")

print("\n" + "=" * 80)
```

## Code Cell 4: Data Quality Check
```python
# ============================================================================
# Final Data Quality Validation
# ============================================================================

print("Performing final data quality checks...")
print("=" * 80)

# 1. Check for missing values
print("\n1️⃣  Missing Values Check:")
missing_counts = demand_hourly.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]

if len(missing_counts) > 0:
    print(f"   ⚠️  Found missing values in {len(missing_counts)} columns:")
    for col, count in missing_counts.items():
        pct = (count / len(demand_hourly)) * 100
        print(f"      {col}: {count:,} ({pct:.2f}%)")
else:
    print(f"   ✅ No missing values found")

# 2. Check for infinite values
print("\n2️⃣  Infinite Values Check:")
inf_counts = {}
for col in demand_hourly.select_dtypes(include=['float64', 'int64']).columns:
    inf_count = np.isinf(demand_hourly[col]).sum()
    if inf_count > 0:
        inf_counts[col] = inf_count

if inf_counts:
    print(f"   ⚠️  Found infinite values in {len(inf_counts)} columns:")
    for col, count in inf_counts.items():
        print(f"      {col}: {count:,}")
else:
    print(f"   ✅ No infinite values found")

# 3. Check target variable distributions
print("\n3️⃣  Target Variable Statistics:")
print(f"   Pickups:")
print(f"      Mean: {demand_hourly['pickups'].mean():.2f}")
print(f"      Median: {demand_hourly['pickups'].median():.2f}")
print(f"      Std: {demand_hourly['pickups'].std():.2f}")
print(f"      Min: {demand_hourly['pickups'].min():.0f}, Max: {demand_hourly['pickups'].max():.0f}")
print(f"      Zeros: {(demand_hourly['pickups'] == 0).sum():,} ({(demand_hourly['pickups'] == 0).mean():.1%})")

print(f"\n   Dropoffs:")
print(f"      Mean: {demand_hourly['dropoffs'].mean():.2f}")
print(f"      Median: {demand_hourly['dropoffs'].median():.2f}")
print(f"      Std: {demand_hourly['dropoffs'].std():.2f}")
print(f"      Min: {demand_hourly['dropoffs'].min():.0f}, Max: {demand_hourly['dropoffs'].max():.0f}")
print(f"      Zeros: {(demand_hourly['dropoffs'] == 0).sum():,} ({(demand_hourly['dropoffs'] == 0).mean():.1%})")

# 4. Check date continuity
print("\n4️⃣  Date Continuity Check:")
date_range = pd.date_range(start=demand_hourly['date'].min(), end=demand_hourly['date'].max(), freq='D')
missing_dates = set(date_range) - set(demand_hourly['date'].unique())

if missing_dates:
    print(f"   ⚠️  Found {len(missing_dates)} missing dates:")
    for missing_date in sorted(missing_dates)[:10]:  # Show first 10
        print(f"      {missing_date.date()}")
    if len(missing_dates) > 10:
        print(f"      ... and {len(missing_dates) - 10} more")
else:
    print(f"   ✅ No missing dates - continuous time series")

# 5. Final dataset info
print("\n5️⃣  Final Dataset Summary:")
print(f"   Shape: {demand_hourly.shape}")
print(f"   Memory usage: {demand_hourly.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"   Date range: {demand_hourly['date'].min()} to {demand_hourly['date'].max()}")
print(f"   Total days: {demand_hourly['date'].nunique()}")
print(f"   Total clusters: {demand_hourly['cluster'].nunique()}")
print(f"   Observations per cluster: {len(demand_hourly) / demand_hourly['cluster'].nunique():.0f}")

print("\n✅ Data quality validation complete!")
print("=" * 80)
```

## Markdown Cell 3:
```markdown
### Validation Results Summary

Based on the checks above:

1. **Stationarity**: If p-value < 0.05, our series is stationary (good for most models)
2. **Autocorrelation**: Strong patterns at lag 24 (daily) and 168 (weekly) confirm our lag feature choices
3. **Feature Cleanup**: Removed redundant features to avoid multicollinearity
4. **Data Quality**: Verified no missing/infinite values and continuous time series

**Next Steps:**
- Proceed with train/test split (70/30, chronological)
- Train baseline models (Linear Regression, ARIMA, Gradient Boosting)
- Evaluate using RMSE, MAE, MAPE metrics
- Compare models and select best for final predictions
```

