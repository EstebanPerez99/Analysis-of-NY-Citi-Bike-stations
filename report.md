# Citi Bike NYC: Demand Forecasting and Fleet Repositioning

**Course:** Data Sciences  
**Authors:** [Team Members]  
**Date:** December 2024

---

## 1. Introduction

Shared mobility services have emerged as a sustainable and efficient transportation option in urban environments. As cities worldwide grapple with climate change, traffic congestion, and the need for accessible transportation alternatives, bike-sharing systems offer a compelling solution that combines environmental benefits with practical urban mobility. Citi Bike, one of the largest station-based bike-sharing systems in the United States, operates over 900 stations and 14,000 bikes across New York City, serving millions of riders annually.

Effective fleet management is critical to ensure bikes are available where and when users need them. Unlike traditional transportation services, bike-sharing systems face a unique operational challenge: bikes naturally accumulate in certain areas (residential neighborhoods in the morning, business districts in the evening) and become depleted in others. This imbalance creates the need for overnight repositioning—a labor-intensive and costly operation that directly impacts service quality and customer satisfaction.

This project addresses three interconnected challenges using the 2018 Citi Bike trip dataset containing 17.5 million rides: (1) spatial clustering of stations into manageable zones for operational efficiency, (2) hourly demand prediction for the next 24 hours using machine learning, and (3) overnight bike repositioning optimization based on predicted demand. Our approach combines exploratory data analysis, sophisticated feature engineering, and ensemble machine learning methods to develop a practical decision-support system for fleet operators. The ultimate goal is to minimize bike shortages while avoiding unnecessary over-provisioning that ties up capital in idle inventory.

---

## 2. Data Analysis and Visualization

### 2.1 Dataset Overview and Cleaning

The raw dataset provided by Citi Bike contained 17,548,339 trip records spanning the entire calendar year of 2018. Each record included rich information about individual trips: start and end timestamps, origin and destination station identifiers, geographic coordinates (latitude and longitude) for both endpoints, trip duration in seconds, bike identifier, user type (Subscriber or Customer), and demographic information including birth year and gender for registered users.

Before proceeding with analysis, we implemented a comprehensive data cleaning pipeline to ensure data quality and consistency:

**Duration Filtering:** Trips with durations less than 60 seconds were removed as these likely represent false starts or system errors (e.g., user immediately re-docking a bike due to mechanical issues). Similarly, trips exceeding 24 hours were excluded as outliers, as the standard Citi Bike pricing model charges significant fees for extended rentals, making such trips rare and likely erroneous.

**Geographic Outlier Detection:** During initial mapping, we discovered two stations located far outside the expected NYC service area—specifically in Montreal, Canada. These erroneous entries were identified and removed by implementing a bounding box filter constraining valid stations to the NYC metropolitan area (latitude: 40.68°-40.85°N, longitude: 73.85°-74.05°W). This geographic validation step was crucial for subsequent spatial clustering.

**Temporal Consistency:** Some trips that started on December 31, 2018, ended on January 1, 2019. To maintain a clean 2018-only dataset for consistent analysis, we filtered to ensure both start and end times fell within the 2018 calendar year.

**Missing Value Treatment:** Station coordinates with null values were dropped, and demographic fields (birth year, gender) with missing values were handled appropriately during feature engineering.

After cleaning, 17,406,891 valid trips across 846 unique stations remained, representing a 99.2% retention rate—indicating high overall data quality.

> **Figure 1:** Interactive map of all 846 Citi Bike stations in NYC, with marker size proportional to total trip activity. The map reveals the concentration of high-activity stations in Manhattan, particularly in Midtown and the Financial District.

### 2.2 Temporal Patterns and Seasonality

Understanding temporal demand patterns is fundamental to accurate forecasting. Our exploratory analysis revealed multiple overlapping cycles in bike demand:

**Annual Seasonality:** Monthly aggregation of total trips revealed dramatic seasonal variation. Summer months (June through September) experienced approximately 2.5x higher demand than winter months (December through February). June 2018 recorded the highest monthly volume with over 1.8 million trips, while February saw the lowest at approximately 720,000 trips. This pattern aligns with expected weather-dependent behavior—pleasant temperatures encourage cycling while cold, wet, or snowy conditions suppress demand. The seasonality insight has direct operational implications: fleet sizing and maintenance schedules should account for predictable summer surges.

> **Figure 2:** Monthly pickups and dropoffs throughout 2018, showing the pronounced summer peak (June-September) and winter trough (December-February). The near-perfect alignment between pickups and dropoffs at the monthly level confirms system-wide balance.

**Weekly and Daily Patterns:** Analysis of hourly demand by day of week revealed two distinct usage profiles:

*Weekday (Monday-Friday) Pattern:* Characterized by sharp bimodal peaks corresponding to commuter behavior. The morning rush (7:00-9:00 AM) shows rapid demand increase as workers travel to offices, followed by relatively flat midday usage, then an even more pronounced evening rush (5:00-7:00 PM) as commuters return home. This pattern was remarkably consistent across all weekdays, with only minor variations (Monday slightly lower, Friday slightly higher evening peaks).

*Weekend (Saturday-Sunday) Pattern:* Displayed a fundamentally different unimodal distribution. Demand gradually increases from morning through early afternoon, peaking between 1:00-4:00 PM, then slowly declining into evening. This recreational usage pattern shows lower overall volume but more dispersed timing, suggesting leisure trips to parks, restaurants, and social activities rather than time-constrained commuting.

> **Figure 3:** Hourly demand patterns comparing weekdays vs. weekends across all clusters. The weekday bimodal commuter pattern (peaks at 8 AM and 6 PM) contrasts sharply with the weekend unimodal recreational pattern (peak at 2 PM).

**Holiday Effects:** We extended our dataset with NYC public holidays (New Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas) and major events. Major holidays showed reduced overall demand, particularly in business-district clusters, as commuter traffic disappeared. However, recreational clusters near Central Park and waterfront areas sometimes showed increased weekend-like patterns even on weekday holidays. Independence Day (July 4th) provided a particularly clear example: morning demand was suppressed compared to a typical Wednesday, but afternoon/evening demand spiked as users traveled to firework viewing locations.

> **Figure 4:** Demand comparison for Independence Day (July 4th, 2018) vs. a typical Wednesday, showing the flattening of commuter peaks and shift toward recreational afternoon usage.

### 2.3 Station and Cluster-Level Analysis

Individual station analysis revealed extreme heterogeneity in usage patterns. The busiest station (Pershing Square North, near Grand Central Terminal) recorded over 300,000 annual trips, while the quietest stations saw fewer than 1,000. This 300x variation in activity levels underscores the challenge of station-level forecasting and motivated our clustering approach.

After applying K-Means spatial clustering (detailed in Section 3.1), we analyzed demand balance across the 30 resulting clusters. **Net flow analysis**—calculating the difference between pickups (departures) and dropoffs (arrivals)—revealed systematic imbalances:

- **Source Clusters:** Residential areas in Brooklyn, Upper Manhattan, and Queens consistently showed positive net flow (more departures than arrivals), particularly during morning hours. These clusters "export" bikes to employment centers.

- **Sink Clusters:** Business districts in Midtown Manhattan and the Financial District showed negative net flow, accumulating bikes throughout the workday. These clusters receive morning commuters and release evening commuters.

- **Balanced Clusters:** Some mixed-use neighborhoods (East Village, Chelsea) showed relatively balanced daily flows, though hourly patterns still fluctuated significantly.

This source-sink dynamic creates the fundamental operational challenge: without intervention, source clusters would be depleted by midday while sink clusters overflow. Overnight repositioning trucks must reverse these flows to prepare for the next day's commute.

> **Figure 5:** Cluster statistics dashboard showing (a) number of stations per cluster, (b) total annual trips per cluster, (c) pickup vs. dropoff scatter plot with balance line, and (d) geographic distribution of net flow (sources in blue, sinks in red).

### 2.4 Feature Engineering

Based on our exploratory analysis, we engineered a comprehensive feature set to capture the patterns discovered. Each feature was designed with operational validity in mind—meaning it would be known or computable at prediction time (end of day D) when forecasting for day D+1.

**Temporal Features:**
- `hour`: Hour of day (0-23) captures within-day demand variation
- `month`: Month of year (1-12) captures annual seasonality
- `day_of_week`: Day of week (0-6, Monday=0) captures weekly patterns
- `is_weekend`: Binary indicator (1 if Saturday/Sunday, 0 otherwise)

**Seasonal Indicators:**
- `season_spring`, `season_summer`, `season_fall`, `season_winter`: One-hot encoded season dummies derived from month. Spring = Mar-May, Summer = Jun-Aug, Fall = Sep-Nov, Winter = Dec-Feb.

**External Calendar Features:**
- `is_holiday`: Binary indicator for NYC public holidays (10 days in 2018)
- `is_special_event`: Binary indicator for major events (marathons, parades, concerts)

**Rush Hour Indicators:**
- `is_morning_rush`: 1 if hour between 7-9 AM
- `is_evening_rush`: 1 if hour between 5-7 PM
- `weekday_morning_rush`: Interaction of weekday × morning rush
- `weekday_evening_rush`: Interaction of weekday × evening rush
- `is_weekend_rush`: Weekend high-demand hours (10 AM - 7 PM on weekends)

**Lag Features (Critical for Time Series):**
- `pickups_lag_24h`: Pickups in same cluster, same hour, previous day
- `pickups_lag_168h`: Pickups in same cluster, same hour, previous week (168 hours)
- `pickups_rolling_24h`: Rolling 24-hour average of pickups for the cluster
- Equivalent features for dropoffs (`dropoffs_lag_24h`, `dropoffs_lag_168h`, `dropoffs_rolling_24h`)

**Cyclic Encoding:**
- `hour_sin`, `hour_cos`: Sine and cosine transformation of hour to capture circular nature of time (hour 23 is close to hour 0). Computed as: `hour_sin = sin(2π × hour/24)`, `hour_cos = cos(2π × hour/24)`.

**Cluster Identity:**
- One-hot encoded `cluster_0` through `cluster_29`: 30 binary columns indicating cluster membership, allowing the model to learn cluster-specific baseline demand levels.

| Category | Features | Count |
|----------|----------|-------|
| Temporal | hour, month, day_of_week, is_weekend | 4 |
| Seasonal | season_spring, season_summer, season_fall, season_winter | 4 |
| External | is_holiday, is_special_event | 2 |
| Rush Hours | is_morning_rush, is_evening_rush, weekday_morning_rush, weekday_evening_rush, is_weekend_rush | 5 |
| Lag Features | pickups_lag_24h, pickups_lag_168h, pickups_rolling_24h, dropoffs_lag_* | 6 |
| Cyclic | hour_sin, hour_cos | 2 |
| Cluster Identity | cluster_0 through cluster_29 | 30 |
| **Total** | | **53** |

---

## 3. Prediction Challenge

### 3.1 Spatial Clustering (Task 1)

The first task required grouping stations into spatially coherent clusters. We applied **K-Means clustering** using station latitude and longitude as features. The algorithm was configured with k=30 clusters, balancing several considerations:

- **Minimum threshold:** The project specification required at least 20 clusters
- **Operational practicality:** Too many clusters would create excessive complexity for repositioning logistics
- **Statistical reliability:** Each cluster needed sufficient trip volume for robust pattern estimation
- **Geographic coherence:** Clusters should represent contiguous neighborhoods

The resulting clusters ranged from 27 to 63 stations each, with an average of 28 stations per cluster. Visual inspection confirmed geographic coherence—clusters naturally corresponded to recognizable NYC neighborhoods (Financial District, Midtown East, Chelsea, Williamsburg, etc.).

> **Figure 6:** Interactive map of NYC with all 846 stations colored by cluster assignment. Cluster boundaries generally align with neighborhood boundaries, validating the spatial coherence of the K-Means solution.

> **Figure 7:** Bar chart showing stations per cluster distribution, with horizontal line indicating mean (28 stations). The distribution shows reasonable balance across clusters.

Each cluster was then treated as a single entity for demand forecasting, with hourly pickups and dropoffs aggregated across all member stations. This aggregation reduced the prediction problem from 846 individual time series to 30 cluster-level time series, dramatically improving statistical power while maintaining operational relevance.

### 3.2 Data Preparation for Time Series Modeling

The aggregated dataset contained cluster-hour observations spanning the full year. After creating lag features (which require historical data), the final dataset contained approximately 250,000 usable observations.

**Chronological Train/Validation/Test Split:**

Unlike standard machine learning problems where random shuffling is acceptable, time series forecasting requires strict chronological ordering to prevent data leakage. We implemented the following split:

- **Training Set:** January 1 - August 31, 2018 (60% of data)
- **Validation Set:** September 1 - October 31, 2018 (10% of data)
- **Test Set:** November 1 - December 31, 2018 (30% of data)

This split ensures that model selection and hyperparameter tuning (using validation set) never sees future data, and final performance evaluation (test set) simulates true out-of-sample prediction during the winter season.

**Target Variables:**
- `pickups`: Number of bike pickups (departures) per cluster-hour
- `dropoffs`: Number of bike dropoffs (arrivals) per cluster-hour

We trained separate models for pickups and dropoffs, as they may have different predictive relationships with features (e.g., morning rush primarily affects pickups in residential clusters).

### 3.3 Model Selection and Comparison

We evaluated multiple model families, progressing from simple baselines to more sophisticated approaches:

**Baseline Models (Linear):**

Linear Regression, Ridge Regression, and Lasso Regression were implemented as interpretable baselines. All three achieved R² ≈ 0.87 on the validation set, explaining 87% of demand variance. The similarity across regularization approaches indicated minimal multicollinearity issues and suggested that the relationship between features and demand contains significant non-linear components not captured by linear models.

**Neural Networks (MLPRegressor):**

We tested several Multi-Layer Perceptron architectures using scikit-learn's MLPRegressor:

| Architecture | Hidden Layers | Pickups R² | Dropoffs R² | Training Time |
|--------------|---------------|------------|-------------|---------------|
| Deep & Wide | (256, 128, 64, 32) | 0.900 | 0.907 | 41s |
| Wider Layers | (512, 256, 128) | 0.899 | 0.910 | 153s |
| Simple & Deep | (100, 100, 100, 100) | 0.891 | 0.905 | 86s |

Neural networks achieved R² ≈ 0.90, improving over linear baselines but requiring careful hyperparameter tuning and significantly longer training times. The "Wider Layers" configuration performed best for dropoffs.

**Ensemble Methods (Random Forest):**

Random Forest with 200 decision trees was evaluated as our primary ensemble method. Tree-based models have several advantages for tabular data: they naturally capture non-linear relationships and feature interactions, require no feature scaling, and are robust to outliers.

| Model | Pickups R² | Dropoffs R² | Training Time |
|-------|------------|-------------|---------------|
| **Random Forest (n=200)** | **0.898** | **0.904** | 30s |

**Data Leakage Discovery and Resolution:**

During early experimentation, we achieved suspiciously high R² scores (0.94+) with Random Forest. Investigation revealed **data leakage**: we had inadvertently included `net_flow` (pickups minus dropoffs) as a feature. Since net_flow is directly derived from the target variables, the model was essentially "cheating" by accessing information that would not be available at prediction time.

After removing the leaked feature and re-running experiments, R² normalized to approximately 0.90 across all models. This experience underscores the importance of careful feature validation in time series contexts.

**Final Model Selection:**

Random Forest (n=200 trees) was selected as the final model based on:
1. Competitive R² scores (0.90) matching neural networks
2. Lowest Mean Absolute Error among comparable models
3. Fast training time (~30 seconds vs. 90+ seconds for neural networks)
4. No feature scaling requirements
5. Robust handling of outliers and anomalies
6. Interpretable feature importance rankings

### 3.4 Prediction Results (Task 2)

The final Random Forest model was retrained on the combined training + validation set (January-October 2018) and evaluated on the held-out test set (November-December 2018).

**Test Set Performance:**

| Target | MAE (bikes/hour) | RMSE (bikes/hour) | R² |
|--------|------------------|-------------------|-----|
| Pickups | 24.86 | 51.72 | 0.897 |
| Dropoffs | 24.84 | 51.81 | 0.900 |

**Interpretation:**
- **R² ≈ 0.90:** The model explains 90% of hourly demand variance, indicating strong predictive power
- **MAE ≈ 25 bikes/hour:** On average, predictions deviate by 25 bikes per cluster per hour. For a cluster averaging 100 pickups/hour, this represents 25% error; for a cluster averaging 500 pickups/hour, this represents only 5% error.
- **Symmetric performance:** Pickups and dropoffs are equally predictable, suggesting similar underlying patterns

> **Figure 8:** Scatter plots of predicted vs. actual demand for (a) pickups and (b) dropoffs on the test set. Points cluster tightly around the 45-degree line, confirming strong correlation. Some underprediction is visible for very high demand hours.

> **Figure 9:** Hourly predictions for a representative weekday (Wednesday, December 12) and weekend day (Saturday, December 15) in a high-demand cluster. The model successfully captures rush hour peaks on Wednesday and the shifted recreational pattern on Saturday.

**Error Analysis:**

Prediction errors were not uniformly distributed:
- **Higher errors on atypical days:** Holidays and days with unusual weather showed larger prediction errors, as the model had limited training examples of such events
- **Higher errors in high-variance clusters:** Tourist-heavy and event-adjacent clusters showed more volatile demand that was harder to predict
- **Lower errors during stable periods:** Regular weekday patterns were predicted with high accuracy

### 3.5 Bike Repositioning Strategy (Task 3)

The ultimate operational goal is translating demand predictions into actionable repositioning decisions. We developed an algorithm to calculate the optimal number of bikes to place at each cluster at the start of each day, using **cumulative flow analysis**.

**Methodology:**

Rather than simply comparing total daily pickups vs. dropoffs, we analyze the hourly evolution of bike inventory throughout the day:

1. **Hourly Net Flow:** For each hour h, calculate: `net_flow(h) = predicted_pickups(h) - predicted_dropoffs(h)`
   - Positive net flow = bikes leaving the cluster (potential shortage)
   - Negative net flow = bikes arriving (potential surplus)

2. **Cumulative Net Flow:** Compute running sum from hour 0: `cumulative_flow(h) = Σ net_flow(0:h)`
   - This represents the net change in cluster inventory since day start

3. **Maximum Deficit:** Identify the worst-case shortage point: `max_deficit = max(0, max(cumulative_flow))`
   - This is the minimum number of bikes needed to avoid any shortage during the day

4. **Safety Margin:** Add buffer to account for prediction uncertainty: `bikes_with_margin = max_deficit × (1 + safety_margin)`
   - We used a 20% safety margin based on observed prediction error

5. **Minimum Baseline:** Ensure every cluster has at least some bikes: `bikes_needed = max(bikes_with_margin, min_baseline)`
   - We set a minimum of 5 bikes per cluster

**Formula:**
```
bikes_needed = max(min_baseline, max_deficit × (1 + safety_margin))
```

> **Figure 10:** Cumulative flow analysis visualization for a high-demand cluster on a sample day. The graph shows hourly net flow (bars) and cumulative flow (line). The maximum cumulative value indicates the minimum bikes needed at day start.

**Repositioning Results:**

We validated the repositioning algorithm on all test set days:

> **Figure 11:** Bar chart showing average daily bike requirements by cluster, ranked from highest to lowest. The top 5 clusters (primarily Manhattan business districts) require 60-100 bikes each, while low-demand clusters need only 10-20 bikes.

> **Figure 12:** Scatter plot comparing recommended bike allocation vs. actual bikes needed (based on realized demand) for each cluster-day. Points above the diagonal represent under-allocation (potential shortage); points below represent over-allocation (wasted capacity).

**Performance Metrics:**
- **Service Level:** Approximately 75-80% of cluster-days would have avoided shortages using recommended allocations
- **Over-allocation:** Average excess of 15-20 bikes per cluster, representing operational buffer
- **Critical clusters:** 5-6 high-demand clusters accounted for 40% of total repositioning needs

---

## 4. Conclusions

### 4.1 Key Findings

1. **Seasonality dominates annual patterns:** Summer months see 2-3x higher bike usage than winter. Fleet sizing and maintenance should be seasonally adjusted.

2. **Weekday commuter patterns are highly predictable:** The bimodal rush-hour pattern (8 AM and 6 PM peaks) repeats with remarkable consistency, enabling accurate forecasting.

3. **Weekend recreational patterns differ fundamentally:** Unimodal afternoon peaks require different operational strategies than weekday commuting.

4. **Spatial clustering effectively reduces complexity:** Aggregating 846 stations into 30 clusters maintains operational relevance while enabling robust statistical modeling.

5. **Random Forest outperforms neural networks for this problem:** On structured tabular data with engineered features, ensemble tree methods achieved comparable accuracy with faster training and easier interpretation.

6. **Cumulative flow analysis enables proactive repositioning:** Analyzing hourly inventory evolution, rather than just daily totals, identifies critical shortage points.

### 4.2 Limitations

- **No real-time weather integration:** Temperature, precipitation, and wind significantly impact cycling; incorporating weather forecasts would likely improve predictions.
- **Static cluster definitions:** Clusters were fixed; dynamic clustering based on demand similarity could adapt to changing patterns.
- **Single-day prediction horizon:** Extending to multi-day forecasts would enable better logistics planning.
- **No cost optimization:** Current approach minimizes shortages but doesn't explicitly optimize repositioning truck routes or labor costs.

### 4.3 Future Work

- **Weather data integration:** Incorporate historical and forecast weather data as additional features
- **Real-time prediction updates:** Implement rolling predictions that update throughout the day based on morning observations
- **Station-level forecasting:** For critical high-traffic stations, develop individual models for finer-grained operations
- **Routing optimization:** Extend from "how many bikes" to "which truck routes" for complete operational planning
- **Causal analysis:** Investigate impact of external factors (subway delays, major events) on demand shifts

### 4.4 Practical Implications

The developed system demonstrates feasibility of data-driven fleet management for bike-sharing systems. Key operational recommendations:

1. **Daily prediction pipeline:** Run demand forecasting at 11 PM each day for next-day planning
2. **Cluster-specific safety margins:** Apply higher buffers (30%+) to high-variance tourist/event clusters
3. **Seasonal fleet adjustment:** Reduce active fleet by 30-40% during winter months
4. **Holiday special handling:** Flag holidays for manual review and adjusted predictions
5. **Continuous model retraining:** Update models quarterly to capture evolving usage patterns

---

## References

1. Citi Bike System Data, Motivate International Inc. (2018). https://citibikenyc.com/system-data
2. NYC Public Holidays Calendar 2018
3. NYC Special Events Database 2018
4. Scikit-learn: Machine Learning in Python, Pedregosa et al. (2011)

---

## Appendix

### A. Alternative Models Explored

See accompanying Jupyter notebook (`project_final.ipynb`) for complete experimental results on baseline models, neural network architectures, and hyperparameter tuning details.

### B. Code Availability

All analysis code, data cleaning pipelines, and trained models are available in the project repository. The main notebook (`project_final.ipynb`) contains reproducible code for all figures and results presented in this report.
