**Last Updated:** October 25, 2025

# Citi Bike NYC 2018 - Demand Forecasting Project

Predictive analytics project for Citi Bike station-level demand forecasting using 17.5M trips from 2018.

## Project Overview

This project analyzes Citi Bike trip data to:

1. **Cluster stations** spatially (minimum 20 clusters)
2. **Forecast demand** for the next 24 hours at cluster level
3. **Optimize bike rebalancing** to minimize shortages

## Quick Start

Required packages (for now):

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Clustering and modeling
- `folium` - Interactive maps
- `matplotlib` - Visualizations

propably we will need more packages when we get to the modeling part.

### Data Setup

1. Download `Trips_2018.csv` (it's a large file, so I didn't upload it to git, go to DTU learn and download it from there)
2. Place it in the `data/` folder
3. The other data files (`holidays_2018_nyc.csv`, `events_2018_nyc.csv`) are already included

## Current Progress

- [x] **Data Preparation**
  - [x] Load and validate 17.5M trip records
  - [x] Clean data (remove invalid trips, outliers)
  - [x] Handle missing values
- [x] **Station Analysis**

  - [x] Extract 846 unique stations
  - [x] Calculate pickup/dropoff metrics
  - [x] Create interactive map visualization

- [x] **Spatial Clustering**

  - [x] K-Means clustering (k=20-40 clusters)
  - [x] Cluster visualization on map
  - [x] Analyze cluster balance (pickups vs dropoffs)

- [ ] **Feature Engineering**

  - [ ] Time-based features (hour, day, week, month)
  - [ ] Holiday/event flags
  - [ ] Weather data integration (optional)
  - [ ] Lag features for time series (maybe)
  - other

- [ ] **Demand Forecasting**

  - [ ] Train/Validation/Test split (60%/15%/25% maybe?)
  - [ ] play with prediction models
  - [ ] Evaluate performance
  - other?

- [ ] **Bike Rebalancing**
  - [ ] Calculate required bikes per cluster
  - [ ] Minimize shortage scenarios
  - other?
