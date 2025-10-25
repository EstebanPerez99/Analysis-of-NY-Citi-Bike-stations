## Introduction to Business Analytics course

Challenge statement

Welcome to this year’s challenge!
The topic this year is mobility. At a time when the world is facing unprecedented challenges
of various kinds, including climate change, pandemics, social inequality, and declining
biodiversity, shared mobility services offer an emission-free mobility option that is both
efficient and attractive. In this project, we invite you to use your best data science skills to
help operators manage their fleet, providing better service and enhancing their business
model. We do not expect you to discover revolutionary business models with a single Data
Sciences project; instead, we want you to address the mandatory questions (below) but also
seek out new questions, new data, and new insights.
You have access to data from Citi Bike (New York)1, one of the biggest station-based bike-
sharing systems in the United States. The dataset includes more than 900 stations and
14000 bikes, and it contains over 17 million bike rides observed during 2018 (yes, it’s huge).
This dataset has the general objective of helping City Bike operate at its best and of making
bike sharing more attractive. You can read more about it here2. The data itself is an
interesting exploration in data science.

## Project
The project has three components:

- Prediction Challenge: All groups must address the same problem.
- Exploratory component: Each group is invited to choose its own research questions and
explore the data accordingly.
- Report: Each group should deliver one report in a paper format 

## The data is provided as a CSV

file. Notice that the variables require extensive treatment to be usable (e.g., Dates,
categorical, strings, different scales, IDs).

For the prediction challenge, you are expected to predict the demand for the bike-sharing
system (number of dropoffs and pickups). You should do the predictions for clusters of
stations. This challenge consists of three tasks:

1. Cluster the stations spatially (nearby departing stations should be grouped together)
in no less than 20 clusters. Tasks 2 and 3 will be based on this clustering, and they
should be completed for at least two clusters (more is preferable) so that you can
compare their respective results and discuss them.

2. You are expected to build a prediction model that, at the end of a day, allows to
predict what the demand for a cluster of stations will be over the next 24 hours – i.e.
not the total demand for the next day, but how the time-series of the demand will look
like for the next day (e.g., given demand data until midnight of day 1, predict the
number of pickups for all 1h intervals (12-1am, 1-2am, 6-7am, 7-8am, …, 11-12pm)
in day 2). You should predict both the arrivals (i.e., bicycle dropoffs) and the
departures (pickups). You should use a time aggregation of one hour or less. You
can choose to use two different models or a single one to predict both. It is up to you
to determine the most effective way to formulate this problem as a machine learning
problem. You should not shuffle the data. You should instead use the data from
January to October (included) to train your model, and the data from November and
December as a test set. You can use any model you want.

3. Overnight, the bike-sharing company manually repositions its bikes to ensure that
demand for the next day can be met. You are expected to use the outputs from the
prediction model above to compute the required number of bicycles to be placed in
each cluster of stations analyzed in Task 2 at the beginning of the next day. To
compute this number, you can use the cumulative of the arrivals and departures. The
goal is to ensure that, over the duration of the next day, there will never be a shortage
of bikes – or, if there is, the goal is to minimize the number of bikes in deficit. The
number of bicycles required can be estimated by extrapolating the maximum
difference between the number of departures and arrivals.


In the exploratory component, each group needs to address at least one new research
question. Here, we expect you to formulate your own question and follow the data sciences
cycle. The project will be positively valued with one or more of the following extensions:
- Extension of the dataset with additional relevant data (such as weather data, national
holidays, and special events).
- Generation and analysis of insightful visualizations;
- Usage of the breadth of techniques from the class beyond regression and data
preparation (e.g., dimensionality reduction, clustering, classification, time series)
Some example research questions:
- How to make predictions for each station? What about a cluster of stations?
- Are there periodic and seasonal trends (e.g., winter, summer), and how can we model
them?
- What is the impact of land use (e.g., proximity to bus/metro station, shops, residential
area vs. business district)?
- What stations are more uncertain in terms of their expected pickups and drop-offs,
and how can we ensure that, with a predefined confidence level (e.g., 90%
confidence), there will be no imbalances between pickups and drop-offs that result in
a shortage of bikes for the next day?
Note: The ordering of tasks we mention is not mandatory. In other words, if you prefer to
start with the exploratory component and then go to the prediction challenge, this is also
acceptable. You can mention that in the report (or invert Sections 2 and 3). Similarly, data
analysis may be presented after the introduction (if relevant). However, please note that a
simple descriptive analysis of the data is insufficient to complete the task. Make sure to go
one step forward and try at least one of the techniques discussed in the course. Finally, in
the exploratory component, you are expected to use at least one of the techniques listed in
class (classification, regression, supervised, unsupervised, dimensionality reduction, …).
Ideally, you should also benchmark multiple techniques (e.g., multiple regressors in the case
of regression).
Evaluation
The evaluation of the paper+notebook(s) will be based on the following criteria:
• Clarity – clarity of the paper and the self-explanatory nature of the notebooks
• Appendices – We will mainly review the paper and only consult the notebook if
needed. If space constraints prevent including certain plots or analyses, they may be
added as appendices in the notebook. These must be clearly referenced in the paper
and, where appropriate, briefly explained.
• Thoroughness - Each research question should be examined to a suitable depth. The
explanatory component is expected to be developed with a level of detail comparable
to the prediction challenge.
• Insightfulness - Don’t stop at summarizing your findings — explain what kind of
business insights or practical implications your results could provide.
• Technical aspects:
- Data have been properly analyzed (data cleaning, data preparation, data pre-
processing).
- Which model has been used (only one model, multiple models, only linear
models, or non-linear models)?
- Is the model and approach appropriate?
- Which performance metrics were used (how performances were evaluated)?
Were they appropriate?
- How was the approach benchmarked (how conclusions were drawn)