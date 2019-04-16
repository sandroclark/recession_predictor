# Recession Predictor

Economic recessions and depressions cause substantial damage to people and businesses well-being. One of the biggest problems with how our economic system handles these events is that we are unable to accurately forecast them resulting in sudden panicked realizations that the economy is contracting, deepening recessions destructive power.  By improving our ability to anticipate these macroeconomic events, we can enable economic actors to plan better which will result in a smoother economic cycle. Using machine learning, I predict the probability a contraction in real GDP will occur in the current month, one month out, three months out, and in the next year.

# Instructions

To get the predicted probability of a recession occuring this month to a year out from now, follow the below instructions:
    1) Make sure all prerequisites are met.
    2) Download the most recent version of the Fed yield curve data from here and save it as Fed10Y_3M.csv in the data folders: https://www.newyorkfed.org/research/capital_markets/ycfaq.html
    3) Run individual models by going into their corresponding sub directories and running the 'FINAL' .py script (ex. python log_reg_final.py).
    

# Prerequisites

In order to run the models, you will need the below:

1) A subscription to Trading Economics via Quandl and the Quandl API plugin installed on your computer.
2) The basis_expansions and regression_tools module written by Matt Drury (@madrury) which can be installed with below:
    pip install git+https://github.com/madrury/basis-expansions.git
    pip install git+https://github.com/madrury/regression-tools.git
This is necessary to run the splines within the logistic regression script.
3) Standard Python DataScience packages.

# Models

Three models were built (Logistic Regression, Random Forest, and Gradient Boosted) yielding results in the attached presentation. I spent a significant amount of time also working on an LSTM Neural Network but was unnsuccessful so will add that as a potential next step.

# Next Steps

1) Implement an LSTM.
2) Find additional data series to incorporate into the model including international data.
3) Apply the model framework to predicting fluctuations in exchange rates between major global economies.

# Data

1) US Manufacturing ISM Purchasing Managers Index surveys purchasing managers around the country to see if they're buying more or less, then calculates an index indicating whether US manufacturing is expanding or contracting. 50 indicates flat, while below 50 is contracting and above 50 shows expansion. Releases monthly for the previous month.

2) US Unemployment shows what percentage of Americans in the labor force filed for unemployment benfits. Releases monthly for the previous month.

3) US Youth Unemployment gauges unemployment of youth 16 - 24 and is calculated similarly to the general unemployment rate; those looking for work/ total labor force. Has a 1 month time lag releasing at the end of the month so almost a 2 month time lag.

4) US Consumer Sentiment is calculated based on a survey conducted by the University of Michigan with confidence benchmarked to 1966 at 100. Survey attempts to gauge consumer financial habits related to saving and spending. Released at the end of every monmth.

5) US Part Time For Economic Reasons refers to the percentage of workers working 1 - 34 hours a week for economic reasons but looking for more work. Releases at the start of the next month on the same cadence as Unemployment.

6) Copper PPI is the Producer Price Index for copper specifically. I look at copper as a gauge for economic and industrial growth because it is used in pretty much everything. 

7) Housing Permits indicates how many permits were issued by the government for new housing construction.

8) Housing Starts is how many buildings actually started construction.

9) Capacity Utilization is calculated by the Fed as output/total industrial capacity. 

10) Personal Savings Rate is calculated by the BEA as the amount of money left after people spend money and pay taxes. 

11) Total value of US Exports.

12) Total value of US Imports.

13) US Trade Balance represents US Exports - US Imports.

14) US Federal Funds Rate is the rate at which banks lend overnight money to each other set by the Federal Reserve in order to meet reserve requirements.

15) M2 is a measure of the amount of curency in circulation encompassing cash, checks, savings and deposits.

16) Total number of new sales dor a given month.

17) PPI all commodities, a measure of the prices of a basket of goods used for the production of goods.

18) CPI, a measure of the prices of a basket of consoumer goods.

19) 3 Year Treasury Yield, interest rate that US 3 year bonds yield.

20) 3 Month Treasury Yield, interest rate that US 3 month bonds yield.

21) Spread (3YT - 3MT). When this goes negative it is called a partial inverted yield curve as the return on short term investments is higher than long term investments flagging pessimism in the long term prospects for the economy. A full inverted yield curve happens when all short term Treasuries have a higher yield than longer term.

22) US Real GDP, total consumption, investment, government spending and exports minus imports adjusted for inflation.


