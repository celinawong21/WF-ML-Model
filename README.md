# Wells Fargo Mortgage Default Predictive Model
## Executive Summary

## Problem Understanding
* The US mortgage market is valued at trillions, ranking second only to interest rates.
Wall Street heavily relies on the success of this market, with banks strategically allocating capital through mortgage bonds.
* Default rates typically remain below 0.2%, but economic crises can cause spikes, reaching highs of 10%.
  * The 2008 financial crisis and the 2020 COVID-19 pandemic highlight the vulnerability of mortgage default rates to economic downturns.
* Collaborating with the Wells Fargo team, the student team's objective is to develop a predictive model for mortgage default over the next 24 months.
* The model will integrate various static, dynamic, and macroeconomic variables to enhance accuracy and robustness.

## The Data
* The dataset utilized for constructing the predictive model is sourced from Freddie Mac and spans from the year 2000 up to the second quarter of 2023.
* It encompasses information on single-family home loans categorized into origination and performance files.
* The primary focus of the student team is on 30-year-fixed rate mortgages as the main dataset for training the model.
* Before sampling, the dataset consisted of approximately 250 million rows.

## Data Preprocessing 
### Sampling Process
* PySpark is employed to handle the large dataset, spanning the last 24 years.
  * Origination and performance data from each year were loaded and joined using the Loan Sequence Number as the common key.
    * Unnecessary variables were filtered out at this stage.  
* All loans not classified as 30-year fixed-rate mortgages are filtered out.
* Loans are then categorized into default or non-default based on their "Current Loan Delinquency Status".
  * Loans with a status of 6 or greater, indicating a delay of at least 6 months, are classified as defaulted, while others are labeled as non-default.
* 10% of non-defaulted loans is sampled, while all defaulted loans are retained, aiming to address the low probability of loan defaults and ensure balanced representation in the model.
* Macroeconomic variables such as inflation, Home Price Index (HPI), and unemployment are loaded.
  * HPI is used nationally to accommodate null values at the state level.
* Estimated Loan-to-Value (ELTV) is calculated to incorporate the financial risk associated with each loan.
  
### Data Dictionary 
| Name | Variable Type | Data Type |  Modeling Role | Description |
|----------|----------|----------|----------|----------|
| Loan Sequence Number| Origination data| Alpha-numeric| Input| Unique identifier assigned to each loan|
| Loan Age| Performance data| Numeric| Input| The number of scheduled payments from the time the loan was originated up to and including the current period.|
| Loan Purpose| Origination data| Alpha| Input| Indicates whether the mortgage loan is a Cash-out Refinance mortgage, No Cash-out Refinance mortgage, or a Purchase mortgage|
| Property Type| Origination data| Alpha| Input| Denotes whether the property type secured by the mortgage is a condominium, leasehold, planned unit development (PUD), cooperative share, manufactured home, or Single-Family home|
| Percentage of Unpaid Balance| Created| Numeric| Input| Content 25|
| Credit Score| Origination data| Numeric| Input| Prepared by third parties, summarizing the borrower’s creditworthiness, which may be indicative of the likelihood that the borrower will timely repay future obligations|
| Current Loan Delinquency Status| Performance data| Alpha-numeric| Input| A value corresponding to the number of days the borrower is delinquent, based on the due date of last paid installment (“DDLPI”) reported by servicers to Freddie Mac|
| Current Interest Rate| Performance data| Numeric| Input| Reflects the current interest rate on the mortgage note, taking into account any loan modifications|
| Estimated Loan-to-Value (ELTV)| Performance data| Numeric| Input| A ratio indicating current LTV based on the estimated current value of the property|
| Unemployment Rate| Macroeconomic| Numeric| Input| Content 50|
| House Price Index| Macroeconomic| Numeric| Input| Content 55|
| Inflation| Macroeconomic| Numeric| Input| Content 60| 
| Default| Content 59| Target| Content 57| Content 60|

## Modeling
* The predictive loan default model utilizes a time series horizon approach.
  * The model aims to forecast the probability of a loan defaulting at a future time (t) based on historical information available up to a snapshot time (s), where s < t.
  * All available information up to time s is utilized, resulting in pairs of snapshots and forecasts, which constitutes stacked data.
  * Each row is duplicated 24 times to predict default probability over the subsequent 24-month period (sample table below).
* The decision to opt for a time series horizon model over a traditional time series model was driven by the latter's diminishing predictive power with increasing time duration.
* Traditional time series models tend to overly emphasize initial lagged time periods, potentially overlooking valuable insights from earlier years.

| age | amount | FICO | delinquency | unemployment | horizon to y | default (y) |
|-----|--------|------|-------------|--------------|--------------|-------------|
| 1   | 125000 | 675  | Current     | 6.2          | 0            | 0           |
| 2   | 125000 | 666  | Current     | 6.2          | 0            | 0           |
| 1   | 125000 | 675  | Current     | 6.2          | 1            | 0           |
| 3   | 125000 | 630  | 30          | 6.2          | 0            | 0           |
| 2   | 125000 | 666  | Current     | 6.2          | 1            | 0           |
| 1   | 125000 | 675  | Current     | 6.2          | 2            | 0           |
| 4   | 125000 | 620  | 60          | 6.1          | 0            | 0           |
| 3   | 125000 | 630  | 30          | 6.2          | 1            | 0           |
| 2   | 125000 | 666  | Current     | 6.2          | 2            | 0           |
| 1   | 125000 | 675  | Current     | 6.2          | 3            | 0           |
