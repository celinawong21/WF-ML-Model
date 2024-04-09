# Wells Fargo Mortgage Default Predictive Model
## Basic Information
* Organization or People Developing Model: GWU Wells Fargo Predictive Mortgage Default Team (Members: Anukshan Ghosh, Allison Ko, Andrew Renga, and Celina Wong)
* Model Date: May, 2023
* Model Version: 1.0
* License: Apache 2.0
* Model Implementation Code: 

## Intended Use
* Primary intended uses: This model is an example of a predictive model for mortgage lenders, financial institutions, and investors to assess and mitigate mortgage lending portfolio risks.
* Primary intended users: Wells Fargo Team, Patrick Hall, and GWU Students in DNSC 6317
* Out-of-scope use cases: Any use beyond an educational example is out-of-scope.

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
* Before sampling, the dataset consisted of approximately *** million rows.

## Data Preprocessing 
### Data Cleansing 
* PySpark is employed to handle the large dataset, spanning the last 24 years.
* Our analysis utilized two main types of datasets: the Origination dataset, which contains mortgage information at the point of loan initiation, and the Performance dataset, which records monthly activities for each loan throughout its contract period. Key strategies implemented in our data preprocessing include:
  * **Handling Missing Data**: We have removed columns that contain Null values across both dataset to ensure the completeness of our analysis.
  * **Merging Origination and Performance Datasets**: To construct a comprehensive analytical framework, we integrate data from the Performance dataset into the Origination dataset, using the LOAN SEQUENCE NUMBER as a key identifier. This merging process guarantees that each record in the Origination dataset is enriched with monthly activities from the performance dataset, offering a complete overview of the loan's lifecycle from its origination to maturity.
  * **The Estimated Loan-to-Value (ELTV) ratio** is a crucial variable for our modeling to incorporate the financial risk associated with each loan.; however, we encountered a  
  significant number of null values for ELTV within our Performance  dataset. To address this, we independently calculated ELTV. This involved dividing the Current Unpaid  
  Balance by the adjusted housing price. We determined the adjusted housing price by applying the change in the Housing Price Index from the loan's origination date to the
  month of prediction, to the original unpaid balance.
  * Our target variable is the probablity of default rate
  * There are 3 types of input variables that look into for our analysis
     * **Variables that don't change over time**: CREDIT SCORE, CURRENT LOAN DELINQUENCY STATUS, ORIGINAL INTEREST RATE, PROPERTY TYPE, LOAN PURPOSE, SELLER NAME, FIRST TIME   
     HOMEBUYER FLAG, OCCUPANCY STATUS 
     * **Variables that change over time**: CURRENT ACTUAL UPB, LOAN AGE, STIMATED LOAN TO VALUE (ELTV)
     * **Variables that change over time and predict for the future**: CURRENT INTEREST RATE, UNEMPLOYMENT RATE, INFLATION RATE, HOUSING PRICE INDEX 
* Macroeconomic variables such as inflation, Home Price Index (HPI), and unemployment are loaded. HPI is used nationally to accommodate null values at the state level.

### Sampling Processes
* Due to the extensive size of our dataset, we employed a strategic sampling method to manage our analysis effectively. The key criteria used for sampling were centered around the "CURRENT LOAN DELINQUENCY STATUS". Our methodology is outlined as follows:
  * **Definition of Default**: We identify a loan as default if its "LOAN DELINQUENCY STATUS" is equal to 6 or marked as "RA". Loans not meeting these conditions are classified as non-default.
  * **True_Default**: For clarity in classification, loans meeting the default criteria at any point in time are tagged as "true_default". This distinction allows for  
  precise identification and analysis of loans that default versus those that do not.
  * **Sampling Proportion**: To ensure a balanced representation of default and non-default loans across the 24-year span of our dataset, we adopted a selective sampling  
  approach. Specifically, we sampled 10% of the non-default loans and 100% of the default loans. This approach addresses the relatively lower incidence of defaults within each  
  year, ensuring that our analysis accurately reflects the dynamics of loan performance over time.
* The following is our sample dataset

| Variables                       | RECORD 0         |
|----------------------------------|---------------|
| LOAN SEQUENCE NUMBER             | F00Q10000050  |
| MONTHLY REPORTING PERIOD         | 2000-02       |
| CURRENT ACTUAL UPB               | 164000.0000   |
| CURRENT LOAN DELINQUENCY STATUS  | 0             |
| LOAN AGE                         | 0             |
| CURRENT INTEREST RATE            | 8.1250000     |
| CURRENT NON-INTEREST BEARING UPB | 0.00000       |
| ZERO BALANCE REMOVAL UPB         | NULL          |
| INTEREST BEARING UPB             | 164000.0000   |
| ESTIMATED LOAN TO VALUE (ELTV)   | NULL          |
| DEFAULT                          | 0             |
  
### Data Dictionary 
| Name | Variable Type | Data Type |  Modeling Role | Description |
|----------|----------|----------|----------|----------|
| Loan Sequence Number| Origination data| Alpha-numeric| Input| Unique identifier assigned to each loan|
| Loan Age| Performance data| Numeric| Input| The number of scheduled payments from the time the loan was originated up to and including the current period.|
| Loan Purpose| Origination data| Alpha| Input| Indicates whether the mortgage loan is a Cash-out Refinance mortgage, No Cash-out Refinance mortgage, or a Purchase mortgage|
| Property Type| Origination data| Alpha| Input| Denotes whether the property type secured by the mortgage is a condominium, leasehold, planned unit development (PUD), cooperative share, manufactured home, or Single-Family home|
| Percentage of Unpaid Balance| Created| Numeric| Input| The proportion of the original loan amount that remains unpaid at a given point in time.|
| Credit Score| Origination data| Numeric| Input| Prepared by third parties, summarizing the borrower’s creditworthiness, which may be indicative of the likelihood that the borrower will timely repay future obligations|
| Current Loan Delinquency Status| Performance data| Alpha-numeric| Input| A value corresponding to the number of days the borrower is delinquent, based on the due date of last paid installment (“DDLPI”) reported by servicers to Freddie Mac|
| Current Interest Rate| Performance data| Numeric| Input| Reflects the current interest rate on the mortgage note, taking into account any loan modifications|
| Estimated Loan-to-Value (ELTV)| Performance data| Numeric| Input| A ratio indicating current LTV based on the estimated current value of the property|
| Unemployment Rate| Macroeconomic| Numeric| Input| The number of unemployed as a percentage of the labor force, reported monthly.|
| House Price Index| Macroeconomic| Numeric| Input| A broad measure of single-family house prices that measures average price changes over a period of time, reported quarterly.|
| Inflation| Macroeconomic| Numeric| Input| The rate of increase in prices over a given period of time, reported monthly.| 
| Default| Target| Binary| Input| Describes whether a loan is 6 months late on payment.|

## Modeling
* The predictive loan default model utilizes a time series horizon approach.
  * The model aims to forecast the probability of a loan defaulting at a future time (t) based on historical information available up to a snapshot time (s), where s < t.
  * All available information up to time s is utilized, resulting in pairs of snapshots and forecasts, which constitute stacked data.
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

## Results
## Risk Considerations
## Potential Next Steps 
## Author Contributions
