# Wells Fargo Mortgage Default Predictive Model
## Basic Information
* Organization or People Developing Model: GWU Wells Fargo Predictive Mortgage Default Team (Members: Anukshan Ghosh, Allison Ko, Andrew Renga, and Celina Wong)
* Model Date: May, 2023
* Model Version: 1.0
* License: Apache 2.0
* Model Implementation Code: [Main Code - PySpark_0412.ipynb](https://github.com/celinawong21/WF-ML-Model/blob/main/Main%20Code%20-%20PySpark_0412.ipynb)

## Intended Use
* Primary intended uses: This model is an example of a predictive model for mortgage lenders, financial institutions, and investors to assess and mitigate mortgage lending portfolio risks.
* Primary intended users: Wells Fargo Team, Patrick Hall, Miguel Maldonado de Santillana, and GWU Students in DNSC 6317
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
 * Origination: contains mortgage information at the point of loan initiation
 * Performance: records monthly activities for each loan throughout its contract period.
* The primary focus of the student team is on 30-year-fixed rate mortgages as the main dataset for training the model.
* Before sampling, the dataset consisted of approximately *** million rows.

## Data Preprocessing 
### Data Cleaning 
* PySpark is employed to handle the large dataset, spanning the last 24 years.
* Key strategies implemented in our data preprocessing include:
  * **Handling Missing Data**: Columns that contain 95% or more null values across both datasets have been removed.
  * **Merging Origination and Performance Datasets**: To construct a comprehensive analytical framework, the Performance and Originiation datasets were joined using the LOAN SEQUENCE NUMBER as a key identifier.
    * This merging process guarantees that each record in the Origination dataset is enriched with monthly activities from the performance dataset, providing an complete overview of the loan's lifecycle from its origination to maturity.
  * **Lack of Estimated Loan-to-Value (ELTV) ratio**: ELTV is crucial for modeling to incorporate the financial risk associated with each loan, but there were a significant number of null values present within the Performance dataset.
     * To address this, ELTV was independently calculated by dividing the CURRENT UNPAID BALANCE by the adjusted housing price. The adjusted housing price is determined by applying the change in the Housing Price Index from the loan's origination date to the month of prediction, to the original unpaid balance.

## Variable Selection
* Target variable: is the probability of default rate
* 3 types of input variables
  * **Variables that don't change over time**: CREDIT SCORE, CURRENT LOAN DELINQUENCY STATUS, ORIGINAL INTEREST RATE, PROPERTY TYPE, LOAN PURPOSE, SELLER NAME, FIRST TIME HOMEBUYER FLAG, OCCUPANCY STATUS
  * **Variables that change over time**: CURRENT ACTUAL UPB, LOAN AGE, ESTIMATED LOAN TO VALUE (ELTV)
  * **Variables that change over time and predict the future**: CURRENT INTEREST RATE, UNEMPLOYMENT RATE, INFLATION RATE, HOUSING PRICE INDEX
    * Macroeconomic variables such as inflation, Home Price Index (HPI), and unemployment are loaded from third-party sources.
    * HPI is used nationally to accommodate null values at the state level.
   
### Data Dictionary 
| Name | Variable Type | Data Type |  Modeling Role | Description |
|----------|----------|----------|----------|----------|
| Credit Score| Origination data| Numeric| Input| Prepared by third parties, summarizing the borrower’s creditworthiness, which may be indicative of the likelihood that the borrower will timely repay future obligations|
| Current Loan Delinquency Status| Performance data| Alpha-numeric| Input| A value corresponding to the number of days the borrower is delinquent, based on the due date of last paid installment (“DDLPI”) reported by servicers to Freddie Mac|
| Original Interest Rate| Origination data| Numeric| Input| The interest rate of the loan as stated on the note at the time the loan was originated|
| Property Type| Origination data| Alpha| Input| Denotes whether the property type secured by the mortgage is a condominium, leasehold, planned unit development (PUD), cooperative share, manufactured home, or Single-Family home|
| Loan Purpose| Origination data| Alpha| Input| Indicates whether the mortgage loan is a Cash-out Refinance mortgage, No Cash-out Refinance mortgage, or a Purchase mortgage|
| Seller Name| Origination data| Alpha-numeric| Input| The entity acting in its capacity as a seller of mortgages to Freddie Mac at the time of acquisition|
| First Time Homebuyer Flag| Origination data| Alpha| Input| Indicates whether the Borrower, or one of a group of Borrowers, is an individual who (1) is purchasing the mortgaged property, (2) will reside in the mortgaged property as a primary residence, and (3) had no ownership interest (sole or joint) in a residential property during the three-year period preceding the date of the purchase of the mortgaged property|
| Occupancy Status| Origination data| Alpha| Input| Denotes whether the mortgage type is owner occupied, second home, or investment property|
| Current Actual UPB| Origination data| Numeric| Input| Reflects the mortgage ending balance as reported by the servicer for the corresponding monthly reporting period|
| Percentage of Unpaid Balance| Created| Numeric| Input| The proportion of the original loan amount that remains unpaid at a given point in time.|
| Loan Age| Performance data| Numeric| Input| The number of scheduled payments from the time the loan was originated up to and including the current period|
| Estimated Loan-to-Value (ELTV)| Performance data| Numeric| Input| A ratio indicating current LTV based on the estimated current value of the property|
| Current Interest Rate| Performance data| Numeric| Input| Reflects the current interest rate on the mortgage note, taking into account any loan modifications|
| Unemployment Rate| Macroeconomic| Numeric| Input| The number of unemployed as a percentage of the labor force, reported monthly|
| House Price Index| Macroeconomic| Numeric| Input| A broad measure of single-family house prices that measures average price changes over a period of time, reported quarterly.|
| Inflation| Macroeconomic| Numeric| Input| The rate of increase in prices over a given period of time, reported monthly| 
| Default| Target| Binary| Input| Describes whether a loan is 6 months late on payment|
 
### Sampling Processes
* Due to the extensive size of our dataset, we employed a strategic sampling method to manage our analysis effectively. The key criteria used for sampling were centered around the "CURRENT LOAN DELINQUENCY STATUS". Our methodology is outlined as follows:
  * **Definition of Default**: We identify a loan as default if its "LOAN DELINQUENCY STATUS" is equal to 6 or marked as "RA". Loans not meeting these conditions are classified as non-default.
  * **True_Default**: For clarity in classification, loans meeting the default criteria at any point in time are tagged as "true_default". This distinction allows for precise identification and analysis of loans that default versus those that do not.
  * **Sampling Proportion**: To ensure a balanced representation of default and non-default loans across the 24-year span of our dataset, we adopted a selective sampling approach. The following is the main criteria.
    * We selected 3,000 loans from each year and sampled an equal amount of 350 defaults and 350 non-defaults for each quarter, ensuring that our analysis accurately reflects the dynamics of loan performance over time.
    * Then, we added three quarter variables: OrigData, OrigYear, and OrigQuarter, to track the effect of the quarter for modeling purposes.
    * Due to a shortage of defaults in certain periods, our sampling faced limitations. Specifically, for the fourth quarter of 2022, we could only sample 264 defaults. In 2023, we were able to sample only 32 defaults in the first quarter and no defaults in the second quarter. 
* The following is an loan from the 2003 sample dataset
  
| **Variables**                           |     **Record 0**                      |
|---------------------------|---------------------------|
| LOAN SEQUENCE NUMBER      | F03Q10000272              |
| MONTHLY REPORTING PERIOD  | 2003-02                   |
| CURRENT ACTUAL UPB        | 51000.0000                |
| CURRENT LOAN DELINQUENCY STATUS | 0                   |
| LOAN AGE                  | 0                         |
| CURRENT INTEREST RATE     | 6.1250000                 |
| ESTIMATED LOAN TO VALUE (ELTV) | Undefined             |
| DEFAULT                   | 0                         |
| CREDIT SCORE              | 745                       |
| FIRST TIME HOMEBUYER FLAG | N                         |
| OCCUPANCY STATUS          | P                         |
| ORIGINAL INTEREST RATE    | 6.1250000                 |
| PROPERTY TYPE             | SF                        |
| LOAN PURPOSE              | P                         |
| SELLER NAME               | Other sellers             |
| OrigYear                  | 2003                      |
| OrigQuarter               | Q1                        |
| OrigDate                  | 2003Q1                    |
| index_sa                  | 168.86                    |
| UNRATE                    | 5.9                       |
| inflation                 | 3.0                       |
| % Change in UPB           | 0.0000                    |
 
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
* insert graphs or tables displaying model's within and out of sample fit.
## Risk Considerations
* **Automation Risk**: Discussion on the consequences of relying solely on predictive models for decision-making without human oversight.

## Potential Next Steps 
* **Integration of Additional Data Sources**: consider incorporating regional economic indicators or property market data alongside existing sources like Freddie Mac to enhance predictive accuracy.
* **Dynamic Feature Selection**: develop adaptive feature selection mechanisms to prioritize relevant features and adjust the model's feature set over time based on their importance.
* **Larger Dataset**: apply the modeling techniques to all Freddie Mac single-family home loan data to further incorporate the changes in the economic scenario over time.
* **Government Intervention**: consideration of any regulatory compliance and ethical implications in future iterations of the project.
* **User Interface**: create a front-end development to input certain criteria about a loan and output its potential rate of default.

## Author Contributions
