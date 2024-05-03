# Wells Fargo Interpretable Machine Learning for Mortgage Default Prediction
<a name="basic-information"></a>
## Basic Information
* **Organization or People Developing Model**: GWU Wells Fargo Predictive Mortgage Default Team (Members: Anukshan Ghosh, Allison Ko, Andrew Renga, and Celina Wong)
* **Model Date**: April, 2024
* **Model Implementation Code**: [Main Code - PySpark_0412.ipynb](https://github.com/celinawong21/WF-ML-Model/blob/main/Main%20Code%20-%20PySpark_0412.ipynb)
* **Freddie Mac Database**: [Single-family home loan data](https://freddiemac.embs.com/FLoan/secure/login.php?pagename=download) 

## Intended Use
* **Primary intended uses**: This model is an example of a predictive model for mortgage lenders, financial institutions, and investors to assess and mitigate mortgage lending portfolio risks.
* **Primary intended users**: Wells Fargo Team, Patrick Hall, Miguel Maldonado de Santillana, and GWU Students in DNSC 4289/6317
* **Out-of-scope use cases**: Any use beyond an educational example is out-of-scope.

## Executive Summary
* The mortgage market ranks as the second-largest globally, trailing only interest rates, emphasizing its immense scale and significance.
* Banks allocate capital strategically through mortgage bonds, highlighting the industry's pivotal role in financial markets.
* Business outcomes for mortgage models include achieving interpretability and accuracy to foster trust and understanding among stakeholders.
* Predicting default and repayment patterns over an extended period is crucial for risk management and proactive mitigation of default risks.
* Key objectives involve identifying predictors of revenue loss for Wells Fargo and forecasting potential losses over the next 24 months.
* Risk mitigation efforts aim to enhance the stability of mortgage-backed securities, particularly addressing potential downward trends.
* Adaptability across diverse economic scenarios, including stress testing during crises like COVID-19, is vital for evaluating the model's robustness.
  
## Problem Understanding

<div align= "center">
    <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/f9bf0863-31f4-47b7-8f94-a0df810c629d" alt="Image Description" width="650">
</div>

* Default rates typically remain below 0.2%, but economic crises can cause spikes, reaching highs of 9.5%.
  * The 2007/2008 financial crisis and the 2020 COVID-19 pandemic highlight the vulnerability of mortgage default rates in the event of an economic downturn.
* Collaborating with the Wells Fargo team, the student team's objective is to develop a predictive model for mortgage default over the next 24 months.
* The model will integrate various static, dynamic, and macroeconomic variables to enhance accuracy and robustness.

## The Data
* The dataset utilized for constructing the predictive model is sourced from Freddie Mac and spans from the year 2000 up to Q2 of 2023.
* It encompasses information on single-family home loans categorized into Origination and Performance datasets.
   * **Origination**: contains mortgage information at the point of loan initiation.
   * **Performance**: records monthly activities for each loan throughout its contract period.
* The primary focus of the student team is on 30-year-fixed rate mortgages as the main dataset for training the model.

## Data Preprocessing 
### Data Cleaning 
* Before any data preprocessing, there were 64 variables across both datasets and over 2.4 billion rows of data.
* PySpark is employed through GWU's High Power Computing system to handle the large dataset.
* Key strategies implemented in our data preprocessing include:
  * **Handling Missing Data**: columns that contain 95% or more null values across both datasets have been removed.
  * **Merging Origination and Performance Datasets**: to construct a comprehensive analytical framework, the Performance and Origination datasets were joined using the _LOAN SEQUENCE NUMBER_ as a key identifier.
    * The merging process ensures that each record in the Origination dataset is subsequently paired with monthly activities from the Performance dataset, providing a complete overview of the loan's lifecycle from origination to maturity.
  * **Lack of Estimated Loan-to-Value (ELTV) ratio**: ELTV is crucial for modeling to incorporate the financial risk associated with each loan, but there were a significant number of null values present within the Performance dataset.
     * To address this, ELTV was independently calculated by dividing _CURRENT UNPAID BALANCE_ by the adjusted housing price. The adjusted housing price is determined by applying the change in the House Price Index from the loan's origination date to the month of prediction, to the original unpaid balance.

### Variable Selection
* **Target variable**: the probability of default
* Three types of input variables
  * **Variables that don't change over time**: Origination Credit Score, Original Interest Rate, Property Type, Loan Purpose, Seller Name, First-Time Homebuyer Flag, Occupancy Status
  * **Variables that change over time**: Current Actual UPB, Current Loan Delinquency Status, Loan Age, Estimated Loan-to-Value (ELTV)
  * **Leading macroeconomic variables**: Current Interest Rate, Unemployment Rate, Inflation Rate, House Price Index
    * Macroeconomic variables such as inflation, House Price Index (HPI), and unemployment are loaded from third-party sources.
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
| First-Time Homebuyer Flag| Origination data| Alpha| Input| Indicates whether the Borrower, or one of a group of Borrowers, is an individual who (1) is purchasing the mortgaged property, (2) will reside in the mortgaged property as a primary residence, and (3) had no ownership interest (sole or joint) in a residential property during the three-year period preceding the date of the purchase of the mortgaged property|
| Occupancy Status| Origination data| Alpha| Input| Denotes whether the mortgage type is owner occupied, second home, or investment property|
| Current Actual UPB| Origination data| Numeric| Input| Reflects the mortgage ending balance as reported by the servicer for the corresponding monthly reporting period|
| Percentage of Unpaid Balance| Created| Numeric| Input| The proportion of the original loan amount that remains unpaid at a given point in time.|
| Loan Age| Performance data| Numeric| Input| The number of scheduled payments from the time the loan was originated up to and including the current period|
| Estimated Loan-to-Value (ELTV)| Performance data| Numeric| Input| A ratio indicating current LTV based on the estimated current value of the property|
| Current Interest Rate| Performance data| Numeric| Input| Reflects the current interest rate on the mortgage note, taking into account any loan modifications|
| Unemployment Rate| Macroeconomic| Numeric| Input| The number of unemployed as a percentage of the labor force, reported monthly|
| House Price Index| Macroeconomic| Numeric| Input| A broad measure of single-family house prices that measures average price changes over a period of time, reported quarterly.|
| Inflation| Macroeconomic| Numeric| Input| The rate of increase in prices over a given period of time, reported monthly| 
| Default| Target| Binary| N/A | Describes whether a loan is 6 months late on payment|
 
## Sampling
### Methodology
* Due to the extensive size of the dataset, a strategic sampling method was employed to manage the data. The key criteria used for sampling were centered around the _CURRENT LOAN DELINQUENCY STATUS_.
  * **Criteria for a Defaulted Loan**: if _CURRENT LOAN DELINQUENCY STATUS_ is equal to 6, meaning that payment on the loan is at least 6 months late or marked as "RA".
    * Loans not meeting these conditions are classified as non-default.
  * **True_Default**: For clarity in classification, loans meeting the default criteria at any point in time are tagged as "true_default". This distinction allows for precise identification and analysis of loans that default versus those that do not.
  * **Sampling Proportion**: to ensure a balanced representation of default and non-default loans across the 24 years of our dataset, a selective sampling approach was adopted. The following are the main criteria.
    * 3,000 loans were selected from each year and sampled an equal amount of 350 defaults and 350 non-defaults for each quarter, to ensure that the analysis accurately reflects the dynamics of loan performance over time.
        * The sampling faced limitations due to a shortage of defaults in certain periods. Specifically, for the fourth quarter of 2022, only 264 defaulted loans were sampled. In 2023, only 32 defaults in the first quarter were sampled and zero defaulted loans were found in the second quarter. 
    * Then, three time variables were added (_OrigDate_, _OrigYear_, and _OrigQuarter_), to track the effect of the quarter for modeling purposes.
    
#### Example from Sampled Data: 2003 Q1, Record 0 
  
```python
-RECORD 0----------------------------------------
 LOAN SEQUENCE NUMBER            | F03Q10000272  
 MONTHLY REPORTING PERIOD        | 2003-02       
 CURRENT ACTUAL UPB              | 51000.0000    
 CURRENT LOAN DELINQUENCY STATUS | 0             
 LOAN AGE                        | 0             
 CURRENT INTEREST RATE           | 6.1250000     
 ESTIMATED LOAN TO VALUE (ELTV)  | Undefined     
 DEFAULT                         | 0             
 CREDIT SCORE                    | 745           
 FIRST-TIME HOMEBUYER FLAG       | N             
 OCCUPANCY STATUS                | P             
 ORIGINAL INTEREST RATE          | 6.1250000     
 PROPERTY TYPE                   | SF            
 LOAN PURPOSE                    | P             
 SELLER NAME                     | Other sellers 
 OrigYear                        | 2003          
 OrigQuarter                     | Q1            
 OrigDate                        | 2003Q1        
 index_sa                        | 168.86        
 UNRATE                          | 5.9           
 inflation                       | 3.0           
 % Change in UPB                 | 0.0000
```     

A sample of the first 20 rows of the [2000 Sample Data](Sample_2000_First_20.csv) is included in the repository.
 
## Modeling Using Non-stacked Data
### Features Selection 
Based on the feature select function in PiML, the following features were chosen. 

**_Numerical variables_**
* Credit Score
* Current Interest Rate
* Estimated Loan-to-Value (ELTV)
* Original Interest Rate
* Index_sa
* UNRATE(Unemployment rate)
* Inflation
* % change in UPB

**_Categorical variables_**
* First-Time Homebuyer Flag
* Occupancy Status
* Property Type
* Loan Purpose
* Seller Name
* OrigYear
* OrigDate

### Sampling for Parameters

* [SampleForParameter.csv](https://github.com/celinawong21/WF-ML-Model/blob/main/sampleforparameter.csv) is a smaller sample that shows the merged data, which is then used to obtain hyperparameters for both the XGB1 and XGB2 models. Subsequently, these parameters, along with the monotonic variables, are utilized to train four different models for both XGB1 and XGB2.
* SampleForParameter.csv is used to obtain parameters for the XGB1 model through the grid search method. The parameters obtained from the first row of the grid search results, corresponding to Rank 1, will be utilized in the XGB1 model and its versions.

### XGBoost 

```python
np.random.seed(12345)

parameters = {'n_estimators': [100, 500, 1000],
              'eta': [0.01, 0.1, 0.5],
              'reg_lambda': [0.0, 0.5, 1.0],
              'reg_alpha': [0.01, 0.5, .99]}
result = exp.model_tune("XGB1", method="grid", parameters=parameters, metric=['MSE', 'MAE'], test_ratio=0.2, random_state = 12345)
result.data
```

| Params                                        | Rank(by MSE) | MSE     | MAE     | Time      |
|-----------------------------------------------|--------------|---------|---------|-----------|
| {'eta': 0.01, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 0.0} | 1            | 0.337091 | 0.337091 | 7.993272  |
| {'eta': 0.01, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 0.5}  | 1            | 0.337091 | 0.337091 | 9.407335 |
| {'eta': 0.01, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 1.0}   | 1            | 0.337091 | 0.337091 | 13.715970 |
| {'eta': 0.01, 'n_estimators': 100, 'reg_alpha': 0.5, 'reg_lambda': 0.0}   | 1            | 0.337091 | 0.337091 | 13.8588046 |
| {'eta': 0.01, 'n_estimators': 100, 'reg_alpha': 0.5, 'reg_lambda': 0.5}   | 1            | 0.337091 | 0.337091 | 12.984432 |

```python
np.random.seed(12345)

parameters = {'n_estimators': [100, 500, 1000],
              'eta': [0.01, 0.1, 0.5],
              'reg_lambda': [0.0, 0.5, 1.0],
              'reg_alpha': [0.01, 0.5, .99]}
result = exp.model_tune("XGB2", method="grid", parameters=parameters, metric=['MSE', 'MAE'], test_ratio=0.2, random_state = 12345)
result.data
```
| Params                                        | Rank(by MSE) | MSE     | MAE     | Time      |
|-----------------------------------------------|--------------|---------|---------|-----------|
| {'eta': 0.5, 'n_estimators': 1000, 'reg_alpha': 0.01, 'reg_lambda': 0.5} | 1            | 0.146026 | 0.146026 | 6.809227  |
| {'eta': 0.5, 'n_estimators': 1000, 'reg_alpha': 0.5, 'reg_lambda': 0.5}  | 1            | 0.146080 | 0.146080 | 6.445565 |
| {'eta': 0.5, 'n_estimators': 1000, 'reg_alpha': 0.99, 'reg_lambda': 0.0}   | 1            | 0.146803 | 0.146803 | 5.550786 |
| {'eta': 0.5, 'n_estimators': 1000, 'reg_alpha': 0.5, 'reg_lambda': 0.0}   | 1            | 0.146856 | 0.146856 | 5.722547 |
| {'eta': 0.5, 'n_estimators': 1000, 'reg_alpha': 0.99, 'reg_lambda': 0.5}   | 1            | 0.146937 | 0.146937 | 5.495173 |

* SampleForParameter.csv  is then used to derive parameters for the XGB2 model through the grid search method. The parameters extracted from the first row of the grid search results, corresponding to Rank 1, will be employed in the XGB2 model and its versions.


### Comparing XGB1 and XBG2
Both of the results are sorted by the highest Area Under the Curve (AUC) value, providing a comprehensive comparison of model performance.
* **XGB**: Base model with default parameters and no monotonic variables.
* **XGB_V2**: Variant of XGB with default parameters and incorporating two monotonic variables: "CURRENT INTEREST RATE" and "ORIGINAL INTEREST RATE" (monotonic increasing) and "CREDIT SCORE" (monotonic decreasing).
* **XGB_V3**: Another variation of XGB1, maintaining default parameters and excluding monotonic variables.
* **XGB_V4**: Similar to XGB_V2, featuring default parameters alongside the two monotonic variables: "CURRENT INTEREST RATE" and "ORIGINAL INTEREST RATE" (monotonic increasing) and "CREDIT SCORE" (monotonic decreasing).

#### XGB1
|   | Model   | test_ACC | test_AUC | test_F1 | test_LogLoss | test_Brier | train_ACC | train_AUC | train_F1 | train_LogLoss | train_Brier |
|---|---------|----------|----------|---------|--------------|------------|-----------|-----------|----------|---------------|-------------|
| 0 | XBG1    | 0.6963   | 0.7698   | 0.7170  | 0.5875       | 0.1993     | 0.6903    | 0.7617    | 0.7242   | 0.5965        | 0.2037      |
| 1 | XBG1_v2 | 0.6963   | 0.7698   | 0.7170  | 0.5875       | 0.1993     | 0.6903    | 0.7617    | 0.7242   | 0.5965        | 0.2037      |
| 3 | XBG1_v4 | 0.6410   | 0.7473   | 0.7235  | 0.6456       | 0.2256     | 0.6054    | 0.6548    | 0.6542   | 0.6666        | 0.2365      |
| 2 | XBG1_v3 | 0.6410   | 0.7452   | 0.7235  | 0.6417       | 0.2243     | 0.6054    | 0.6535    | 0.6542   | 0.6659        | 0.2364      |


#### XGB2
|   | Model   | test_ACC | test_AUC | test_F1 | test_LogLoss | test_Brier | train_ACC | train_AUC | train_F1 | train_LogLoss | train_Brier |
|---|---------|----------|----------|---------|--------------|------------|-----------|-----------|----------|---------------|-------------|
| 0  | XBG2    | 0.7278   | 0.8284   | 0.7549  | 0.5281       | 0.1768     | 0.7496    | 0.8270    | 0.7559   | 0.5150        | 0.1709      |
| 1 | XBG2_v2 | 0.7333   | 0.8143   | 0.7401  | 0.5288       | 0.1768     | 0.7500    | 0.8280    | 0.7551   | 0.5151        | 0.1707      |
| 3 | XBG2_v4 | 0.6635   | 0.7199   | 0.6405  | 0.8378       | 0.2521     | 0.8241    | 0.9019    | 0.8246   | 0.4026        | 0.1262      |
| 2 | XBG2_v3 | 0.6461   | 0.7145   | 0.6978  | 0.9345       | 0.2728     | 0.78305   | 0.9073    | 0.8311   | 0.3955        | 0.1234      |


## Model Interpretation: XGB2_v2

### Effect Plot

* Monotonicity adjustments for three variables: 
    * **Monotonic Increasing**: Current Interest Rate, Original Interest Rate
    * **Monotonic Decreasing**: Credit Score

Monotonicity adjustments were made to the model's feature to ensure that the relationship between the feature and the target variable follows a consistent trend, either always increasing or always decreasing, thereby improving the model's interpretability.

<table>
  <tr>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/349c32d1-4616-47df-b1e4-7098933a7439" alt="Before Monotonic Adjustment - Current Interest Rate" width="450"/>
    </td>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/2144f137-2d7c-4116-b0f7-84c46e063296" alt="After Monotonic Adjustment - Current Interest Rate" width="450"/>
    </td>
  </tr>
  <tr>
    <td>Before Monotonic Adjustment - Current Interest Rate</td>
    <td>After Monotonic Adjustment - Current Interest Rate</td>
  </tr>
</table>

Regarding the Current Interest Rate, it has a positive relationship with the target variable, implying that as the current interest rate increases, the probability of default also increases. Following the monotonicity adjustment, the influence of the current interest rate on the predictions slightly rose from 3.2% to 3.7%.


<table>
  <tr>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/91e05703-5530-4598-a64f-c761aabed586" alt="Before Monotonic Adjustment - Original Interest Rate" width="450"/>
    </td>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/f685db71-f82e-48a9-ad12-b391522a6f70" alt="After Monotonic Adjustment - Original Interest Rate" width="450"/>
    </td>
  </tr>
  <tr>
    <td>Before Monotonic Adjustment - Original Interest Rate</td>
    <td>After Monotonic Adjustment - Original Interest Rate</td>
  </tr>
</table>

The original interest rate has a positive relationship with the target variable, indicating that as the original interest rate increases, the probability of default increases. Following the monotonicity adjustments, the effect of the original interest rate on the predictions has not changed and remained at 0.9. This implies that the relationship between the original interest rate and the probability of default was already consistent with the expected behavior.

<table>
  <tr>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/1352fd58-136a-451d-a1b2-e57a33613133" alt="Before Monotonic Adjustment - Credit Score" width="450"/>
    </td>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/3c51717e-77cb-46f2-94a3-c6248af1154d" alt="After Monotonic Adjustment - Credit Score" width="450"/>
    </td>
  </tr>
  <tr>
    <td>Before Monotonic Adjustment - Credit Score</td>
    <td>After Monotonic Adjustment - Credit Score</td>
  </tr>
</table>

For the Credit Score, a negative relationship with the target variable is observed, indicating that as the credit score increases, the probability of default decreases. After the monotonicity adjustment, the impact of the credit score on the model's predictions decreased slightly from 4.1% to 4.0%. This suggests that the adjustment may have smoothed out irregularities in the data that initially contributed to a stronger relationship.


### Global Interpretability

<table>
  <tr>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/19b956ee-3eae-4eec-9991-ea8422a9f8f5" alt="Effect Importance" width="450"/>
    </td>
    </td>
  </tr>
</table>


### Interaction Effect: Three interaction effects with the highest percentages
The interaction plots show how the interaction between two features affect the probability of default. The top 3 interactions were selected based on the highest percentage values.


* **ELTV x % Change in UPB**
<img width="550" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/ebf23807-4f55-4bca-8ef2-b0598b82883b">

The interaction plot shows that if a loan amount is nearly the same or more than what the property is worth, and the borrower isn't paying it off well, there's a much bigger risk (3.2% higher) that they won't be able to finish paying the loan. This can often lead to a situation where the borrower may not be able to pay back the loan at all, especially if the value of the property doesn't cover the loan and the remaining loan amount isn't dropping. But if the borrower is making regular payments and reducing the loan amount, the risk goes down, even if the loan started out big compared to the property's value. 


* **HPI x Unemployment Rate**
<img width="550" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/e3a262c9-f70d-4a5b-a740-7e5b83b6933a">

The interaction effect between the housing price index and the unemployment rate is observed to be the second highest at 1.7%. The analysis of the plot suggests that a stagnant or declining housing market, combined with a rising unemployment rate, increase the risk of loan defaults. This correlation aligns with economic rationale: if individuals are unable to secure employment, their capacity to fulfill financial obligations, such as loan repayments, is compromised, thereby elevating the likelihood of default. In essence, the inability to find employment, coupled with depreciating housing values, significantly amplifies the probability of default.


* **% ELTV x HPI** 

<img width="550" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/74a4a5a2-b682-4b0f-a933-35b1e8cb2598">

The intereaction plot shows that when the loan is as much as the property's worth or more, the risk that the borrower won't be able to pay back the loan is higher. This risk is even greater when the housing market isn't doing well. Basically, if a home loan is worth more than the house, and the market is weak, there's a bigger chance the loan won't be paid off.


## Results
### Accuracy Descriptions of XGB2_v2 Model

|         |  ACC   |  AUC   |   F1   | LogLoss |  Brier |
|---------|--------|--------|--------|---------|--------|
|  Train  | 0.7500 | 0.8280 | 0.7551 |  0.5151 | 0.1707 |
|  Test   | 0.7333 | 0.8143 | 0.7401 |  0.5288 | 0.1768 |
|   Gap   |-0.0168 |-0.0137 |-0.0150 |  0.0137 | 0.0060 |

<img width="500" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/5ea5dc31-3c8a-4d2d-b555-b84bc7cc1e87">


### Residual Box Plot of Predicted Default Variable from XGB2_v2 Model

<img width= "650" alt = "image" src= "https://github.com/celinawong21/WF-ML-Model/assets/159848729/1d75d30a-eaab-4469-b1c3-a287e143dd61">

## Modeling Using Stacked Data 
### Overview of Time Series Horizon
* The predictive loan default model utilizes a paneled approach with horizons representing different periods in time across the entire dataset. For the purposes of this project and code, the paneled data will be referred to as time series data. 
  * The model aims to forecast the probability of a loan defaulting at a future time (t) based on historical information available up to a snapshot time (s), where s < t.
  * All available information up to time s is utilized, resulting in pairs of snapshots and forecasts, which constitute stacked data.
  * Each row is duplicated 24 times to predict default probability over the subsequent 24-month period (sample table below).
* The decision to opt for a time series horizon model over a traditional time series model was driven by the latter's diminishing predictive power with increasing time duration.
* Traditional time series models tend to overly emphasize initial lagged time periods, potentially overlooking valuable insights from earlier years.

### Creating a Stacked Dataset
#### Vectorized Process to Create the Stacked Dataset
* The process begins with obtaining a sample file containing 3,000 loans from each of the 24 years. Subsequently, the data undergoes a vectorized transformation to generate a time series dataframe.
* Then, the minimum _LOAN AGE_ for each _LOAN SEQUENCE NUMBER_ group is identified, which is the starting point for each loan.
    * During the vectorization process, the combination of _LOAN SEQUENCE NUMBER_ and _LOAN AGE_ is documented for each iteration of the horizons.
    * In cases where multiple _LOAN SEQUENCE NUMBERS_ exhibit the same LOAN AGE throughout the duration of a HORIZON, adjustments are made to ensure chronological order within each _LOAN SEQUENCE NUMBER_. This involves recalculating loan ages for duplicate rows, thus preserving the sequential progression of loan ages. 
* Each row in the merged dataset is replicated 24 times to project loan information for 24 months into the future. This duplication enables forecasting loan behavior over an extended period.
* To enhance analysis, two new columns, _HORIZON_ and _SOURCE_, are introduced.
    * **HORIZON**: tracks past information, with each horizon representing a month in the past.
        * For instance, if a loan's monthly reporting period is "2013-06", HORIZON(1) corresponds to duplicated data from "2013-05", and HORIZON(2) corresponds to duplicated data from "2013-04".
    * **SOURCE**: distinguishes between original sample rows ("orig") and those generated through the vectorized process ("Duplicated").
* The process will continue, incrementing the 'LOAN AGE' by one consistently until the loan reaches the end of its lifecycle.

#### Example of Stacked Data

| Group | DEFAULT | Horizon | Source | LOAN SEQUENCE NUMBER | MONTHLY REPORTING PERIOD | CURRENT ACTUAL UPB | CURRENT LOAN DELINQUENCY STATUS | LOAN AGE | CURRENT INTEREST RATE |
|-------|---------|---------|--------|-----------------------|---------------------------|---------------------|---------------------------------|----------|-----------------------|
| 0     | 0       | 0       | orig   | F00Q10000066          | 2000-02                   | 132000.0            | 0                               | 0        | 8.0                   |
| 1     | 0       | 0       | orig   | F00Q10000066          | 2000-03                   | 132000.0            | 0                               | 1        | 8.0                   |
| 1     | 0       | 1       | Dupli… | F00Q10000066          | 2000-02                   | 132000.0            | 0                               | 0        | 8.0                   |
| 2     | 0       | 0       | orig   | F00Q10000066          | 2000-04                   | 131000.0            | 0                               | 2        | 8.0                   |
| 2     | 0       | 1       | Dupli… | F00Q10000066          | 2000-03                   | 132000.0            | 0                               | 1        | 8.0                   |
| 2     | 0       | 2       | Dupli… | F00Q10000066          | 2000-02                   | 132000.0            | 0                               | 0        | 8.0                   |
| 3     | 0       | 0       | orig   | F00Q10000066          | 2000-05                   | 131000.0            | 0                               | 3        | 8.0                   |
| 3     | 0       | 1       | Dupli… | F00Q10000066          | 2000-04                   | 131000.0            | 0                               | 2        | 8.0                   |
| 3     | 0       | 2       | Dupli… | F00Q10000066          | 2000-03                   | 132000.0            | 0                               | 1        | 8.0                   |
| 3     | 0       | 3       | Dupli… | F00Q10000066          | 2000-02                   | 132000.0            | 0                               | 0        | 8.0                   |

A sample of the first 48 rows of the [2000 Stacked Data](Stacked_2000_First_48.csv) is included in the repository.

### XGBoost2

|     | Model    | test_ACC | test_AUC | test_F1 | test_LogLoss | test_Brier | train_ACC | train_AUC | train_F1 | train_LogLoss | train_Brier |
|-----|----------|----------|----------|---------|--------------|------------|-----------|-----------|----------|---------------|-------------|
|  0  | XGB2     | 0.6656   | 0.7361   | 0.6367  | 0.6137       | 0.2125     | 0.7004    | 0.7695    | 0.7083   | 0.5729        | 0.1952      |
|  1  | XGB2_v2  | 0.6681   | 0.7287   | 0.6416  | 0.6257       | 0.2160     | 0.7003    | 0.7702    | 0.7070   | 0.5726        | 0.1950      |

## Model Interpretation: XGB2_v2

### Effect Plot

* Monotonicity adjustments for two variables: 
    * **Monotonic Increasing**: Current Interest Rate 
    * **Monotonic Decreasing**: Credit Score

<table>
  <tr>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/42744c3f-7c26-451e-9dd4-71f6da05e70a" alt="Before Monotonic Adjustment - Credit Score" width="450"/>
    </td>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/20d39f03-a94d-40c5-840e-42a3ed31393c" alt="After Monotonic Adjustment - Credit Score" width="450"/>
    </td>
  </tr>
  <tr>
    <td>Before Monotonic Adjustment - Credit Score</td>
    <td>After Monotonic Adjustment - Credit Score</td>
  </tr>
</table>

For Credit Score, a negative relationship with the target variable is observed, indicating that as the credit score increases, the probability of default decreases. After the monotonicity adjustment, the impact of the credit score on the model's predictions decreased slightly from 16.4% to 16.2%. This suggests that the adjustment may have smoothed out irregularities in the data that initially contributed to a stronger relationship.

<table>
  <tr>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/ef8a6432-a3e7-484b-895a-034931fbfe20" alt="Before Monotonic Adjustment - Current Interest Rate" width="450"/>
    </td>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/28f55109-75a8-43f4-8e25-9bc4d8ea8d75" alt="After Monotonic Adjustment - Current Interest Rate" width="450"/>
    </td>
  </tr>
  <tr>
    <td>Before Monotonic Adjustment - Current Interest Rate</td>
    <td>After Monotonic Adjustment - Current Interest Rate</td>
  </tr>
</table>

Regarding Current Interest Rate, it has a positive relationship with the target variable, implying that as the current interest rate increases, the probability of default also increases. Following the monotonicity adjustment, the influence of the current interest rate on the predictions slightly rose from 14.2% to 15.2%.

### Global Interpretability

<table>
  <tr>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/f6bfbd59-e2cf-4cf1-ae85-458e8d589c76" alt="Effect Importance" width="450"/>
    </td>
    </td>
  </tr>
</table>

The effect importance graph shows which the model finds most important when predicting outcomes. The feature with the highest effect is "index_sa," indicating it has a strong relationship with the model's predictions, followed by Credit scores and current interest rates. Economic indicators like the unemployment rate ("UNRATE") and the original interest rate are less influential, but still notable. The importance graph depicts the effect, aiding in the understanding of prioritized data and interpretation of predictions.

### Interaction Effect: Three interaction effects with the highest percentages
* **%Change in UPB x ELTV**

<img width="550" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/d548a5cc-beb8-428c-9fd5-ff901b81b05b">

The interaction plot reveals that small changes in Unpaid Principal Balance (UPB) paired with high Estimated Loan to Value (ELTV) ratios can lead to an increased chance of default. This relationship tends to weaken as the UPB change grows. The cited 2.1% figure measures the relative contribution of this interaction to the accuracy of the model’s default probability predictions.

* **Horizon x Unemployment Rate**

<img width="550" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/5a8da587-4866-4e6b-89e2-1ceabe59721c">

The interaction plot demonstrates how the unemployment rate (UNRATE) may influence predictions of default probability across varying time horizons. The graph indicates that the effect of the unemployment rate on risk assessment may evolve as the time horizon extends. The indicated 1.4% represents the interaction’s proportional influence on the likelihood of default, emphasizing that the impact of economic factors like unemployment can shift over time.

* **Horizon x HPI**
  
<img width="550" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/40d35c78-bd6c-45be-984e-d1b802690666">

The interaction plot illustrates that the interaction between the housing market index ('index_sa') and the horizon could affect default risk assessments. The significance of the housing market's status in predicting defaults may alter as the horizon grows. The 1.1% highlighted implies that this interaction accounts for a certain amount of the variability in predicting defaults, highlighting the need to factor in long-term economic trends in risk models.

## Results
### Accuracy Descriptions of XGB2_v2 Model

|         |  ACC   |  AUC   |   F1   | LogLoss |  Brier |
|---------|--------|--------|--------|---------|--------|
|  Train  | 0.7003 | 0.7702 | 0.7070 |  0.5726 | 0.1950 |
|  Test   | 0.6681 | 0.7287 | 0.6414 |  0.6257 | 0.2160 |
|   Gap   |-0.0323 |-0.0415 |-0.0656 |  0.0530 | 0.0210 |

<img width="500" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/d684fd69-ea70-461c-a996-7fd36acbe9e9">

### Residual Box Plot of Predicted Default Variable from XGB2_v2 Model

<img width= "650" alt = "image" src= "https://github.com/celinawong21/WF-ML-Model/assets/159848729/9e8f2fcd-ef9f-4825-8d33-c5effd0d0bf7">

### Predicted vs. Actual Default Rate: XGB2_v2  

<img width="600" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/158225115/c6e6ba99-25e1-4794-bb0f-2e6d6496f522">

## Overfit Test
### Property Type
<img width="400" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/69b4f8fd-d7bb-4951-9cf9-13d7c2183314">

|   | (PROPERTY TYPE) | (PROPERTY TYPE) | #Test  | #Train | test_AUC | train_AUC | Gap     |
|---|-----------------|-----------------|--------|--------|----------|-----------|---------|
| 0 | 1.8             | 3.4             | 160040 | 128905 | 0.5946   | 0.7733    | -0.1787 |




### Original Interest Rate
<img width="400" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/ff3c011e-b8cd-4f66-ba63-cd15978575d2">

|            | Original Interest Rate | Original Interest Rate | #Test   | #Train   | test_AUC | train_AUC | Gap     |
|------------|------------------------|------------------------|---------|----------|----------|-----------|---------|
| 0          | 0.3555                 | 0.5111                 | 260486  | 31852579 | 0.6115   | 0.7716    | -0.1601 |
| 1          | -0.0334                | 0.1221                 | 5029161 | 3415691  | 0.6712   | 0.8478    | -0.1766 |


### Seller Name
<img width="400" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/5560d835-8e49-4d1e-b4fb-773a10bd8d84">

|            | Seller Name | Seller Name | #Test  | #Train  | test_AUC | train_AUC | Gap     |
|------------|-------------|-------------|--------|---------|----------|-----------|---------|
| 0          | 34.8        | 40.1        | 105782 | 2151194 | 0.5421   | 0.727     | -0.1849 |


### HPI
<img width="400" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/37a7eb60-7b8e-4ddf-8437-74622059c0fa">

|            | index_sa | index_sa | #Test   | #Train  | test_AUC | train_AUC | Gap     |
|------------|----------|----------|---------|---------|----------|-----------|---------|
| 0          | 0.854    | 1.0      | 2956998 | 1266981 | 0.5595   | 0.7434    | -0.1838 |


### Inflation
<img width="400" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/a02def46-2ad7-4fa4-a329-99d974854957">

|            | Inflation | Inflation | #Test   | #Train  | test_AUC | train_AUC | Gap     |
|------------|-----------|-----------|---------|---------|----------|-----------|---------|
| 0          | 0.5018    | 0.5848    | 409544  | 7663750 | 0.6347   | 0.7874    | -0.1527 |
| 1          | 0.7509    | 1.000     | 3235874 | 1474168 | 0.5695   | 0.7337    | -0.1642 |


### % Change in UPB
<img width="353" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/730233f9-56a4-4f1a-bd99-98cca36755da">

|            | % Change in UPB | % Change in UPB | #Test   | #Train  | test_AUC | train_AUC | Gap     |
|------------|-----------------|-----------------|---------|---------|----------|-----------|---------|
|            | 0               | 0.0             | 0.0579  | 14883   | 0.6495   | 0.7812    | -0.1317 |



## Risk Considerations
* **Automation Risk**: Potential consequences of solely relying on predictive models for decision-making without human oversight. 
* **Sampling Bias**: Careful consideration is given to the implications of sampling a minuscule proportion of the overall data, which may introduce biases or limit the model's ability to accurately predict defaults during periods of crisis. 
* **Biased Data during Crisis**: Inherent biases in data collected during times of crisis, as loans may predominantly be issued to customers with strong financial profiles, skewing the dataset. It underscores the significance of not only predicting defaults but also anticipating and mitigating crises beforehand. By identifying early warning signs, proactive measures can be implemented to avert potential crises and minimize their impact.
* **Racial Disparities in Mortgage Loans**: There are significant disparities in mortgage lending practices across racial demographics within the industry.

## Potential Next Steps 
* **Larger Dataset**: Apply the modeling techniques to all Freddie Mac single-family home loan data to further incorporate the changes in the economic scenario over time.
* **Integration of Additional Data Sources**: Consider incorporating regional economic indicators or property market data alongside existing sources like Freddie Mac to enhance predictive accuracy.
* **Government Intervention**: Consider any regulatory compliance and ethical implications in future iterations of the project.
* **User Interface**: Create a front-end development to input certain specifics about a loan and/or macroeconomic variables to output a potential rate of default. This application will take user input, visually explain the impact of each variable, and attempt to boost the interpretability of the model.
* **Calibration**: Implement calibration adjustments, which involves post-processing the model's output probabilities to better align with the true distribution of the data. This will ensure more accurate probability estimates, mitigating any biases introduced by the oversampling technique.

## Appendix
### Local Interpretability: Non-stacked Data

<table>
  <tr>
    <td>
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/22421870-f4a2-4ef2-acdd-c461e36c3811">
    </td>
  </tr>
</table>


### Resilience Test - Worst Sample for Top 4 Most Important Features from XGB2_v2

#### Distribution Shift Analysis

This section contains visualizations of the distribution shifts for various features.

<table style="width:100%;">
  <tr>
    <td style="text-align:center;">
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/9939fb94-c6c7-4d0d-b14c-edc8ca5fa084" alt="Distribution Shift: % Change in UPB" style="width:100%;" />
      <p style="margin-top: 10px; font-weight: bold;">Distribution Shift: Housing Price Index Seasonally Adjusted (index_sa)</p>
    </td>
    <td style="text-align:center;">
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/66a3f567-2ce7-42d8-b823-3303dd2d956e" alt="Distribution Shift: Estimated Loan-to-Value (ELTV)" style="width:100%;" />
      <p style="margin-top: 10px; font-weight: bold;">Distribution Shift: % Change in UPB
</p>
    </td>
  </tr>
  <tr>
    <td style="text-align:center;">
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/a7f7df12-e9f4-4c5c-9290-3186ef2a02fe" alt="Distribution Shift: Housing Price Index Seasonally Adjusted (index_sa)" style="width:100%;" />
      <p style="margin-top: 10px; font-weight: bold;">Distribution Shift: Estimated Loan-to-Value (ELTV)</p>
    </td>
    <td style="text-align:center;">
      <img src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/12eeb9b3-108c-4938-acb1-de5b58df6ee3" alt="Distribution Shift: Unemployment Rate (UNRATE)" style="width:100%;" />
      <p style="margin-top: 10px; font-weight: bold;">Distribution Shift: Unemployment Rate (UNRATE)</p>
    </td>
  </tr>
</table>

### Local Interpretability: Stacked Data

<img width="900" alt="image" src="https://github.com/celinawong21/WF-ML-Model/assets/159848729/308793a6-3dc2-4d72-9a3c-cb1a84fcbb2b">

