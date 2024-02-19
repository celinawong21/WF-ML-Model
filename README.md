# Wells Fargo Mortgage Default Predictive Model
## Executive Summary

In partnership with Wells Fargo, the objective is to build a predictive model to forecast mortgage default rates for the next 24 months. By leveraging Freddie Mac's loan origination and performance records and incorporating key macroeconomic factors that impact the housing industry, the model will simulate future scenarios and the likelihood of default. The success of the model is important to financial institutions like Wells Fargo, as a primary source of revenue for lies in mortgage-backed securities.  Identifying high default rates before they occur allows the bank to be proactive in risk management to protect its financial health and reputation. 

The model will analyze various factors influencing default rates, such as current loan-to-value ratios, historical performance data, and unemployment, among many others. The model will be trained on a sample of non-default and defaulted mortgages with probability  calibrations to accurately assign weights to the variables. It will also focus on key economic crisis periods, including the 2008 Great Recession and the COVID-19 pandemic to test the accuracy of the prediction.
With the model, anticipated outcomes include improved risk management capabilities and sustained long-term profitability for Wells Fargo's mortgage operations. 

## Problem Understanding 
Wells Fargo's business model and financial stability hinge significantly on accurately predicting default rates. As one of the largest retail mortgage originators in the United States, the trillion-dollar industry relies heavily on the performance and safety of its mortgage-backed securities. Forecasting default rates enables Wells Fargo to anticipate potential revenue losses and implement effective risk mitigation strategies, thus safeguarding against economic downturns. The precision of this predictive model is crucial, as it enhances Wells Fargo's resilience during challenging market conditions and ensures sustainable long-term profitability.

The dataset selected for analysis comprises Freddie Mac's 30-year fixed-rate mortgage data spanning from 1999 to 2023. Both origination and performance data will be leveraged throughout the model development process. It is also worthwhile to acknowledge the presence of quality issues within this dataset that could potentially impact the robustness of predictions. Null values and missing data points pose challenges, potentially leading to inaccurate or incomplete interpretations of mortgage performance. Additionally, the model will integrate forecasted macroeconomic factors, such as unemployment rates and the House Price Index, to provide comprehensive insights, which are subject to change due to unforeseen circumstances.

Key stakeholders for this project include the Wells Fargo team, the U.S. government, and homebuyers. Wells Fargo must ensure that their mortgage-backed securities are worthwhile to the customer in this competitive market, while also maintaining profitability for the business. Government agencies, like the Federal Reserve and the Federal Housing Finance Agency, have special interest in these activities to regulate mortgage lending and ensure economic stability. The government also holds responsibility for ensuring the accuracy of default rates, particularly in scenarios where banks may require bailouts. Lastly, the homebuyer is of the utmost importance, as customers rely on lenders to provide a fair and transparent service. Understanding default rates can also impact a homebuyer’s loan approval process, overall interest rates, and affordability.

## Methodology (Data Analyzed)
The Freddie Mac dataset outlines Single-Family Loan Level Factors that may contribute to defaulting on a mortgage. Among Americans, the two most popular mortgage types are 15 and 30-year fixed rates for their varying interest rates and monthly payments which provide homeowners with financial flexibility. For the model, the 30-year amortized fixed-rate mortgage product will be assessed due to its high popularity and low default rate (< 0.2%) during normal economic times. By modeling the 30-year mortgage as opposed to the 15-year, Wells Fargo will have an upper hand in predicting the possibility of future default for one of the most popular mortgage types, overall preventing the risk of reputation impacts and saving the company money. 

The dataset consists of both an origination and monthly performance dataset for each year spanning 1999 to 2023. The origination file contains loan-level origination information such as loan term, purpose, original loan-to-value (LTV), and debt-to-income (DTI) ratio. The performance file uses a monthly format to outline loan-level performance and actual loss data for each respective loan such as total expenses, loan age, and current actual unpaid balance (UPB). Both data files are connected by a unique Loan Sequence Number associated with each loan, which will connect data to congruent information. 

While the current Freddie Mac dataset is a comprehensive source of loan-specific data,  it fails to take into account macroeconomic factors that have large impacts on the housing market, leading to higher or lower default rates. The country’s unemployment rate, FHFA house price index, and inflation rate, among other factors, will be used to give an economy-wide influence on the current dataset. Since unemployment leads to lowered income, understanding the rate of Americans who are unemployed but actively looking to find work may be an indicator of difficulty in making a mortgage payment and overall default. Similarly, the FHFA Housing Price Index may be a leading indicator of future default as it measures average price changes in property sales or refinancings. An increase in HPI can influence an inability or undesire to fully pay a mortgage due to increased property prices. Finally, the country’s inflation rate directly impacts mortgage rates and thus, default rates. Since inflation erodes consumer buying power due to price increases, banks may indirectly increase their mortgage rates to reduce financial losses. With higher rates coupled with the loss of purchasing power, the change of default for homeowning consumers may rise. By injecting economic data into loan-level information, the model will be able to recognize indirect factors outside of the loan itself that impact the payment capacity of homeowners with a Wells Fargo 30-year amortized fixed-rate mortgage.  

Since mortgages themselves are time-based loans, national interest, unemployment, and other rates are highly influential. To ensure a proper default prediction, the team plans to use Freddie Mac loan-level data paired with macroeconomic data from its time period as indicators of how the loan can perform in the future. With macroeconomic factors that are specific to time, loan performance can be interpreted as a function of the loan itself and environmental conditions created by the economic period. With macroeconomic factors implemented into the data, the model will also be able to predict default rates during periods of economic recession. Periods of crisis, such as the 2007-2008 Housing Crisis or the COVID-19 Pandemic, have direct impacts on the country’s macroeconomic measures. Thus, by instituting such factors in the data, the inherent imbalance of normal default rates of < 0.2% and crisis period default rates as high as 10% will be accounted for. 

First, because the Freddie Mac dataset consists of individual quarterly information across the 24-year 1999 to 2023 time period, it was too large to download and work with on the existing computer systems available to the team. As a result, the team decided to use the sample data provided by Freddie Mac which consists of a simple random sample of 50,000 loans from the original dataset for each year. This makes the entire dataset more functional for preprocessing and modeling techniques. Secondly, the team used Jupyter Notebook to create both the origination and performance datasets for every year. To to this, each corresponding “.txt” file was downloaded and loaded into the workspace using “|” as the delimiter to specific columns. Column names were then added to each dataset using the user guide found in the Freddie Mac Read_Me file. The individual years from which the loans originated were also added to both datasets as a way to understand how macroeconomic features respective to time affected default rates. Of the full data for each origination and performance file, the type of amortization for all loans was also checked to ensure they were fixed-rate mortgages (FRM).

In the upcoming week, the team will use research on individual variables, as well as input from professionals in the sector, to understand the data’s variables with NULL values. Since all NULL values should not be mindlessly removed, we must determine which variables are important influencing factors in default rates regardless of their NULL inputs. In some cases, a NULL variable may not represent a missing value, but instead may signify that the value is not “Yes.” In this sense, the team will change the NULL values to read as their correct values as explained in the data dictionary section of the user guide. Additionally, we will research the impact of the normalization of numerical values in our data, to prevent certain variables like Original UPB from outweighing others in the model such as Original Interest Rate. Finally, explanatory data analysis will be conducted to explore patterns or an underlying structure of the dataset. For instance, a correlation heatmap can be used to find connections between if there are indirect effects of variables against each other. While there is still work to be done to fully grasp the story that this dataset is trying to tell, the team has a plan to ensure successful preprocessing and exploration of the data before modeling. 

## Methodology: Analytics Techniques
We will utilize PiML, a Python toolbox for interpretable machine learning, to model mortgage default and prepayment rates. There are 9 different interpretable modeling approaches we can explore: 

- GLM(Generalized Linear Models)L: GLM, a flexible extension of ordinary linear regression, generalizes the linear model by incorporating a link function connecting it to the response variable. Moreover, GLMs permit the variance of each measurement to be influenced by its predicted value. This adaptability makes GLMs well-suited for handling response variables that deviate from normal distribution, and they can employ various link functions to model diverse relationships. GLM encompasses linear regression, logistic regression, Poisson regression, among other types. It is commonly used for risk assessment and predicting probabilities
  - Advantages:   
    - Interpretability (Very easy to understand and explain)
  - Disadvantage: Limited flexibility
    - Lower performance in comparison to other models 
    - Tends to focus only on linear relationships
    - Unable to detect nonlinearity directly

- GAM (Generalized Additive Model): GAM extends GLMs by allowing for non-linear relationships between features and the response variable. It is formed by the sum of many splines, resulting in a highly flexible model that retains some of the explainability of linear regression. This flexibility is particularly useful when capturing complex patterns in the data where a linear model may be inadequate
  - Advantages:
    - GAMs offer high flexibility in modeling non-linear relationships within the data
    - The additivity of GAMs allows for the interpretation of each predictor's contribution while keeping other predictors fixed
    - GAMs have the potential to outperform linear models, particularly in predictive accuracy.
  - Disadvantages:
    - Highly non-linear models, such as those generated by GAMs with numerous splines, may become complex, making overall model interpretation challenging

- XGB1(XGBoost Depth 1): XGBoost Depth 1 (XGB1) is a tree-based model well-suited for both regression and classification tasks. This 	specific variant of XGBoost limits the maximum depth of the tree to 1, commonly known as boosted stumps. XGB1 stands 
out for its interpretability, as it can be understood as a GAM with piecewise constant main effect
  - Advantages:   
    - High accuracy and efficiency in handling large datasets
  - Disadvantages: 
    - Limited modeling capacity 

- XGB2(XGBoost Depth 2): Similar to XGB1, XGB2 refers to XGBoost with a maximum tree depth of 2. This configuration enables the construction of decision trees with a slightly increased complexity, facilitating the modeling of pairwise interactions between features. XGBoost with a depth of 2 strikes a balance between capturing nuanced patterns in the data while maintaining computational efficiency, making it suitable for scenarios where a moderate level of model complexity is desired
  - Advantages:   
    - High accuracy and efficiency in handling large datasets
  - Disadvantages: 
    - Limited modeling capacity 

- EBM(Explainable Boosting Machines): The model seamlessly integrates the interpretability of GAM with the predictive strength of boosting algorithms, ensuring both high predictive performance and interpretability, making it particularly suited for applications where understanding the model’s decisions is paramount. In contrast to GAM models, EBM models possess the ability to automatically detect and incorporate pairwise interaction terms 
  - Advantages:   
    - Better predictive power 
    - One of the fastest models to execute at prediction time 
    - Light memory usage
    - Fast computation 
    - Nice visualization 
    - Good support from Microsoft Research 
  - Disadvantages: 
    - Runs slower than other models
    - Non-smooth and jumpy shape functions
    - Lacking monotonicity constraint
    - Lacking pruning for main effects

- GAMI_NetL: GAMI-Net is similar to GAM in capturing non-linear patterns but different in how it deals with interactions between variables. While both can become less interpretable with complex patterns, GAMI-Net uses a neural network approach to handle interactions, allowing it to understand complex relationships between variables. It's like a special kind of network designed to focus on specific effects or interactions, making it easier to understand and interpret 
  - Advantages:   
    - Effectively model complex non-linear interactions between variables, enhancing the model's flexibility and its ability to capture intricate patterns in the data.
    - Allows it to focus on specific main effects or pairwise interactions, leading to a more accurate understanding of individual aspects of relationships in the data.
  - Disadvantages: 
    - Computationally intensive
    - It can be a non-interoperable model depending on feature selection due to its neural network approach

- ReLU-DNN(ReLU Neural Network): The ReLU-DNN model is a variant of deep neural networks, incorporating the Rectified Linear Unit (ReLU) activation function in its hidden layers. ReLU allows the model to capture complex, non-linear relationships between input data and output labels. During training, the network adjusts its weights using optimization algorithms to minimize a loss function, measuring the disparity between predicted outputs and actual labels. ReLU activation functions in deep neural networks have proven highly successful, attributed to their simple form, providing advantages like rapid convergence, superior predictive performance, and inherent interpretability.
  - Advantages:   
    - Excel in learning non-linear relationships between inputs and outputs.
    - Well-suited for handling large-scale and complex datasets.
    - Demonstrates relatively fast convergence during the training process.
    - Successfully mitigates the vanishing gradient problem often encountered in deep neural networks that use other activation functions.
    - Achieved state-of-the-art performance in image classification, speech recognition, and natural language processing.
    - Simple functional form contributes to the model's simplicity, making it easier to implement and understand.
  - Disadvantages: 
    - Demand substantial amounts of training data and computational resources.
    - Can be sensitive to the choice of hyperparameters, including the learning rate and regularization strength.
    - Interpreting and understanding the learned representations in ReLU-DNN can be challenging

- FIGS(Fast Interpretable Greedy-tree Sums): Fast Interpretable Greedy-Tree Sums (FIGS), introduced recently, is a machine-learning algorithm that extends classification and regression trees (CART). Imagine FIGS as a boosted tree model, with the final FIGS model comprised of multiple trees (predictor variables) added iteratively, fully considering the split for an ensemble of trees during the model-building process. The predictions of a trained FIGS model are derived by combining the predictions of all these individual trees 
  - Advantages:   
    - Interpretable for Relatively Small-Sized Trees
    - More expressive model form with ensemble of multiple trees
    - Flexible (used for both classification and regression)
    - The flexibility to either boost a new tree or grow existing trees, depending on whichever reduces the loss most
    - Relatively Robust Against Noise
    - More Stable than CART models
    - Larger search space compared to boosted trees, given the same number of split iterations
    - Decouples Feature Interactions

  - Disadvantages: 
    - Less Interpretable than CART Models (Difficult to Follow Multiple Separate Trees)
    - Prone to Overfitting
    - High Variance
   
After completing sampling and data cleaning, the team will utilize a regression model to identify the primary factors influencing the default and prepayment rates in the dataset provided by Freddie Mac. Subsequently, we will integrate these factors with macroeconomic variables, including the unemployment rate and pricing index. The following is a list of variables we are currently considering for our predictive modeling of default rate

- For Non-Categorical Variables:
  - CLTV (Combined Loan-to-Value): Higher CLTV ratios indicate that borrowers have relatively less equity in their homes, making them more vulnerable to financial difficulties. This increases the risk of default as borrowers may find it challenging to sell the property or refinance in case of financial stress.
  - ELTV (Estimated Loan-to-Value): Most predictive value for default rate. Similar to CLTV, a higher LTV ratio implies a higher loan amount compared to the property value. This increases the risk of default, as borrowers with less equity have less financial cushion in the event of economic downturns or changes in personal financial situations. The team will estimate the Estimated Loan-to-Value (LTV) by dividing the Current Actual Unpaid Principal Balance (UPB) by the Housing Price, calculated as the percentage change in the Home Price Index (HPI) from origination to the month we are predicting
  - DTI (Debt-to-Income) Ratio: A higher DTI ratio suggests that a larger portion of the borrower's income is committed to debt payments. This increases the likelihood of financial strain, making it more difficult for borrowers to meet mortgage obligations and raising the risk of default.
  - Unemployment Rate: The unemployment rate is a macroeconomic indicator that can impact borrowers' ability to make mortgage payments. High unemployment rates may lead to increased default rates as borrowers may face job losses and income reduction.
  - Original Interest Rate: Higher interest rates increase the cost of borrowing, potentially straining the financial capacity of borrowers. This can lead to an increased risk of default as borrowers may struggle to meet higher monthly payments.
  - Current Loan Delinquency Status: A value corresponding to the number of days the borrower is delinquent. Delinquency is a strong predictor of default, as borrowers who are behind on payments are more likely to face challenges in meeting future obligations. ( 0 = Current, or less than, 30 days delinquent, 1 = 30-59 days delinquent, 2 = 60 – 89 days delinquent, 3 = 90 – 119 days, delinquent, and so on, RA = REO Acquisition.
 
- For Categorical Variables:
  - First Home-Buyer Indicator: First-time homebuyers may have different risk profiles than repeat buyers. Their financial stability and experience in homeownership can impact default and prepayment rates.
  - Loan Purpose Indicator: The purpose of the loan can influence borrower behavior. Loans for different purposes may have distinct default and prepayment patterns.
  - Credit Score: Credit score is a crucial indicator of a borrower's creditworthiness. Lower credit scores are associated with higher default risk, and credit score can also influence prepayment behavior.
  - Year of Origination for the Loan: Economic conditions and lending standards can vary over time. The year of origination captures the vintage of the loan and can be a proxy for economic cycles and market conditions.
  - Occupation Indicator: The borrower's occupation may provide insights into income stability and the likelihood of job loss, which can impact default and prepayment rates.
  - State: Housing market conditions, economic factors, and regional variations can significantly influence default and prepayment rates. Analyzing performance at the state level helps capture these localized effects.
  - Modification Flag:  A flag indicating if the loan has been modified in the current period or a prior period. Modifying a loan can be a response to financial distress on the part of the borrower, and such modifications may influence the likelihood of default

To assess our models, the team will employ distinct metrics gauging accuracy and robustness. The evaluation of accuracy involves analyzing Mean Squared Error (MSE) and R-squared (R2). MSE  measures the average squared difference between the predicted and actual values. It provides a measure of the overall model performance with lower values indicating better accuracy. R2, a statistical metric, indicates the proportion of variance in the target variable explainable by the independent variables. R2 values range from 0 to 1, with 1 indicating a perfect fit and 0 indicating no discernible relationship between variables within the training set. Concerning robustness, the impact of model performance under distribution shifts is a crucial consideration. Unforeseen shift in data distribution can lead to incorrect predictions, particularly in the face of unexpected economic conditions. To gauge the model's resilience to such shifts, the team will utilize PIML to assess the model’s robustness to input perturbations.
For stress testing, the team will compare how default and prepayment rates change under a stressed economic environment with respect to different scenarios. This will help the team determine the sensitivity of mortgage default rates to various economic conditions. The selected crises for analysis are as follows: 

- Crisis 1: The Dot-com Recession (March 2001–November 2001)
- Crisis 2: The Great Recession (December 2007–June 2009)
- Crisis 3: The COVID-19 Recession (February 2020–April 2020)
- Crisis 4: Silicon Valley Bank (March 2023–April 2023)

For predictive modeling focused on mortgage default and prepayment rates, bias testing is not a concern at the moment, as our objective is to predict portfolio performance to assess whether the bank will sustain its operations. However, the complexity of the modeling task arises from the need to predict defaults for the upcoming month and the subsequent two months up to 24 months

## Results/Conclusions/Recommendations: Outlined
- Presentation of the predicted default rates for Wells Fargo over the next 24 months.
- Comparison of the model's performance against baseline benchmarks and industry standards.
- Interpretation of the findings, including insights into the impact of macroeconomic factors on default rates.
- Recommendations for Wells Fargo, such as proactive risk management strategies or adjustments to lending practices based on the predictions.



