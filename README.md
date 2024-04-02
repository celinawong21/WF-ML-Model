# Wells Fargo Mortgage Default Predictive Model

## Data Preprocessing 
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




