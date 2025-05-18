# Solution to Lab 4.10.1: European Stock Indices Analysis

## Problem 1: Analysis of Time Series Plots of Four European Stock Indices

After examining the time series plots of the four European stock indices (DAX, SMI, CAC, and FTSE) from 1991 to 1998, I can provide the following analysis:

### Stationarity Assessment
None of the four indices exhibit stationarity. All indices show clear trends over the time period:

- **DAX (German)**: Shows a strong upward trend throughout the period, particularly accelerating after 1995, reaching peaks around 7000 in 1998.
- **SMI (Swiss)**: Displays an upward trend until 1994, followed by a decline and then stabilization around 2000-2500.
- **CAC (French)**: Shows an upward trend until 1994, followed by a significant decline until 1996, and then a partial recovery.
- **FTSE (UK)**: Demonstrates a consistent upward trend throughout the period, with particularly strong growth in 1998.

### Volatility Characteristics
The fluctuations in the series are not of constant size:

- All indices show evidence of **volatility clustering**, where periods of high volatility are followed by more high volatility.
- The magnitude of fluctuations tends to increase as the index values increase, suggesting that percentage changes might be more stable than absolute changes.
- There appears to be increased volatility around 1997-1998, possibly related to the Asian financial crisis.
- The DAX index exhibits the most pronounced increase in volatility over time, particularly during its rapid growth phase after 1995.

### Co-movement Patterns
There is noticeable co-movement among the indices, suggesting correlation between European markets, though each has its own distinct pattern. This indicates both common European economic factors and country-specific influences affecting these markets.

## Problem 2: Analysis of Log Returns on the Four Indices

The time series plots of the log returns for the four European stock indices reveal the following characteristics:

### Stationarity Assessment
The log returns appear much more stationary than the original price series:

- All four indices' log returns fluctuate around a mean close to zero.
- There are no obvious trends in the log returns series.
- This stationarity in the mean is a typical characteristic of financial returns and confirms that the log transformation has effectively addressed the non-stationarity present in the original price series.

### Volatility Characteristics
The fluctuations in the log returns are not of constant size over time:

- Clear evidence of **volatility clustering** is present in all four indices.
- Periods of high volatility (wider fluctuations) alternate with periods of lower volatility (narrower fluctuations).
- Increased volatility is noticeable around 1997-1998, likely corresponding to the Asian financial crisis.
- The volatility appears more constant in magnitude compared to the original series, but still exhibits time-varying characteristics.

### Distributional Properties
The log returns exhibit several important distributional characteristics:

- All series show occasional extreme returns (both positive and negative), which appear as spikes in the time series plots.
- These extreme returns occur more frequently than would be expected under a normal distribution, suggesting heavy-tailed distributions.
- The returns show a tendency to revert to the mean (around zero), consistent with the efficient market hypothesis in its weak form.

### Normal Probability Plots and Shapiro-Wilk Test Results

The normal Q-Q plots for all four indices show that the log returns follow a relatively normal distribution in the central part but deviate in the tails. This suggests heavier tails than a normal distribution, which is consistent with financial time series data.

The Shapiro-Wilk test results for the log returns are:
- DAX: W = 0.999586, p-value = 0.9568926812
- SMI: W = 0.999661, p-value = 0.9869562351
- CAC: W = 0.999306, p-value = 0.6550039695
- FTSE: W = 0.999449, p-value = 0.8377937763

Interestingly, the p-values are all greater than 0.05, suggesting that we cannot reject the null hypothesis of normality at the 5% significance level. This is somewhat surprising given the visual evidence of heavy tails in the Q-Q plots. This discrepancy might be due to the large sample size, which can make the Shapiro-Wilk test less sensitive to deviations in the tails.

## Summary Statistics for Log Returns

|                | DAX          | SMI          | CAC          | FTSE         |
|----------------|--------------|--------------|--------------|--------------|
| count          | 2087.000000  | 2087.000000  | 2087.000000  | 2087.000000  |
| mean           | 0.000583     | 0.000162     | -0.000065    | 0.000530     |
| std            | 0.012057     | 0.010072     | 0.011104     | 0.009195     |
| min            | -0.046609    | -0.036260    | -0.035994    | -0.033345    |
| 25%            | -0.007375    | -0.006744    | -0.007342    | -0.005618    |
| 50%            | 0.000671     | 0.000159     | -0.000053    | 0.000592     |
| 75%            | 0.008777     | 0.006976     | 0.007309     | 0.006666     |
| max            | 0.040209     | 0.038189     | 0.034375     | 0.035112     |
| skewness       | -0.046257    | 0.014705     | 0.004889     | -0.052267    |
| kurtosis       | 0.041656     | 0.066010     | 0.047539     | 0.068792     |

## Comprehensive Analysis and Implications

### Price Series Characteristics
- All four indices show distinct trends, with the DAX and FTSE exhibiting the strongest upward movements.
- The series are non-stationary with increasing volatility over time.
- There is significant co-movement among the indices, suggesting correlation between European markets.

### Log Returns Characteristics
- The log returns are stationary with means close to zero.
- Volatility clustering is evident in all series.
- The DAX shows the highest volatility (std = 0.012057), while the FTSE shows the lowest (std = 0.009195).

### Distributional Properties
- The log returns appear approximately normally distributed but with slightly heavier tails.
- The skewness values are close to zero, indicating relatively symmetric distributions.
- The kurtosis values are slightly positive, confirming the presence of heavier tails than a normal distribution.

### Implications for Financial Modeling
- The time-varying volatility suggests that GARCH-type models would be appropriate for modeling these returns.
- The correlation between indices indicates potential benefits from international diversification, but also suggests common systematic risk factors affecting European markets.
- The relatively normal distribution of returns with slightly heavier tails suggests that risk models should account for the possibility of extreme events.

This analysis provides a foundation for more sophisticated modeling approaches, such as multivariate GARCH models or copula-based dependence modeling, that could capture the changing dynamics of these European stock markets.
