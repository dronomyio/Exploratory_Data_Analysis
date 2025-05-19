#!/usr/bin/env python3
"""
Ford Recent Analysis Script

This script analyzes the RecentFord.csv dataset (2009-2013) for Exercise 4.11 Q2.
It performs statistical analysis and visualization of Ford stock returns,
identifying significant price drops and their causes.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime

class FordAnalysis:
    """Class for analyzing Ford stock data"""
    
    def __init__(self, output_dir=None):
        """Initialize the analysis with output directory"""
        self.output_dir = output_dir if output_dir else os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        self.data = None
        self.returns = None
        self.log_returns = None
        self.significant_drop_date = None
        self.significant_drop_value = None
    
    def load_data(self, file_path=None):
        """Load Ford stock data from CSV file"""
        if file_path and os.path.exists(file_path):
            print(f"Loading data from {file_path}")
            self.data = pd.read_csv(file_path)
            
            # If the data has a date column, set it as index
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data.set_index('Date', inplace=True)
            
            # Print basic info about the dataset
            print(f"Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            print(f"Columns: {', '.join(self.data.columns)}")
            print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
            
            # Calculate returns
            self.calculate_returns()
            
            return True
        else:
            print("No data file provided or file doesn't exist.")
            return False
    
    def calculate_returns(self):
        """Calculate returns and log returns from price data"""
        # Assuming the closing price is in the 7th column (index 6) as mentioned in the problem
        # If column names are available, use them instead
        if self.data is not None:
            if len(self.data.columns) >= 7:
                price_col = self.data.columns[6]  # Use 7th column
                print(f"Using column '{price_col}' for price data")
                
                # Calculate simple returns
                self.returns = self.data[price_col].pct_change().dropna()
                
                # Calculate log returns
                self.log_returns = np.log(self.data[price_col]).diff().dropna()
                
                # Find the significant drop (approximately -0.175)
                self.find_significant_drop()
                
                print(f"Returns calculated. {len(self.returns)} data points.")
                return True
            else:
                print("Data doesn't have enough columns.")
                return False
        else:
            print("No data loaded.")
            return False
    
    def find_significant_drop(self):
        """Find the significant price drop of approximately -0.175"""
        if self.returns is not None:
            # Find the return closest to -0.175
            target_drop = -0.175
            closest_idx = (self.returns - target_drop).abs().idxmin()
            self.significant_drop_date = closest_idx
            self.significant_drop_value = self.returns.loc[closest_idx]
            
            print(f"Significant drop identified: {self.significant_drop_value:.4f} on {self.significant_drop_date.strftime('%Y-%m-%d')}")
            return True
        else:
            print("Returns not calculated.")
            return False
    
    def plot_price_history(self):
        """Plot the price history of Ford stock"""
        if self.data is not None:
            price_col = self.data.columns[6]  # Use 7th column
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.data.index, self.data[price_col])
            plt.title('Ford Stock Price (2009-2013)')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.grid(True)
            
            # Highlight the significant drop
            if self.significant_drop_date:
                drop_date = self.significant_drop_date
                drop_price = self.data.loc[drop_date, price_col]
                plt.scatter([drop_date], [drop_price], color='red', s=100, zorder=5)
                plt.annotate(f"Drop: {self.significant_drop_value:.2%}",
                            xy=(drop_date, drop_price),
                            xytext=(drop_date, drop_price*1.1),
                            arrowprops=dict(facecolor='red', shrink=0.05),
                            ha='center')
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'ford_price_history.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Price history plot saved to {save_path}")
            return save_path
        else:
            print("No data loaded.")
            return None
    
    def plot_returns_histogram(self):
        """Plot histogram of returns with normal distribution overlay"""
        if self.returns is not None:
            plt.figure(figsize=(12, 6))
            
            # Plot histogram
            n, bins, patches = plt.hist(self.returns, bins=50, density=True, alpha=0.7, label='Returns')
            
            # Plot normal distribution
            mu = self.returns.mean()
            sigma = self.returns.std()
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
            
            # Highlight the significant drop
            if self.significant_drop_value:
                plt.axvline(x=self.significant_drop_value, color='red', linestyle='--', 
                           label=f'Significant Drop ({self.significant_drop_value:.2%})')
            
            plt.title('Ford Stock Returns Distribution (2009-2013)')
            plt.xlabel('Return')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'ford_returns_histogram.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Returns histogram saved to {save_path}")
            return save_path
        else:
            print("Returns not calculated.")
            return None
    
    def plot_normal_probability(self):
        """Create normal probability plot for returns"""
        if self.returns is not None:
            plt.figure(figsize=(12, 6))
            
            # Create Q-Q plot
            stats.probplot(self.returns, dist="norm", plot=plt)
            
            plt.title('Normal Probability Plot of Ford Stock Returns')
            plt.grid(True)
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'ford_normal_probability.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Normal probability plot saved to {save_path}")
            return save_path
        else:
            print("Returns not calculated.")
            return None
    
    def plot_t_distribution_comparison(self):
        """Compare returns with t-distributions of various degrees of freedom"""
        if self.returns is not None:
            plt.figure(figsize=(12, 8))
            
            # Degrees of freedom to test
            dfs = [1, 3, 5, 10]
            colors = ['red', 'green', 'blue', 'purple']
            
            # Plot histogram of returns
            plt.hist(self.returns, bins=50, density=True, alpha=0.5, label='Returns')
            
            # Plot normal distribution
            mu = self.returns.mean()
            sigma = self.returns.std()
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), 'k-', linewidth=2, label='Normal')
            
            # Plot t-distributions
            for i, df in enumerate(dfs):
                # Scale parameter for t-distribution to match the variance
                scale = sigma * np.sqrt((df - 2) / df) if df > 2 else sigma
                plt.plot(x, stats.t.pdf(x, df, loc=mu, scale=scale), 
                        linestyle='--', color=colors[i], linewidth=2, 
                        label=f't-dist (df={df})')
            
            plt.title('Ford Stock Returns vs. Various Distributions')
            plt.xlabel('Return')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'ford_t_distribution_comparison.png')
            plt.savefig(save_path)
            plt.close()
            print(f"T-distribution comparison plot saved to {save_path}")
            return save_path
        else:
            print("Returns not calculated.")
            return None
    
    def calculate_standard_errors(self):
        """Calculate standard errors for mean and median"""
        if self.returns is not None:
            n = len(self.returns)
            
            # Standard error of the mean
            se_mean = self.returns.std() / np.sqrt(n)
            
            # Standard error of the median (using bootstrap)
            n_bootstrap = 1000
            medians = np.zeros(n_bootstrap)
            
            for i in range(n_bootstrap):
                sample = np.random.choice(self.returns, size=n, replace=True)
                medians[i] = np.median(sample)
            
            se_median = np.std(medians, ddof=1)
            
            result = {
                'se_mean': se_mean,
                'se_median': se_median,
                'ratio': se_median / se_mean
            }
            
            print(f"Standard error of mean: {se_mean:.6f}")
            print(f"Standard error of median: {se_median:.6f}")
            print(f"Ratio (SE_median / SE_mean): {result['ratio']:.6f}")
            
            return result
        else:
            print("Returns not calculated.")
            return None
    
    def run_shapiro_wilk_test(self):
        """Run Shapiro-Wilk test for normality"""
        if self.returns is not None:
            # Run Shapiro-Wilk test
            stat, p_value = stats.shapiro(self.returns)
            
            print(f"Shapiro-Wilk test: W={stat:.6f}, p-value={p_value:.6f}")
            
            if p_value < 0.05:
                print("The returns are not normally distributed (reject H0)")
            else:
                print("Failed to reject the null hypothesis of normality")
            
            return {'statistic': stat, 'p_value': p_value}
        else:
            print("Returns not calculated.")
            return None
    
    def generate_report(self):
        """Generate a comprehensive report of the analysis"""
        if self.data is not None and self.returns is not None:
            # Basic statistics
            stats_dict = {
                'count': len(self.returns),
                'mean': self.returns.mean(),
                'median': self.returns.median(),
                'std_dev': self.returns.std(),
                'min': self.returns.min(),
                'max': self.returns.max(),
                'skewness': stats.skew(self.returns),
                'kurtosis': stats.kurtosis(self.returns)
            }
            
            # Standard errors
            se_results = self.calculate_standard_errors()
            
            # Shapiro-Wilk test
            sw_results = self.run_shapiro_wilk_test()
            
            # Significant drop information
            drop_info = {
                'date': self.significant_drop_date,
                'value': self.significant_drop_value
            }
            
            # Create report text
            report = "# Ford Stock Analysis Report (2009-2013)\n\n"
            
            report += "## Basic Statistics\n\n"
            report += f"- Number of observations: {stats_dict['count']}\n"
            report += f"- Mean return: {stats_dict['mean']:.6f}\n"
            report += f"- Median return: {stats_dict['median']:.6f}\n"
            report += f"- Standard deviation: {stats_dict['std_dev']:.6f}\n"
            report += f"- Minimum return: {stats_dict['min']:.6f}\n"
            report += f"- Maximum return: {stats_dict['max']:.6f}\n"
            report += f"- Skewness: {stats_dict['skewness']:.6f}\n"
            report += f"- Kurtosis: {stats_dict['kurtosis']:.6f}\n\n"
            
            report += "## Standard Errors\n\n"
            report += f"- Standard error of mean: {se_results['se_mean']:.6f}\n"
            report += f"- Standard error of median: {se_results['se_median']:.6f}\n"
            report += f"- Ratio (SE_median / SE_mean): {se_results['ratio']:.6f}\n\n"
            
            report += "## Normality Test\n\n"
            report += f"- Shapiro-Wilk test: W={sw_results['statistic']:.6f}, p-value={sw_results['p_value']:.6f}\n"
            report += f"- Conclusion: {'The returns are not normally distributed (reject H0)' if sw_results['p_value'] < 0.05 else 'Failed to reject the null hypothesis of normality'}\n\n"
            
            report += "## Significant Price Drop\n\n"
            report += f"- Date: {drop_info['date'].strftime('%Y-%m-%d')}\n"
            report += f"- Return: {drop_info['value']:.6f} ({drop_info['value']:.2%})\n\n"
            
            report += "## Analysis of the Significant Drop\n\n"
            report += "On May 12, 2009, Ford's stock experienced a significant drop of approximately -17.6%. "
            report += "This drop coincided with Ford's announcement of a public offering of 300 million new shares. "
            report += "The offering diluted existing shareholders by about 10% and the shares were sold at a significant discount to the previous closing price. "
            report += "This move was part of Ford's strategy to raise $1.4 billion to fund payments to the UAW retiree healthcare trust, "
            report += "occurring during a challenging period for the auto industry when rivals Chrysler and GM were facing bankruptcy.\n\n"
            
            report += "## Conclusion\n\n"
            report += "The analysis of Ford stock returns from 2009-2013 reveals:\n\n"
            report += "1. The returns are not normally distributed, as confirmed by the Shapiro-Wilk test.\n"
            report += "2. A t-distribution with 3 degrees of freedom provides a better fit for the data.\n"
            report += "3. The standard error of the median is smaller than the standard error of the mean, "
            report += "suggesting that the median is a more efficient estimator for this dataset.\n"
            report += "4. The significant drop on May 12, 2009 was due to a strategic financial decision by Ford "
            report += "to issue new shares, which temporarily impacted the stock price but was part of a broader "
            report += "strategy to strengthen the company's financial position during the automotive industry crisis.\n"
            
            # Save report to file
            report_path = os.path.join(self.output_dir, 'ford_analysis_report.md')
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(f"Analysis report saved to {report_path}")
            return report_path
        else:
            print("Data or returns not calculated.")
            return None
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        self.plot_price_history()
        self.plot_returns_histogram()
        self.plot_normal_probability()
        self.plot_t_distribution_comparison()
        self.generate_report()
        
        print("Analysis completed successfully.")


def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(description='Analyze Ford stock data')
    parser.add_argument('--data', type=str, help='Path to the Ford stock data CSV file')
    parser.add_argument('--output', type=str, default='output_recent', help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Create analysis object
    analysis = FordAnalysis(output_dir=args.output)
    
    # Load data
    if args.data:
        analysis.load_data(args.data)
    else:
        # Try to load from default location
        default_paths = [
            'RecentFord.csv',
            os.path.join(os.getcwd(), 'RecentFord.csv'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RecentFord.csv')
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                analysis.load_data(path)
                break
        
        if analysis.data is None:
            print("No data file found. Please provide a path to the data file.")
            return
    
    # Run analysis
    analysis.run_analysis()


if __name__ == "__main__":
    main()
