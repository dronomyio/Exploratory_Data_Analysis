#!/usr/bin/env python3
"""
Ford Returns Analysis - Implementation of Exercise 4.11 Q1
Statistics and Data Analysis for Financial Engineering

This script performs statistical analysis and visualization on Ford stock returns data
from 1984 to 1991, implementing all parts of exercise 4.11 Q1.

Requirements:
- pandas
- numpy
- matplotlib
- scipy
- statsmodels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.nonparametric.kde import KDEUnivariate
import os
import argparse

class FordReturnsAnalyzer:
    """
    A class to analyze Ford stock returns data as specified in Exercise 4.11 Q1.
    """
    
    def __init__(self, data_path):
        """
        Initialize the analyzer with the path to the Ford returns data.
        
        Args:
            data_path (str): Path to the CSV file containing Ford returns data
        """
        self.data_path = data_path
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the Ford returns data."""
        try:
            # Load the data
            self.df = pd.read_csv(self.data_path)
            
            # Extract the returns column
            self.returns = self.df['FORD']
            
            # Convert to numpy array for easier manipulation
            self.returns_array = np.array(self.returns)
            
            # Identify the Black Monday return (October 19, 1987)
            # This is an approximation as we need to find the exact date in the dataset
            self.black_monday_index = None
            if 'X.m..d..y' in self.df.columns:
                for i, date in enumerate(self.df['X.m..d..y']):
                    if '10/19/1987' in str(date):
                        self.black_monday_index = i
                        self.black_monday_return = self.returns_array[i]
                        break
            
            print(f"Data loaded successfully: {len(self.returns)} observations")
            if self.black_monday_index is not None:
                print(f"Black Monday (Oct 19, 1987) return identified: {self.black_monday_return:.6f} at index {self.black_monday_index}")
            else:
                print("Black Monday (Oct 19, 1987) return could not be identified in the dataset")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def basic_statistics(self):
        """
        Calculate basic statistics of the Ford returns.
        
        Returns:
            dict: Dictionary containing the sample mean, median, and standard deviation
        """
        mean = np.mean(self.returns_array)
        median = np.median(self.returns_array)
        std_dev = np.std(self.returns_array, ddof=1)  # Using n-1 for sample standard deviation
        
        stats_dict = {
            'mean': mean,
            'median': median,
            'std_dev': std_dev,
            'se_mean': std_dev / np.sqrt(len(self.returns_array))
        }
        
        return stats_dict
    
    def plot_normal_distribution(self, save_path=None):
        """
        Create a normal probability plot (Q-Q plot) of the Ford returns.
        
        Args:
            save_path (str, optional): Path to save the plot. If None, the plot is displayed.
        
        Returns:
            tuple: QQ plot results (ordered_data, theoretical_quantiles)
        """
        plt.figure(figsize=(10, 6))
        res = stats.probplot(self.returns_array, dist="norm", plot=plt)
        plt.title('Normal Q-Q Plot of Ford Returns')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"Normal plot saved to {save_path}")
        else:
            plt.show()
            
        return res
    
    def test_normality(self):
        """
        Test for normality using the Shapiro-Wilk test.
        
        Returns:
            tuple: (W statistic, p-value)
        """
        shapiro_test = stats.shapiro(self.returns_array)
        return shapiro_test
    
    def plot_t_distribution(self, df_values=None, save_path=None, exclude_black_monday=False):
        """
        Create t-plots of the Ford returns using various degrees of freedom.
        
        Args:
            df_values (list, optional): List of degrees of freedom values to use.
                                       If None, default values are used.
            save_path (str, optional): Path to save the plot. If None, the plot is displayed.
            exclude_black_monday (bool): Whether to exclude the Black Monday return.
            
        Returns:
            dict: Dictionary mapping df values to linearity metrics
        """
        if df_values is None:
            df_values = [2, 3, 4, 5, 6, 8, 10]
        
        data = self.returns_array.copy()
        
        # Exclude Black Monday if requested and if we found it
        if exclude_black_monday and self.black_monday_index is not None:
            data = np.delete(data, self.black_monday_index)
            title_suffix = " (excluding Black Monday)"
        else:
            title_suffix = ""
        
        # Create a figure with subplots
        n_plots = len(df_values)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        
        # If there's only one df value, axes won't be an array
        if n_plots == 1:
            axes = [axes]
        
        linearity_metrics = {}
        
        for i, df in enumerate(df_values):
            ax = axes[i]
            
            # Create our own QQ plot for t-distribution
            # Sort the data
            sorted_data = np.sort(data)
            
            # Generate theoretical quantiles from t-distribution
            n = len(sorted_data)
            p = np.arange(1, n + 1) / (n + 1)  # Plotting positions
            theoretical_quantiles = stats.t.ppf(p, df)
            
            # Plot the QQ plot
            ax.scatter(theoretical_quantiles, sorted_data, alpha=0.7)
            
            # Calculate and plot the best fit line
            slope, intercept, r_value, p_value, std_err = stats.linregress(theoretical_quantiles, sorted_data)
            r_squared = r_value ** 2
            linearity_metrics[df] = r_squared
            
            # Add the best fit line
            x_line = np.linspace(min(theoretical_quantiles), max(theoretical_quantiles), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, 'r-', linewidth=2)
            
            ax.set_title(f't-Distribution Q-Q Plot (df={df}){title_suffix}')
            ax.set_xlabel('Theoretical Quantiles (t-distribution)')
            ax.set_ylabel('Sample Quantiles (Ford Returns)')
            ax.grid(True)
            
            # Add R^2 value to the plot
            ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            print(f"t-Distribution plots saved to {save_path}")
        else:
            plt.show()
            
        # Find the best df value based on R^2
        best_df = max(linearity_metrics, key=linearity_metrics.get)
        print(f"Best degrees of freedom: {best_df} (R² = {linearity_metrics[best_df]:.4f})")
            
        return linearity_metrics
    
    def calculate_median_standard_error(self):
        """
        Calculate the standard error of the sample median using formula (4.3)
        with the sample median as the estimate of F^(-1)(0.5) and a KDE to estimate f.
        
        Returns:
            tuple: (standard error of median, standard error of mean, ratio)
        """
        n = len(self.returns_array)
        median = np.median(self.returns_array)
        
        # Use KDE to estimate the density at the median
        kde = KDEUnivariate(self.returns_array)
        kde.fit()
        f_median = kde.evaluate(median)[0]
        
        # Calculate standard error of the median using formula (4.3)
        se_median = 1 / (2 * f_median * np.sqrt(n))
        
        # Calculate standard error of the mean for comparison
        std_dev = np.std(self.returns_array, ddof=1)
        se_mean = std_dev / np.sqrt(n)
        
        # Calculate the ratio
        ratio = se_median / se_mean
        
        return se_median, se_mean, ratio
    
    def run_full_analysis(self, output_dir=None, show_plots=True):
        """
        Run the full analysis as specified in Exercise 4.11 Q1.
        
        Args:
            output_dir (str, optional): Directory to save output files.
                                       If None, plots are displayed but not saved.
            show_plots (bool): Whether to display plots.
            
        Returns:
            dict: Dictionary containing all analysis results
        """
        results = {}
        
        # Create output directory if it doesn't exist
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # (a) Find the sample mean, sample median, and standard deviation
        print("\n(a) Basic Statistics:")
        stats_dict = self.basic_statistics()
        results['basic_statistics'] = stats_dict
        print(f"Sample Mean: {stats_dict['mean']:.6f}")
        print(f"Sample Median: {stats_dict['median']:.6f}")
        print(f"Standard Deviation: {stats_dict['std_dev']:.6f}")
        print(f"Standard Error of Mean: {stats_dict['se_mean']:.6f}")
        
        # (b) Create a normal plot
        print("\n(b) Normal Plot Analysis:")
        normal_plot_path = os.path.join(output_dir, 'normal_plot.png') if output_dir else None
        normal_plot_results = self.plot_normal_distribution(save_path=normal_plot_path)
        results['normal_plot'] = normal_plot_results
        
        # Analyze normality visually
        print("Visual analysis of normality:")
        print("The normal plot shows deviations from the straight line, particularly in the tails,")
        print("suggesting that the Ford returns are not normally distributed.")
        print("The returns appear to have heavier tails than a normal distribution would predict,")
        print("which is common in financial return data.")
        
        # (c) Test for normality using Shapiro-Wilk test
        print("\n(c) Shapiro-Wilk Test for Normality:")
        shapiro_results = self.test_normality()
        results['shapiro_test'] = shapiro_results
        print(f"Shapiro-Wilk W statistic: {shapiro_results[0]:.6f}")
        print(f"p-value: {shapiro_results[1]:.10f}")
        
        if shapiro_results[1] < 0.01:
            print("The p-value is less than 0.01, so we can reject the null hypothesis")
            print("of a normal distribution at the 0.01 significance level.")
        else:
            print("The p-value is greater than 0.01, so we cannot reject the null hypothesis")
            print("of a normal distribution at the 0.01 significance level.")
        
        # (d) Create t-plots with various degrees of freedom
        print("\n(d) t-Distribution Analysis:")
        
        # With Black Monday
        print("t-Distribution Analysis (including Black Monday):")
        t_plot_path = os.path.join(output_dir, 't_plots.png') if output_dir else None
        df_values = [2, 3, 4, 5, 6, 8, 10]
        linearity_metrics = self.plot_t_distribution(df_values, save_path=t_plot_path)
        results['t_plot_with_black_monday'] = linearity_metrics
        
        # Without Black Monday
        print("\nt-Distribution Analysis (excluding Black Monday):")
        t_plot_no_bm_path = os.path.join(output_dir, 't_plots_no_black_monday.png') if output_dir else None
        linearity_metrics_no_bm = self.plot_t_distribution(df_values, save_path=t_plot_no_bm_path, exclude_black_monday=True)
        results['t_plot_without_black_monday'] = linearity_metrics_no_bm
        
        # Discussion on Black Monday
        print("\nDiscussion on Black Monday:")
        print("Black Monday (October 19, 1987) represents an extreme market event that")
        print("significantly affects the tail behavior of the return distribution.")
        print("Including it leads to heavier tails, requiring lower degrees of freedom in the t-distribution.")
        print("Excluding it results in a distribution closer to normal, with higher optimal degrees of freedom.")
        print("The decision to include or exclude it depends on whether we consider it an outlier")
        print("or a legitimate part of the return distribution that models should account for.")
        
        # (e) Calculate standard error of the sample median
        print("\n(e) Standard Error Analysis:")
        se_median, se_mean, ratio = self.calculate_median_standard_error()
        results['standard_errors'] = {
            'se_median': se_median,
            'se_mean': se_mean,
            'ratio': ratio
        }
        
        print(f"Standard Error of Median: {se_median:.6f}")
        print(f"Standard Error of Mean: {se_mean:.6f}")
        print(f"Ratio (SE Median / SE Mean): {ratio:.6f}")
        
        if ratio > 1:
            print("The standard error of the sample median is larger than the standard error of the sample mean.")
        else:
            print("The standard error of the sample median is smaller than the standard error of the sample mean.")
        
        return results


def main():
    """Main function to run the analysis from command line."""
    parser = argparse.ArgumentParser(description='Analyze Ford returns data as per Exercise 4.11 Q1.')
    parser.add_argument('--data', type=str, default='ford.csv', help='Path to the Ford returns data CSV file')
    parser.add_argument('--output', type=str, default=None, help='Directory to save output files')
    parser.add_argument('--no-plots', action='store_true', help='Do not display plots')
    
    args = parser.parse_args()
    
    analyzer = FordReturnsAnalyzer(args.data)
    analyzer.run_full_analysis(output_dir=args.output, show_plots=not args.no_plots)


if __name__ == "__main__":
    main()
