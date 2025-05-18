#!/usr/bin/env python3
"""
Ford Returns Analysis - Streamlit Web Application
Based on Exercise 4.11 Q1 from Statistics and Data Analysis for Financial Engineering

This web application provides an interactive interface for analyzing Ford stock returns data,
implementing all parts of exercise 4.11 Q1 with additional interactive features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.nonparametric.kde import KDEUnivariate
import os
import io
import base64
from matplotlib.figure import Figure

class FordReturnsAnalyzer:
    """
    A class to analyze Ford stock returns data as specified in Exercise 4.11 Q1.
    """
    
    def __init__(self, data):
        """
        Initialize the analyzer with Ford returns data.
        
        Args:
            data (pd.DataFrame): DataFrame containing Ford returns data
        """
        self.df = data
        self.process_data()
        
    def process_data(self):
        """Process the Ford returns data."""
        try:
            # Extract the returns column
            self.returns = self.df['FORD']
            
            # Convert to numpy array for easier manipulation
            self.returns_array = np.array(self.returns)
            
            # Identify the Black Monday return (October 19, 1987)
            self.black_monday_index = None
            if 'X.m..d..y' in self.df.columns:
                for i, date in enumerate(self.df['X.m..d..y']):
                    if '10/19/1987' in str(date):
                        self.black_monday_index = i
                        self.black_monday_return = self.returns_array[i]
                        break
                
        except Exception as e:
            st.error(f"Error processing data: {e}")
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
    
    def plot_normal_distribution(self):
        """
        Create a normal probability plot (Q-Q plot) of the Ford returns.
        
        Returns:
            fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        res = stats.probplot(self.returns_array, dist="norm", plot=plt)
        plt.title('Normal Q-Q Plot of Ford Returns')
        plt.grid(True)
        
        return fig
    
    def test_normality(self):
        """
        Test for normality using the Shapiro-Wilk test.
        
        Returns:
            tuple: (W statistic, p-value)
        """
        shapiro_test = stats.shapiro(self.returns_array)
        return shapiro_test
    
    def plot_t_distribution(self, df_values=None, exclude_black_monday=False):
        """
        Create t-plots of the Ford returns using various degrees of freedom.
        
        Args:
            df_values (list, optional): List of degrees of freedom values to use.
                                       If None, default values are used.
            exclude_black_monday (bool): Whether to exclude the Black Monday return.
            
        Returns:
            tuple: (fig, linearity_metrics)
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
        
        return fig, linearity_metrics
    
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
    
    def plot_returns_time_series(self):
        """
        Create a time series plot of the Ford returns.
        
        Returns:
            fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the returns
        ax.plot(self.returns_array, linewidth=1)
        
        # Highlight Black Monday if identified
        if self.black_monday_index is not None:
            ax.scatter(self.black_monday_index, self.black_monday_return, 
                      color='red', s=100, zorder=5, label='Black Monday (Oct 19, 1987)')
            
        ax.set_title('Ford Returns Time Series (1984-1991)')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Return')
        ax.grid(True)
        
        if self.black_monday_index is not None:
            ax.legend()
        
        return fig
    
    def plot_returns_histogram(self):
        """
        Create a histogram of the Ford returns with normal and t-distribution overlays.
        
        Returns:
            fig: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate optimal number of bins using Freedman-Diaconis rule
        q75, q25 = np.percentile(self.returns_array, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(self.returns_array) ** (1/3))
        num_bins = int(np.ceil((max(self.returns_array) - min(self.returns_array)) / bin_width))
        
        # Plot histogram
        n, bins, patches = ax.hist(self.returns_array, bins=num_bins, density=True, 
                                  alpha=0.7, label='Returns')
        
        # Add normal distribution curve
        mu, sigma = np.mean(self.returns_array), np.std(self.returns_array)
        x = np.linspace(min(self.returns_array), max(self.returns_array), 1000)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label=f'Normal Distribution\n(μ={mu:.4f}, σ={sigma:.4f})')
        
        # Add t-distribution curve with best fit df
        # Find best df from t-plot analysis
        _, linearity_metrics = self.plot_t_distribution()
        best_df = max(linearity_metrics, key=linearity_metrics.get)
        
        # Scale t-distribution to match data
        scale_factor = sigma * np.sqrt((best_df - 2) / best_df)
        ax.plot(x, stats.t.pdf(x, best_df, loc=mu, scale=scale_factor), 'g--', linewidth=2,
               label=f't-Distribution\n(df={best_df}, loc={mu:.4f}, scale={scale_factor:.4f})')
        
        ax.set_title('Histogram of Ford Returns with Distribution Fits')
        ax.set_xlabel('Return')
        ax.set_ylabel('Density')
        ax.grid(True)
        ax.legend()
        
        return fig


def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="Ford Returns Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Ford Returns Analysis")
    st.subheader("Implementation of Exercise 4.11 Q1 from Statistics and Data Analysis for Financial Engineering")
    
    # Sidebar for file upload and options
    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader("Upload Ford returns CSV file", type=["csv"])
        
        if uploaded_file is None:
            st.info("Please upload a CSV file containing Ford returns data.")
            st.markdown("""
            ### Expected Format:
            The CSV should contain a column named 'FORD' with the returns data.
            Optionally, it can include a date column named 'X.m..d..y'.
            
            ### Sample Data:
            ```
            "","X.m..d..y","FORD"
            "1","2/2/1984",0.02523659
            "2","2/3/1984",-0.03692308
            ...
            ```
            """)
            
            # Option to use sample data
            use_sample = st.checkbox("Use built-in sample data for demonstration", value=False)
            
            if use_sample:
                # Create sample data similar to Ford returns
                np.random.seed(42)
                dates = pd.date_range(start='1984-01-01', periods=2000, freq='B')
                
                # Generate returns with heavy tails to mimic financial data
                returns = np.random.standard_t(df=5, size=2000) * 0.02
                
                # Add a large negative return to simulate Black Monday
                black_monday_idx = 937
                returns[black_monday_idx] = -0.18
                
                # Create DataFrame
                sample_data = pd.DataFrame({
                    'X.m..d..y': dates.strftime('%m/%d/%Y'),
                    'FORD': returns
                })
                
                data = sample_data
                st.success("Using sample data for demonstration.")
            else:
                st.stop()
        else:
            # Read the uploaded file
            data = pd.read_csv(uploaded_file)
            
        # Analysis options
        st.header("Analysis Options")
        exclude_black_monday = st.checkbox("Exclude Black Monday for t-distribution analysis", value=False)
        df_values = st.multiselect(
            "Degrees of freedom for t-distribution",
            options=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30],
            default=[2, 3, 4, 5, 6, 8, 10]
        )
        
        # Download options
        st.header("Download Options")
        download_format = st.selectbox("Report format", ["HTML", "PDF"])
    
    # Initialize analyzer with the data
    try:
        analyzer = FordReturnsAnalyzer(data)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Basic Statistics", 
            "Normality Analysis", 
            "t-Distribution Analysis", 
            "Standard Error Analysis",
            "Additional Visualizations"
        ])
        
        # Tab 1: Basic Statistics
        with tab1:
            st.header("(a) Basic Statistics of Ford Returns")
            
            # Calculate basic statistics
            stats_dict = analyzer.basic_statistics()
            
            # Display statistics in a nice format
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sample Mean", f"{stats_dict['mean']:.6f}")
            col2.metric("Sample Median", f"{stats_dict['median']:.6f}")
            col3.metric("Standard Deviation", f"{stats_dict['std_dev']:.6f}")
            col4.metric("Standard Error of Mean", f"{stats_dict['se_mean']:.6f}")
            
            # Display time series plot
            st.subheader("Ford Returns Time Series")
            fig_ts = analyzer.plot_returns_time_series()
            st.pyplot(fig_ts)
            
            # Display data table
            with st.expander("View Data Table"):
                st.dataframe(data)
        
        # Tab 2: Normality Analysis
        with tab2:
            st.header("(b) Normal Probability Plot")
            
            # Create normal plot
            fig_normal = analyzer.plot_normal_distribution()
            st.pyplot(fig_normal)
            
            st.markdown("""
            ### Interpretation:
            The normal plot shows the relationship between the sample quantiles (Ford returns) and the theoretical quantiles from a normal distribution. 
            
            - If the returns were perfectly normally distributed, all points would fall on the straight line.
            - Deviations from the straight line, particularly in the tails, suggest that the Ford returns are not normally distributed.
            - The returns appear to have heavier tails than a normal distribution would predict, which is common in financial return data.
            """)
            
            st.header("(c) Shapiro-Wilk Test for Normality")
            
            # Perform Shapiro-Wilk test
            shapiro_results = analyzer.test_normality()
            
            # Display results
            col1, col2 = st.columns(2)
            col1.metric("Shapiro-Wilk W statistic", f"{shapiro_results[0]:.6f}")
            col2.metric("p-value", f"{shapiro_results[1]:.10f}")
            
            # Interpretation
            if shapiro_results[1] < 0.01:
                st.success("The p-value is less than 0.01, so we can reject the null hypothesis of a normal distribution at the 0.01 significance level.")
            else:
                st.info("The p-value is greater than 0.01, so we cannot reject the null hypothesis of a normal distribution at the 0.01 significance level.")
            
            # Display histogram with distribution fits
            st.subheader("Returns Distribution")
            fig_hist = analyzer.plot_returns_histogram()
            st.pyplot(fig_his
(Content truncated due to size limit. Use line ranges to read in chunks)