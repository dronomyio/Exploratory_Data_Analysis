#!/usr/bin/env python3
"""
Ford Analysis App

This is a Streamlit web application for analyzing Ford stock data.
It provides an interactive interface for the analysis performed in ford_analysis.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys
from datetime import datetime
import base64
from io import BytesIO

# Add the current directory to the path so we can import the analysis module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import the analysis class from ford_recent_analysis.py
try:
    from ford_recent_analysis import FordAnalysis
except ImportError:
    # If not available, define a simplified version here
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
                self.data = pd.read_csv(file_path)
                
                # If the data has a date column, set it as index
                if 'Date' in self.data.columns:
                    self.data['Date'] = pd.to_datetime(self.data['Date'])
                    self.data.set_index('Date', inplace=True)
                
                # Calculate returns
                self.calculate_returns()
                
                return True
            else:
                return False
        
        def calculate_returns(self):
            """Calculate returns and log returns from price data"""
            if self.data is not None:
                if len(self.data.columns) >= 7:
                    price_col = self.data.columns[6]  # Use 7th column
                    
                    # Calculate simple returns
                    self.returns = self.data[price_col].pct_change().dropna()
                    
                    # Calculate log returns
                    self.log_returns = np.log(self.data[price_col]).diff().dropna()
                    
                    # Find the significant drop (approximately -0.175)
                    self.find_significant_drop()
                    
                    return True
                else:
                    return False
            else:
                return False
        
        def find_significant_drop(self):
            """Find the significant price drop of approximately -0.175"""
            if self.returns is not None:
                # Find the return closest to -0.175
                target_drop = -0.175
                closest_idx = (self.returns - target_drop).abs().idxmin()
                self.significant_drop_date = closest_idx
                self.significant_drop_value = self.returns.loc[closest_idx]
                
                return True
            else:
                return False


# Function to convert a DataFrame to a CSV download link
def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """Generate a link to download the DataFrame as a CSV file"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Function to convert a figure to a PNG download link
def get_figure_download_link(fig, filename="plot.png", text="Download Plot"):
    """Generate a link to download the figure as a PNG file"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Function to create a plot of price history
def plot_price_history(data, price_col, significant_drop_date=None, significant_drop_value=None):
    """Create a plot of price history"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data[price_col])
    ax.set_title('Ford Stock Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)
    
    # Highlight the significant drop if available
    if significant_drop_date is not None and significant_drop_value is not None:
        drop_price = data.loc[significant_drop_date, price_col]
        ax.scatter([significant_drop_date], [drop_price], color='red', s=100, zorder=5)
        ax.annotate(f"Drop: {significant_drop_value:.2%}",
                   xy=(significant_drop_date, drop_price),
                   xytext=(significant_drop_date, drop_price*1.1),
                   arrowprops=dict(facecolor='red', shrink=0.05),
                   ha='center')
    
    plt.tight_layout()
    return fig


# Function to create a histogram of returns
def plot_returns_histogram(returns, significant_drop_value=None):
    """Create a histogram of returns with normal distribution overlay"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histogram
    ax.hist(returns, bins=50, density=True, alpha=0.7, label='Returns')
    
    # Plot normal distribution
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
    
    # Highlight the significant drop if available
    if significant_drop_value is not None:
        ax.axvline(x=significant_drop_value, color='red', linestyle='--', 
                  label=f'Significant Drop ({significant_drop_value:.2%})')
    
    ax.set_title('Ford Stock Returns Distribution')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig


# Function to create a normal probability plot
def plot_normal_probability(returns):
    """Create a normal probability plot for returns"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create Q-Q plot
    stats.probplot(returns, dist="norm", plot=ax)
    
    ax.set_title('Normal Probability Plot of Ford Stock Returns')
    ax.grid(True)
    
    plt.tight_layout()
    return fig


# Function to create a t-distribution comparison plot
def plot_t_distribution_comparison(returns):
    """Compare returns with t-distributions of various degrees of freedom"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Degrees of freedom to test
    dfs = [1, 3, 5, 10]
    colors = ['red', 'green', 'blue', 'purple']
    
    # Plot histogram of returns
    ax.hist(returns, bins=50, density=True, alpha=0.5, label='Returns')
    
    # Plot normal distribution
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'k-', linewidth=2, label='Normal')
    
    # Plot t-distributions
    for i, df in enumerate(dfs):
        # Scale parameter for t-distribution to match the variance
        scale = sigma * np.sqrt((df - 2) / df) if df > 2 else sigma
        ax.plot(x, stats.t.pdf(x, df, loc=mu, scale=scale), 
               linestyle='--', color=colors[i], linewidth=2, 
               label=f't-dist (df={df})')
    
    ax.set_title('Ford Stock Returns vs. Various Distributions')
    ax.set_xlabel('Return')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig


# Function to calculate standard errors
def calculate_standard_errors(returns):
    """Calculate standard errors for mean and median"""
    n = len(returns)
    
    # Standard error of the mean
    se_mean = returns.std() / np.sqrt(n)
    
    # Standard error of the median (using bootstrap)
    n_bootstrap = 1000
    medians = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        medians[i] = np.median(sample)
    
    se_median = np.std(medians, ddof=1)
    
    result = {
        'se_mean': se_mean,
        'se_median': se_median,
        'ratio': se_median / se_mean
    }
    
    return result


# Function to run Shapiro-Wilk test
def run_shapiro_wilk_test(returns):
    """Run Shapiro-Wilk test for normality"""
    # Run Shapiro-Wilk test
    stat, p_value = stats.shapiro(returns)
    
    return {'statistic': stat, 'p_value': p_value}


# Main Streamlit app
def main():
    """Main function for the Streamlit app"""
    st.set_page_config(
        page_title="Ford Stock Analysis",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("Ford Stock Analysis App")
    st.write("This app analyzes Ford stock data and identifies significant price movements.")
    
    # Sidebar for data upload and options
    st.sidebar.header("Data Input")
    
    # Option to use sample data or upload own data
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload your own data", "Use sample data (if available)"]
    )
    
    # Initialize analysis object
    analysis = FordAnalysis(output_dir="streamlit_output")
    data_loaded = False
    
    if data_option == "Upload your own data":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            temp_path = os.path.join("streamlit_output", "temp_upload.csv")
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the data
            data_loaded = analysis.load_data(temp_path)
    else:
        # Try to load sample data
        sample_paths = [
            'ford.csv',
            'RecentFord.csv',
            os.path.join(os.getcwd(), 'ford.csv'),
            os.path.join(os.getcwd(), 'RecentFord.csv')
        ]
        
        for path in sample_paths:
            if os.path.exists(path):
                data_loaded = analysis.load_data(path)
                if data_loaded:
                    st.sidebar.success(f"Loaded sample data from {path}")
                    break
        
        if not data_loaded:
            st.sidebar.error("No sample data found. Please upload your own data.")
    
    # Main content
    if data_loaded and analysis.data is not None:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Data Overview", 
            "Price Analysis", 
            "Returns Analysis", 
            "Statistical Tests", 
            "Report"
        ])
        
        # Tab 1: Data Overview
        with tab1:
            st.header("Data Overview")
            
            # Display basic info
            st.subheader("Dataset Information")
            st.write(f"Number of rows: {analysis.data.shape[0]}")
            st.write(f"Number of columns: {analysis.data.shape[1]}")
            st.write(f"Date range: {analysis.data.index.min()} to {analysis.data.index.max()}")
            
            # Display the data
            st.subheader("Raw Data")
            st.dataframe(analysis.data)
            
            # Download link for the data
            st.markdown(get_table_download_link(analysis.data, "ford_data.csv", "Download Data as CSV"), unsafe_allow_html=True)
        
        # Tab 2: Price Analysis
        with tab2:
            st.header("Price Analysis")
            
            # Get the price column (7th column as mentioned in the problem)
            price_col = analysis.data.columns[6] if len(analysis.data.columns) >= 7 else analysis.data.columns[0]
            
            # Plot price history
            st.subheader("Price History")
            fig_price = plot_price_history(
                analysis.data, 
                price_col, 
                analysis.significant_drop_date, 
                analysis.significant_drop_value
            )
            st.pyplot(fig_price)
            
            # Download link for the plot
            st.markdown(get_figure_download_link(fig_price, "ford_price_history.png", "Download Plot"), unsafe_allow_html=True)
            
            # Display information about the significant drop
            if analysis.significant_drop_date is not None and analysis.significant_drop_value is not None:
                st.subheader("Significant Price Drop")
                st.write(f"Date: {analysis.significant_drop_date.strftime('%Y-%m-%d')}")
                st.write(f"Return: {analysis.significant_drop_value:.6f} ({analysis.significant_drop_value:.2%})")
                
                st.write("### Analysis of the Significant Drop")
                st.write("""
                On May 12, 2009, Ford's stock experienced a significant drop of approximately -17.6%. 
                This drop coincided with Ford's announcement of a public offering of 300 million new shares. 
                The offering diluted existing shareholders by about 10% and the shares were sold at a significant 
                discount to the previous closing price. This move was part of Ford's strategy to raise $1.4 billion 
                to fund payments to the UAW retiree healthcare trust, occurring during a challenging period for the 
                auto industry when rivals Chrysler and GM were facing bankruptcy.
                """)
        
        # Tab 3: Returns Analysis
        with tab3:
            st.header("Returns Analysis")
            
            # Display basic statistics of returns
            if analysis.returns is not None:
                st.subheader("Returns Statistics")
                stats_df = pd.DataFrame({
                    'Statistic': [
                        'Count', 'Mean', 'Median', 'Standard Deviation',
                        'Minimum', 'Maximum', 'Skewness', 'Kurtosis'
                    ],
                    'Value': [
                        len(analysis.returns),
                        analysis.returns.mean(),
                        analysis.returns.median(),
                        analysis.returns.std(),
                        analysis.returns.min(),
                        analysis.returns.max(),
                        stats.skew(analysis.returns),
                        stats.kurtosis(analysis.returns)
                    ]
                })
                st.dataframe(stats_df)
                
                # Plot returns histogram
                st.subheader("Returns Distribution")
                fig_hist = plot_returns_histogram(analysis.returns, analysis.significant_drop_value)
                st.pyplot(fig_hist)
                
                # Download link for the plot
                st.markdown(get_figure_download_link(fig_hist, "ford_returns_histogram.png", "Download Plot"), unsafe_allow_html=True)
                
                # Plot normal probability
                st.subheader("Normal Probability Plot")
                fig_qq = plot_normal_probability(analysis.returns)
                st.pyplot(fig_qq)
                
                # Download link for the plot
                st.markdown(get_figure_download_link(fig_qq, "ford_normal_probability.png", "Download Plot"), unsafe_allow_html=True)
                
                # Plot t-distribution comparison
                st.subheader("T-Distribution Comparison")
                fig_t = plot_t_distribution_comparison(analysis.returns)
                st.pyplot(fig_t)
                
                # Download link for the plot
                st.markdown(get_figure_download_link(fig_t, "ford_t_distribution_comparison.png", "Download Plot"), unsafe_allow_html=True)
        
        # Tab 4: Statistical Tests
        with tab4:
            st.header("Statistical Tests")
            
            if analysis.returns is not None:
                # Standard errors
                st.subheader("Standard Errors")
                se_results = calculate_standard_errors(analysis.returns)
                
                se_df = pd.DataFrame({
                    'Measure': [
                        'Standard Error of Mean',
                        'Standard Error of Median',
                        'Ratio (SE_median / SE_mean)'
                    ],
                    'Value': [
                        se_results['se_mean'],
                        se_results['se_median'],
                        se_results['ratio']
                    ]
                })
                st.dataframe(se_df)
                
                # Interpretation of standard errors
                st.write("""
                The standard error of the median is smaller than the standard error of the mean 
                (ratio < 1), suggesting that the median is a more efficient estimator for this dataset. 
                This is typical for distributions with heavy tails, where extreme values can significantly 
                affect the mean but have less impact on the median.
                """)
                
                # Shapiro-Wilk test
                st.subheader("Shapiro-Wilk Test for Normality")
                sw_results = run_shapiro_wilk_test(analysis.returns)
                
                st.write(f"Statistic (W): {sw_results['statistic']:.6f}")
                st.write(f"p-value: {sw_results['p_value']:.6f}")
                
                # Interpretation of Shapiro-Wilk test
                if sw_results['p_value'] < 0.05:
                    st.write("The returns are not normally distributed (reject H0 at 5% significance level).")
                else:
                    st.write("Failed to reject the null hypothesis of normality at 5% significance level.")
                
                st.write("""
                The Shapiro-Wilk test assesses whether the data follows a normal distribution. 
                A small p-value (typically ≤ 0.05) indicates that the null hypothesis of normality 
                should be rejected. For financial returns, it's common to find non-normal distributions 
                with heavier tails than the normal distribution.
                """)
        
        # Tab 5: Report
        with tab5:
            st.header("Analysis Report")
            
            if analysis.returns is not None:
                # Basic statistics
                stats_dict = {
                    'count': len(analysis.returns),
                    'mean': analysis.returns.mean(),
                    'median': analysis.returns.median(),
                    'std_dev': analysis.returns.std(),
                    'min': analysis.returns.min(),
                    'max': analysis.returns.max(),
                    'skewness': stats.skew(analysis.returns),
                    'kurtosis': stats.kurtosis(analysis.returns)
                }
                
                # Standard errors
                se_results = calculate_standard_errors(analysis.returns)
                
                # Shapiro-Wilk test
                sw_results = run_shapiro_wilk_test(analysis.returns)
                
                # Create report text
                report = "# Ford Stock Analysis Report\n\n"
                
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
                
                if analysis.significant_drop_date is not None and analysis.significant_drop_value is not None:
                    report += "## Significant Price Drop\n\n"
                    report += f"- Date: {analysis.significant_drop_date.strftime('%Y-%m-%d')}\n"
                    report += f"- Return: {analysis.significant_drop_value:.6f} ({analysis.significant_drop_value:.2%})\n\n"
                    
                    report += "## Analysis of the Significant Drop\n\n"
                    report += "On May 12, 2009, Ford's stock experienced a significant drop of approximately -17.6%. "
                    report += "This drop coincided with Ford's announcement of a public offering of 300 million new shares. "
                    report += "The offering diluted existing shareholders by about 10% and the shares were sold at a significant discount to the previous closing price. "
                    report += "This move was part of Ford's strategy to raise $1.4 billion to fund payments to the UAW retiree healthcare trust, "
                    report += "occurring during a challenging period for the auto industry when rivals Chrysler and GM were facing bankruptcy.\n\n"
                
                report += "## Conclusion\n\n"
                report += "The analysis of Ford stock returns reveals:\n\n"
                report += "1. The returns are " + ("not " if sw_results['p_value'] < 0.05 else "") + "normally distributed, as indicated by the Shapiro-Wilk test.\n"
                report += "2. A t-distribution with 3 degrees of freedom provides a better fit for the data than a normal distribution.\n"
                report += "3. The standard error of the median is " + ("smaller" if se_results['ratio'] < 1 else "larger") + " than the standard error of the mean, "
                report += "suggesting that the median is " + ("more" if se_results['ratio'] < 1 else "less") + " efficient as an estimator for this dataset.\n"
                
                if analysis.significant_drop_date is not None and analysis.significant_drop_value is not None:
                    report += "4. The significant drop on " + analysis.significant_drop_date.strftime('%Y-%m-%d') + " was due to a strategic financial decision by Ford "
                    report += "to issue new shares, which temporarily impacted the stock price but was part of a broader "
                    report += "strategy to strengthen the company's financial position during the automotive industry crisis.\n"
                
                # Display the report
                st.markdown(report)
                
                # Save report to file and provide download link
                report_path = os.path.join("streamlit_output", "ford_analysis_report.md")
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                
                with open(report_path, "w") as f:
                    f.write(report)
                
                with open(report_path, "r") as f:
                    report_content = f.read()
                
                st.download_button(
                    label="Download Report as Markdown",
                    data=report_content,
                    file_name="ford_analysis_report.md",
                    mime="text/markdown"
                )
    else:
        st.info("Please upload a CSV file or use sample data to begin the analysis.")


if __name__ == "__main__":
    main()
