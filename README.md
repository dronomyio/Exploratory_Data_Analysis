# Ford Returns Analysis Application

## Overview
This application provides an interactive interface for analyzing Ford stock returns data based on Exercise 4.11 Q1 from "Statistics and Data Analysis for Financial Engineering". It performs statistical analysis and visualization of Ford stock returns data from 1984 to 1991.

## Features
- Basic statistics calculation (mean, median, standard deviation)
- Normal probability plot generation and analysis
- Shapiro-Wilk test for normality
- t-Distribution analysis with various degrees of freedom
- Standard error analysis for mean and median
- Additional visualizations (time series, histogram, KDE, CDF)
- Interactive web interface with Streamlit
- Report generation in HTML format

## Installation

### Option 1: Using pip
```bash
pip install ford-returns-analysis
```

### Option 2: From source
```bash
git clone https://github.com/yourusername/ford-returns-analysis.git
cd ford-returns-analysis
pip install -e .
```

### Option 3: Using Docker
```bash
docker pull yourusername/ford-returns-analysis
docker run -p 8501:8501 yourusername/ford-returns-analysis
```

## Usage

### Command-line script
```bash
# Basic usage
python ford_analysis.py --data path/to/ford.csv --output path/to/output_dir

# Help
python ford_analysis.py --help
```

### Streamlit web application
```bash
streamlit run ford_analysis_app.py
```

Then open your browser and navigate to http://localhost:8501

## Data Format
The application expects a CSV file with a column named 'FORD' containing the returns data. Optionally, it can include a date column named 'X.m..d..y'.

Example:
```
"","X.m..d..y","FORD"
"1","2/2/1984",0.02523659
"2","2/3/1984",-0.03692308
...
```

## Building the Docker Image
```bash
docker build -t ford-returns-analysis .
```

## Deployment Options

### Local Deployment
Run the Streamlit app locally:
```bash
streamlit run ford_analysis_app.py
```

### Cloud Deployment
The application can be deployed to various cloud platforms:

1. **Streamlit Cloud**: Upload to GitHub and deploy directly on Streamlit Cloud
2. **Heroku**: Use the provided Procfile
3. **AWS/GCP/Azure**: Deploy using the Docker container

## License
MIT

## Author
Your Name

## Acknowledgments
- David Ruppert and David S. Matteson, authors of "Statistics and Data Analysis for Financial Engineering"
