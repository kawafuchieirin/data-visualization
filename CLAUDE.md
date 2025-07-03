# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Yamanote Line-focused delay and weather correlation analysis project built with Python and Streamlit. The project specializes in analyzing how weather conditions (especially rain) affect delays on Tokyo's Yamanote Line, one of Japan's busiest urban rail lines.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py

# Generate sample data for testing
python src/data_analyzer.py
```

## Architecture

The project follows a modular architecture with data collection, analysis, and visualization components:

- **app.py**: Main Streamlit dashboard application that provides interactive visualizations
- **src/train_delay_collector.py**: Handles collection of train delay data from various sources
- **src/weather_collector.py**: Collects weather data using APIs (OpenWeatherMap)
- **src/data_analyzer.py**: Core analysis logic that merges delay and weather data, calculates correlations, and generates statistics

The visualization includes:
1. Overview statistics showing Yamanote Line delays and correlation metrics
2. Time-based heatmap visualization of delays by hour and weather condition
3. Rush hour vs off-peak analysis specific to Yamanote Line patterns
4. Time series analysis of delay patterns with 7-day moving averages
5. Detailed data view with filtering and export capabilities
6. Yamanote Line-specific insights including operational characteristics

## Key Dependencies

- **streamlit**: Web application framework for the dashboard
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualizations
- **requests & beautifulsoup4**: Web scraping for delay data
- **python-dotenv**: Environment variable management