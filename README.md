# Change Point Analysis of Brent Oil Prices

This project analyzes structural breaks in Brent crude oil prices using Bayesian Change Point detection. It identifies significant shifts in price behavior and correlates them with geopolitical and economic events.

## Project Structure

```
.
├── data/             # Data storage
├── notebooks/        # Jupyter notebooks for analysis
├── src/              # Source code
├── dashboard/        # Interactive dashboard (Flask + React)
├── docs/             # Documentation
└── tests/            # Unit tests
```

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+ (for dashboard)
- Conda (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/change-point-analysis-time-series.git
   cd change-point-analysis-time-series
   ```

2. Create and activate conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate brent-analysis
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install frontend dependencies:
   ```bash
   cd dashboard/frontend
   npm install
   ```

## Usage

1. Run Jupyter notebooks for analysis:
   ```bash
   jupyter notebook
   ```

2. Start the dashboard:
   ```bash
   # In one terminal (backend)
   cd dashboard/backend
   python app.py
   
   # In another terminal (frontend)
   cd dashboard/frontend
   npm start
   ```

## Data Sources

- **Brent Oil Prices**: [Source]
- **Geopolitical Events**: Manually compiled dataset of key events

## Methodology

1. **Data Collection**: Gather historical Brent oil prices and relevant events
2. **Exploratory Analysis**: Analyze time series properties and stationarity
3. **Change Point Detection**: Implement Bayesian Change Point model using PyMC3
4. **Event Correlation**: Map detected change points to geopolitical/economic events
5. **Dashboard Development**: Create interactive visualizations for exploration

## Results

Key findings and visualizations will be available in the dashboard and final report.

## License

[Specify License]

## Acknowledgments

- [List any references or acknowledgments]