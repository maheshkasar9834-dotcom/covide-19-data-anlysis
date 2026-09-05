# COVID-19 Case Growth Analysis

This project analyzes COVID-19 case growth over time using Python, pandas, matplotlib, and scikit-learn. It explores trends in total cases, deaths, and recoveries, then uses polynomial regression to estimate total cases for a selected day.

## Project Files

- `Covid-19 project-02/covid-19.py` - data cleaning, visualization, modeling, and prediction script
- `Covid-19 project-02/COVID19_Case_Growth_Dataset (3).csv` - COVID-19 case dataset
- `Covid-19 project-02/Figure_1.png` - total cases over time
- `Covid-19 project-02/Figure_2.png` - recovered cases compared with deaths

## Requirements

Install the required Python packages:

```bash
python3 -m pip install --user pandas matplotlib scikit-learn
```

## How to Run

From the repository root, run:

```bash
cd "Covid-19 project-02"
python3 covid-19.py
```

The script prints the dataset summary, model evaluation metrics, the highest and lowest total-case days, and a prediction for day 50. It also opens two charts.

## Visual Results

### Total COVID Cases Over Time

![Total COVID Cases vs Time](Covid-19%20project-02/Figure_1.png)

### Recovered Cases vs Deaths

![Recovered vs Death Cases](Covid-19%20project-02/Figure_2.png)

## Model Results

Using a degree-3 polynomial regression model, the current run produced:

- Mean Absolute Error (MAE): `1994.45`
- R² score: `0.9996`
- Predicted total cases on day 50: `135786`

The dataset contains 200 daily records from March 1, 2020 through September 16, 2020.

## Notes

The model currently uses a random train/test split. For a more realistic time-series evaluation, the training data should contain earlier dates and the test data should contain later dates. Also, the reported peak and low days refer to total cases, not daily case growth.