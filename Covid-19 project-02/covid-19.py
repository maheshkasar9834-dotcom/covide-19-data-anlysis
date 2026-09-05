import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1 – Load Dataset

df = pd.read_csv("COVID19_Case_Growth_Dataset (3).csv")

# Step 2 – Data Cleaning

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.dropna()

print("Columns:", df.columns)

# Convert Date
df['date'] = pd.to_datetime(df['date'])

# Convert Date → Numeric
df['days'] = (df['date'] - df['date'].min()).dt.days

print("First 5 Rows:\n", df.head())

# Step 3 – EDA

print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

# Graph 1: Total Cases vs Time

plt.figure()
plt.plot(df['days'], df['total_cases'], 'o-')
plt.title("Total COVID Cases vs Time")
plt.xlabel("Days")
plt.ylabel("Total Cases")
plt.show()

# Graph 2: Deaths vs Recovered

plt.figure()
plt.plot(df['days'], df['total_recovered'], label='Recovered')
plt.plot(df['days'], df['total_deaths'], label='Deaths')
plt.title("Recovered vs Death Cases")
plt.xlabel("Days")
plt.ylabel("Count")
plt.legend()
plt.show()

# Step 4 – Feature Engineering

X = df[['days']]
y = df['total_cases']

# Polynomial Features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)


# Step 5 – Train Model

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


# Step 6 – Prediction

y_pred = model.predict(X_test)

# Step 7 – Results & Insights

print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Peak & Low Day
peak_day = df.groupby('days')['total_cases'].mean().idxmax()
low_day = df.groupby('days')['total_cases'].mean().idxmin()

print("\nHighest Growth Day:", peak_day)
print("Lowest Growth Day:", low_day)


# Step 8 – Custom Prediction

sample = [[50]]
sample_poly = poly.transform(sample)

pred = model.predict(sample_poly)

print("\nPredicted Total Cases on Day 50:", int(pred[0]))