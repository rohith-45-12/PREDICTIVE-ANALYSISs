# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix


# Step 1: Load dataset
df = pd.read_csv(r"C:\Users\Rohit\OneDrive\Attachments\Desktop\ore\pgurutharun_17934417763455353.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print(df.head())
print(df.columns)
print(df.shape)


# Step 2: Select water quality columns

min_temp = 'min_temp'
max_temp = 'max_temp'

min_do = 'min_do'
max_do = 'max_do'

min_ph = 'min_ph'
max_ph = 'max_ph'

min_cond = 'min_cond'
max_cond = 'max_cond'

min_bod = 'min_bod'
max_bod = 'max_bod'

min_nitrate = 'min_nitrate'
max_nitrate = 'max_nitrate'

min_fecal = 'min_fecal'
max_fecal = 'max_fecal'

min_tc = 'min_tc'
max_tc = 'max_tc'

min_fs = 'min_fs'
max_fs = 'max_fs'

# Step 3: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
print(df.isnull().sum().sum())

# Step 4: Select numerical water quality columns

water_cols = [
    min_temp, max_temp,
    min_do, max_do,
    min_ph, max_ph,
    min_cond, max_cond,
    min_bod, max_bod,
    min_nitrate, max_nitrate,
    min_fecal, max_fecal,
    min_tc,max_tc,
    min_fs,max_fs
]

# Convert columns into numeric values
df[water_cols] = df[water_cols].apply(pd.to_numeric, errors="coerce")

# Fill missing values with mean
df[water_cols] = df[water_cols].fillna(df[water_cols].mean())

# Step 5: Bar graph showing Average Maximum BOD Across Indian States
state_bod = df.groupby('State/Ut Name')['max_bod'].mean().reset_index()

state_bod = state_bod.sort_values(by='max_bod', ascending=False)

plt.figure(figsize=(14, 7))
sns.barplot(data=state_bod, x='State/Ut Name', y='max_bod')

plt.title("Average Maximum BOD Across Indian States")
plt.xlabel("States / Union Territories")
plt.ylabel("Average Maximum BOD")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
# Step 5: Correlation Analysis

correlation = df[water_cols].corr()

print("\nCorrelation Matrix:")
print(correlation)

short_names = [
    "Min Temp", "Max Temp",
    "Min DO", "Max DO",
    "Min pH", "Max pH",
    "Min Cond", "Max Cond",
    "Min BOD", "Max BOD",
    "Min Nitrate", "Max Nitrate",
    "Min Fecal", "Max Fecal",
    "Min TC", "Max TC",
    "Min FS", "Max FS"
]

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    xticklabels=short_names,
    yticklabels=short_names
)

plt.title("Correlation Heatmap of River Water Quality")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Step 6: Boxplot 

plt.figure(figsize=(12, 6))
plt.boxplot(df[water_cols].values)

plt.xticks(range(1, len(water_cols) + 1), short_names, rotation=90)
plt.title("Boxplot to Identify Outliers")
plt.tight_layout()
plt.show()


# Step 7: Outlier Detection using IQR

Q1 = df[water_cols].quantile(0.25)
Q3 = df[water_cols].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = (
    (df[water_cols] < lower_bound) |
    (df[water_cols] > upper_bound)
)

print("\nOutliers in each column:")
print(outliers.sum())

print("\nTotal outliers:")
print(outliers.sum().sum())


# Step 8: Remove rows containing outliers

df = df[~outliers.any(axis=1)].copy()

print("\nDataset after outlier removal:")
print(df.shape)


# ==========================================
# MODEL 1: MULTIPLE LINEAR REGRESSION
# Predict Maximum BOD
# ==========================================

features = [
    min_do, max_do,
    min_ph, max_ph,
    min_cond, max_cond,
    min_bod,
    min_nitrate, max_nitrate
]

X = df[features]
y = df[max_bod]

data = pd.concat([X, y], axis=1).dropna()

X = data[features]
y = data[max_bod]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMULTIPLE LINEAR REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Maximum BOD")
plt.ylabel("Predicted Maximum BOD")
plt.title("Multiple Linear Regression")
plt.show()


# ==========================================
# MODEL 2: POLYNOMIAL REGRESSION
# Predict Maximum Temperature
# ==========================================

X = df[[min_temp]]
y = df[max_temp]

data = pd.concat([X, y], axis=1).dropna()

X = data[[min_temp]]
y = data[max_temp]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

print("\nPOLYNOMIAL REGRESSION")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, label="Actual")
plt.scatter(X_test, y_pred, label="Predicted")

plt.xlabel("Minimum Temperature")
plt.ylabel("Maximum Temperature")
plt.title("Polynomial Regression")
plt.legend()
plt.show()


# ==========================================
# MODEL 3: LOGISTIC REGRESSION
# Classify Fecal Coliform Contamination
# ==========================================

features = [
    min_temp, max_temp,
    min_do, max_do,
    min_ph, max_ph,
    min_cond, max_cond,
    min_bod, max_bod,
    min_nitrate, max_nitrate
]

X = df[features]
y_value = df[max_fecal]

data = pd.concat([X, y_value], axis=1).dropna()

X = data[features]
y_value = data[max_fecal]

# Create high and low classes using median
threshold = y_value.median()
y = (y_value >= threshold).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(5, 4))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# MODEL 4: K-NEAREST NEIGHBORS
# Classify Dissolved Oxygen
# ==========================================

features = [
    min_temp, max_temp,
    min_ph, max_ph,
    min_cond, max_cond,
    min_bod, max_bod,
    min_nitrate, max_nitrate,
    max_fecal
]

X = df[features]
y_value = df[max_do]

data = pd.concat([X, y_value], axis=1).dropna()

X = data[features]
y_value = data[max_do]

# Create high and low classes using median
threshold = y_value.median()
y = (y_value >= threshold).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\nK-NEAREST NEIGHBORS")
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(5, 4))

sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
