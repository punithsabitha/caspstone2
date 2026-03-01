import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = sns.load_dataset('penguins_size')

# --- Basic Exploration ---
print(df.head(10))
print(df.shape)
print(df.tail())
print(df.info())
print(df.dtypes)

# Missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe(include='all').T)


# --- Correlation & Heatmap ---
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), cmap='Wistia', annot=True)
plt.title("Correlation Heatmap")
plt.show()

# --- Histograms ---
df.hist(figsize=(12,8))
plt.suptitle("Distribution of Numeric Features")
plt.show()

# --- Boxplots ---
df.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False,
        figsize=(10,12))
plt.suptitle("Boxplots of Features")
plt.show()

# --- Value counts ---
print(df['sex'].value_counts())
print(df['island'].value_counts())
print(df['species'].value_counts())

# --- Countplots ---
sns.countplot(data=df, x='sex', palette='summer')
plt.show()

sns.countplot(data=df, x='island', palette='RdPu')
plt.show()

sns.countplot(data=df, x='species', palette='YlOrRd')
plt.show()

sns.countplot(data=df, x='sex', hue='species', palette='rocket')
plt.show()

sns.countplot(data=df, x='island', hue='species', palette='husl')
plt.show()

sns.countplot(data=df, x='island', hue='sex', palette='spring')
plt.show()

# ---- pairplot ----
sns.pairplot(data=df, hue='species', palette='mako')
plt.show()