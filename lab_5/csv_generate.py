import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Tips dataset
tips = sns.load_dataset("tips")

# Create a pivot table of total bill by day and time
pivot_table = tips.pivot_table(index="day", columns="time", values="total_bill", aggfunc="sum")

# Save pivot table to CSV
csv_filename = "tips_chart.csv"
pivot_table.to_csv(csv_filename)
print(f"CSV saved: {csv_filename}")