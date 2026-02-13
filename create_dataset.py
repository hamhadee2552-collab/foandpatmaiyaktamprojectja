import pandas as pd
import numpy as np

np.random.seed(42)

rows = 800

data = {
    "Temp_C": np.random.uniform(15, 35, rows),
    "Humidity_%": np.random.uniform(40, 100, rows),
    "Rain_mm": np.random.uniform(0, 30, rows),
    "LeafWet_hours": np.random.uniform(0, 12, rows),
    "PlantAge_days": np.random.randint(10, 90, rows)
}

df = pd.DataFrame(data)


def risk_label(row):
    if row["Humidity_%"] > 85 and row["LeafWet_hours"] > 8 and 18 <= row["Temp_C"] <= 25:
        return 2   # High
    elif row["Humidity_%"] > 70 and row["LeafWet_hours"] > 5:
        return 1   # Medium
    else:
        return 0   # Low

df["Risk"] = df.apply(risk_label, axis=1)

df.to_csv("potato_blight_environment_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())