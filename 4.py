import pandas as pd
import numpy as np

# Load your crop dataset
df = pd.read_csv("crop_production.csv")

# Assign cost per hectare based on crop
cost_dict = {
    "Rice": 50000,
    "Wheat": 40000,
    "Maize": 30000,
    "Sugarcane": 80000,
    "Cotton": 60000
}

# Fill missing crops with default
df["cost_per_hectare"] = df["Crop"].map(cost_dict).fillna(35000)

# Calculate total cost
df["cost"] = df["Area"] * df["cost_per_hectare"]

# Save
df.to_csv("cost_added.csv", index=False)