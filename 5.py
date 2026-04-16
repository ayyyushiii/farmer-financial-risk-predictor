import pandas as pd

# =========================
# STEP 1: LOAD CROP DATA
# =========================

crop = pd.read_csv("cost_added.csv")

# =========================
# STEP 2: LOAD PRICE DATA (AUTO FIX HEADERS)
# =========================

file_path = "Market_Wise_Price_Arrival_12-04-2026_04-18-37_PM.csv"

price = None

# Try different skiprows automatically
for i in range(10):
    try:
        temp = pd.read_csv(file_path, skiprows=i)

        cols = [c.lower() for c in temp.columns]

        if any("commodity" in c for c in cols):
            print(f"✅ Correct header found at skiprows = {i}")
            price = temp
            break

    except:
        continue

if price is None:
    raise Exception("❌ Could not detect correct header")

print("Columns found:", price.columns)

# =========================
# STEP 3: FIND PRICE COLUMN
# =========================

price_col = None
for col in price.columns:
    if "price" in col.lower():
        price_col = col
        break

if price_col is None:
    raise Exception("❌ No price column found")

print("Using price column:", price_col)

# =========================
# STEP 4: CLEAN PRICE DATA
# =========================

price = price[["Commodity", price_col]]

price.rename(columns={
    "Commodity": "Crop",
    price_col: "Price"
}, inplace=True)

# =========================
# STEP 5: CLEAN TEXT
# =========================

crop["Crop"] = crop["Crop"].str.lower().str.strip()
price["Crop"] = price["Crop"].str.lower().str.strip()

# Fix common mismatch
price["Crop"] = price["Crop"].replace({
    "paddy": "rice"
})

# =========================
# STEP 6: MERGE
# =========================

df = pd.merge(crop, price, on="Crop", how="inner")

print("Merged rows:", len(df))

if len(df) == 0:
    print("⚠️ WARNING: Merge returned 0 rows")

# =========================
# STEP 7: FIX DATA TYPES (IMPORTANT)
# =========================

df["Production"] = pd.to_numeric(df["Production"], errors="coerce")

df["Price"] = df["Price"].astype(str).str.replace(",", "")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

# Drop invalid rows
df = df.dropna(subset=["Production", "Price", "cost"])

# =========================
# STEP 8: FINANCIAL CALCULATIONS
# =========================

df["revenue"] = df["Production"] * df["Price"]
df["profit"] = df["revenue"] - df["cost"]

# =========================
# STEP 9: CREATE RISK LABEL
# =========================

def assign_risk(profit):
    if profit < 0:
        return "High"
    elif profit < 200000:
        return "Medium"
    else:
        return "Low"

df["risk"] = df["profit"].apply(assign_risk)

# =========================
# STEP 10: SAVE FINAL DATASET
# =========================

df.to_csv("final_dataset.csv", index=False)

print("✅ DONE - final_dataset.csv created successfully")