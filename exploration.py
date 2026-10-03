import pandas as pd

# ============================================================
# 1. LOAD THE DATASETS
# ============================================================

train_values = pd.read_csv("dataset/train_values.csv")
train_labels = pd.read_csv("dataset/train_labels.csv")
test_values = pd.read_csv("dataset/test_values.csv")

print("=" * 70)
print("DATASETS LOADED SUCCESSFULLY")
print("=" * 70)


# ============================================================
# 2. DATASET SIZE
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET SIZE")
print("=" * 70)

print("Train values :", train_values.shape)
print("Train labels :", train_labels.shape)
print("Test values  :", test_values.shape)


# ============================================================
# 3. COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("2. TRAINING FEATURE COLUMNS")
print("=" * 70)

for i, column in enumerate(train_values.columns, start=1):
    print(f"{i:2}. {column}")


print("\n" + "=" * 70)
print("3. LABEL COLUMNS")
print("=" * 70)

print(list(train_labels.columns))


# ============================================================
# 4. FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 70)
print("4. FIRST 5 TRAINING ROWS")
print("=" * 70)

print(train_values.head())


print("\n" + "=" * 70)
print("5. FIRST 5 LABEL ROWS")
print("=" * 70)

print(train_labels.head())


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("6. DATA TYPES")
print("=" * 70)

print(train_values.dtypes)


# ============================================================
# 6. NUMERICAL COLUMNS
# ============================================================

numerical_columns = train_values.select_dtypes(
    include=["int64", "float64"]
).columns

print("\n" + "=" * 70)
print("7. NUMERICAL COLUMNS")
print("=" * 70)

for column in numerical_columns:
    print(column)

print("\nNumber of columns detected as numerical:",
      len(numerical_columns))


# ============================================================
# 7. CATEGORICAL COLUMNS
# ============================================================

categorical_columns = train_values.select_dtypes(
    include=["object", "category"]
).columns

print("\n" + "=" * 70)
print("8. CATEGORICAL COLUMNS")
print("=" * 70)

for column in categorical_columns:
    print(column)

print("\nNumber of columns detected as categorical:",
      len(categorical_columns))


# ============================================================
# 8. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("9. MISSING VALUES")
print("=" * 70)

missing_report = pd.DataFrame({
    "Missing Count": train_values.isnull().sum(),
    "Missing Percentage":
        train_values.isnull().mean() * 100
})

# Only display columns that actually contain missing values
missing_report = missing_report[
    missing_report["Missing Count"] > 0
]

if missing_report.empty:
    print("No missing values found in train_values.")
else:
    print(
        missing_report.sort_values(
            by="Missing Count",
            ascending=False
        )
    )


# ============================================================
# 9. MISSING VALUES IN LABELS
# ============================================================

print("\n" + "=" * 70)
print("10. MISSING VALUES IN LABELS")
print("=" * 70)

label_missing = train_labels.isnull().sum()

print(label_missing)


# ============================================================
# 10. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("11. DUPLICATE ROWS")
print("=" * 70)

duplicate_rows = train_values.duplicated().sum()

print("Duplicate rows:", duplicate_rows)


# ============================================================
# 11. DUPLICATE BUILDING IDs
# ============================================================

print("\n" + "=" * 70)
print("12. DUPLICATE BUILDING IDs")
print("=" * 70)

duplicate_ids = train_values["building_id"].duplicated().sum()

print("Duplicate building IDs:", duplicate_ids)


# ============================================================
# 12. TARGET / DAMAGE GRADE
# ============================================================

print("\n" + "=" * 70)
print("13. TARGET COLUMN")
print("=" * 70)

print("Target column: damage_grade")


# ============================================================
# 13. TARGET DISTRIBUTION - COUNTS
# ============================================================

print("\n" + "=" * 70)
print("14. TARGET DISTRIBUTION - COUNTS")
print("=" * 70)

target_counts = train_labels["damage_grade"].value_counts()

print(target_counts.sort_index())


# ============================================================
# 14. TARGET DISTRIBUTION - PERCENTAGES
# ============================================================

print("\n" + "=" * 70)
print("15. TARGET DISTRIBUTION - PERCENTAGES")
print("=" * 70)

target_percentages = (
    train_labels["damage_grade"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

for damage_grade in target_percentages.index:

    print(
        f"Damage Grade {damage_grade}: "
        f"{target_counts[damage_grade]:,} buildings "
        f"({target_percentages[damage_grade]:.2f}%)"
    )


# ============================================================
# 15. UNIQUE VALUES IN CATEGORICAL COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("16. CATEGORICAL FEATURE VALUES")
print("=" * 70)

for column in categorical_columns:

    print(f"\n--- {column} ---")

    print(
        train_values[column]
        .value_counts(dropna=False)
    )


# ============================================================
# 16. NUMERICAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("17. NUMERICAL FEATURE STATISTICS")
print("=" * 70)

print(train_values.describe())


# ============================================================
# 17. CATEGORICAL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("18. CATEGORICAL FEATURE STATISTICS")
print("=" * 70)

print(train_values.describe(include="object"))


print("\n" + "=" * 70)
print("EXPLORATION COMPLETE")
print("=" * 70)