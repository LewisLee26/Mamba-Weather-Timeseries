import pandas as pd

# Read Parquet file into a DataFrame
df = pd.read_parquet("gfs_dataframe_13.0.parquet")

# Display the DataFrame
print(df)

def normalize_column(dataframe, column):
    # Find the minimum and maximum values in the list
    min_val = dataframe[column].min()
    max_val = dataframe[column].max()

    # Normalize each value in the list
    dataframe[column] = [(x - min_val) / (max_val - min_val) * 2 - 1 for x in list(dataframe[column])]

    return dataframe

df["coord_index"] = df["coord_index"] - df['coord_index'].min()
columns = list(df.keys())
columns.remove("coord_index")
for column in columns:
    df = normalize_column(df, column)

df.iloc[:, :] = df.iloc[::-1, :]

print(df)

df.to_parquet("gfs_dataframe_train_test.parquet")