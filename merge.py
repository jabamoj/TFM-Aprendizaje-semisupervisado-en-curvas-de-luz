import pandas as pd

num_partes = 16  # del 0 al 15 incluidos

# Combinar características
features_list = []
for i in range(num_partes):
    filename = f"features_savgol_KEPID_PART_{i}.csv"
    print(f"Leyendo {filename}")
    df = pd.read_csv(filename)
    features_list.append(df)

df_features_total = pd.concat(features_list, ignore_index=True)
df_features_total.to_csv(f"features_var_savgol_KEPID_total.csv", index=False)
print(f"Guardado: features_var_savgol_KEPID_total.csv")

# Combinar etiquetas
labels_list = []
for i in range(num_partes):
    filename = f"labels_savgol_KEPID_PART_{i}.csv"
    print(f"Leyendo {filename}")
    df = pd.read_csv(filename, index_col=0)
    series = df.iloc[:, 0]
    labels_list.append(series)

y_total = pd.concat(labels_list)
y_total.to_csv(f"labels_var_savgol_KEPID_total.csv")
print(f"Guardado: labels_var_savgol_KEPID_total.csv")
