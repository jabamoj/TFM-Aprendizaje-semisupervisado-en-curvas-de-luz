import lightkurve as lk
import pandas as pd
import numpy as np
from tsfresh.feature_extraction import extract_features, EfficientFCParameters
from multiprocessing import Pool

### CARGA Y FILTRADO INICIAL ###
df_koi = pd.read_csv('cumulative_2022.08.09_10.37.12.csv', skiprows=144)
df_koi = df_koi[df_koi.koi_disposition != 'CANDIDATE'].reset_index(drop=True)

### GLOBAL PARA MULTIPROCESSING ###
df_koi_global = None

def init_worker(df):
    global df_koi_global
    df_koi_global = df

### FUNCION PRINCIPAL POR FILA ###
def process_row_raw(i):
    row = df_koi_global.iloc[i]
    kepid = row['kepid']                      # ← capturamos el identificador único

    try:
        kic = f"KIC {kepid}"
        koi_disposition = row['koi_disposition']
        print(f"[{i+1}] Descargando curva cruda para {kic}")

        lc_collection = lk.search_lightcurve(kic, mission='Kepler')[2:4].download_all()
        lc = lc_collection.stitch().remove_outliers().remove_nans()

        df_raw = pd.DataFrame({
            'kepid': [kepid] * len(lc.time.value),   # ← usamos kepid como columna de agrupación
            'time': lc.time.value,
            'flux': lc.flux.value
        })

        features = extract_features(
            df_raw,
            column_id="kepid",               # ← agrupamos por kepid
            column_sort="time",
            column_value="flux",
            default_fc_parameters=EfficientFCParameters(),
            n_jobs=0
        )

        y_val = koi_disposition == 'CONFIRMED'
        return (kepid, features, y_val)      # ← devolvemos kepid como clave
    except Exception as e:
        print(f"Error en fila {i} (KIC {kepid}): {e}")
        return None

### FUNCION DE EJECUCIÓN PARALELA ###
def parallel_extract_raw(df_koi, save_every=100, save_prefix='features_raw_part', cpu_cores=10):
    print(f"Procesando con {cpu_cores} núcleos")
    total_len = len(df_koi)

    with Pool(processes=cpu_cores, initializer=init_worker, initargs=(df_koi,)) as pool:
        results = pool.map(process_row_raw, range(total_len))

    features_list = []
    y_dict = {}

    for result in results:
        if result is None:
            continue
        kepid, features, y_val = result
        features_list.append(features)       # cada features ya indexado por kepid
        y_dict[kepid] = y_val                # diccionario con etiquetas por kepid

    features_total = pd.concat(features_list, axis=0) if features_list else pd.DataFrame()
    y_series = pd.Series(y_dict, name='label') if y_dict else pd.Series()

    return features_total, y_series

### BUCLE PRINCIPAL ###
if __name__ == "__main__":
    len_df = len(df_koi)
    epoch_num = len_df // 500 + 1

    for i in range(epoch_num):
        print(f"\nEpoch: {i*500}-{(i+1)*500}")
        df_partial = df_koi.iloc[i*500 : min((i+1)*500, len_df)]
        features_total, y_series = parallel_extract_raw(df_partial)

        features_total.to_csv(f"features_raw_KEPID_PART_{i}.csv")
        y_series.to_csv(f"labels_raw_KEPID_PART_{i}.csv")
