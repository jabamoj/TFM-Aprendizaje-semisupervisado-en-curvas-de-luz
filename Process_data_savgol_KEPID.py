import lightkurve as lk
import matplotlib.pyplot as plt # para plotear datos
import pandas as pd
import numpy as np
import pywt
import pywt.data
from matplotlib.pyplot import figure

from math import log

from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1

from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, EfficientFCParameters

from scipy.signal import savgol_filter

### EXTRACCIÖN ###

df_koi = pd.read_csv('cumulative_2022.08.09_10.37.12.csv', skiprows=144)
#Si queremos eliminar los candidatos
df_koi = df_koi[df_koi.koi_disposition != 'CANDIDATE']
#Reseteamos el índice tras eliminar exoplanetas candidatos del dataset
df_koi = df_koi.reset_index(drop=True)
df_koi

def extract_curves(row,drop_collection=False,stitch_curves=True):
  '''
  Identifica un objeto en el fichero y lo descarga usando lightkurve

  https://docs.lightkurve.org/reference/index.html

  El resultado es una colección de curvas stitch() puede usarse para coserlas

  Devuelve la colección y la versión cosida

  '''
  period, t0, duration_hours, koi_disposition = row['koi_period'], row['koi_time0bk'], row['koi_duration'], row['koi_disposition']
  kic = "KIC " + str(row['kepid'])
  lc_search = lk.search_lightcurve(kic, mission='Kepler')
  lc_collection = lc_search.download_all()
  if (stitch_curves):
    lc = lc_collection.stitch().remove_outliers(sigma=20, sigma_upper=4)
    lc_nonans=lc.remove_nans()
  else:
    lc_nonans=None

  if drop_collection:
    del lc_collection
    lc_collection=None

  return lc_collection,lc_nonans

### DATA EXTRACTION ###

df_koi_global = None  # fuera de la función principal

def init_worker(df):
    global df_koi_global
    df_koi_global = df

def process_row(i):
    row = df_koi_global.iloc[i]
    kepid = row['kepid']                            

    try:
        period, t0 = row['koi_period'], row['koi_time0bk']
        print(f"[{i+1}] Procesando KIC {kepid} – {row['kepler_name']}")

        # 1) descargar y coser
        lc_collection = lk.search_lightcurve(f"KIC {kepid}", mission='Kepler')[2:4].download_all()
        lc = lc_collection.stitch().remove_outliers().remove_nans()
        klc = lc.fold(period=period, epoch_time=t0)

        # 2) suavizar
        def smooth_flux(masked):
            x = masked.flux.value
            n = len(x)
            wl = round(n/250) or 3
            if wl % 2 == 0: wl += 1
            po = min(round(1 + log(wl)), 5)
            return savgol_filter(x, window_length=wl, polyorder=po)

        odd_df = klc[klc.odd_mask]
        even_df = klc[klc.even_mask]

        # 3) construir DataFrame de eventos con columna 'kepid'
        def make_df(lc_masked, flux_smoothed, label):
            return pd.DataFrame({
                'kepid': [kepid] * len(lc_masked),   # ②
                'time': lc_masked.time_original,
                'flux': flux_smoothed[:len(lc_masked)],
                'event': [label] * len(lc_masked)
            })

        df_odd  = make_df(odd_df,  smooth_flux(odd_df),  'ODD')
        df_even = make_df(even_df, smooth_flux(even_df), 'EVEN')
        all_events = pd.concat([df_odd, df_even], ignore_index=True)

        # 4) extraer características usando 'kepid' como column_id
        features = extract_features(
            all_events,
            column_id="kepid",        
            column_sort="time",
            column_kind="event",
            column_value="flux",
            default_fc_parameters=EfficientFCParameters(),
            n_jobs=0
        )

        y_val = (row['koi_disposition'] == 'CONFIRMED')
        return (kepid, features, y_val)        # ④

    except Exception as e:
        print(f"Error en fila {i} (KIC {kepid}): {e}")
        return None
    
from multiprocessing import Pool

def parallel_create_dataset(df_koi, cpu_cores=16):
    with Pool(processes=cpu_cores, initializer=init_worker, initargs=(df_koi,)) as pool:
        results = pool.map(process_row, range(len(df_koi)))

    feats      = []
    kepid_list = []
    y_list     = []

    for res in results:
        if res is None:
            continue
        kepid, ftr, y = res

        feats.append(ftr)         # DataFrame indexado por kepid
        kepid_list.append(kepid)  # guardamos cada repetición
        y_list.append(y)

    # 1) Concatenar Features
    features_total = pd.concat(feats, axis=0)

    # 2) Serie de labels, índice y valores en el mismo orden
    y_series = pd.Series(data=y_list, index=kepid_list, name='label')

    return features_total, y_series

if __name__ == "__main__":

    len_df = len(df_koi)
    epoch_num = len_df // 500 + 1
    for i in range (epoch_num):
        print(f"Epoch: {i*500}-{(i+1)*500}")
        df_partial = df_koi.iloc[i*500 : min((i+1)*500, len_df)]
        extracted_features_total, y_series = parallel_create_dataset(df_partial)
    
        extracted_features_total.to_csv(f"features_savgol_KEPID_PART_{i}.csv")
        y_series.to_csv(f"labels_savgol_KEPID_PART_{i}.csv")
