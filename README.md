# TFM Aprendizaje semisupervisado en curvas de luz

Los codigos de este repositorio desempeñan las siguientes funciones:
- Codigo_modelos_pruebas_DatosCrudos.ipynb: Se desarrolla el entrenamiento y análisis de modelos basados en la extraccón cruda de curvas de luz
- Codigo_modelos_pruebas_Savitzky-Golay.ipynb: Se desarrolla el entrenamiento y análisis de modelos basados en la extraccón de curvas de luz procesadas mediante el filtro de Savitzky-Golay
- Codigo_modelos_pruebas_Wavelet.ipynb: Se desarrolla el entrenamiento y análisis de modelos basados en la extraccón de curvas de luz procesadas mediante Wavelets
- Process_data_raw_KEPID.py: Se extraen las curvas del repositorio Kepler y se extraen sus features directamente
- Process_data_savgol_KEPID.py: Se extraen las curvas del repositorio Kepler y se extraen sus features aplicando previamente el filtrado por Savitzky-Golay
- merge.py: Fusiona los archivos temporrales creados en los Process, de forma que toda la extracció de features y etiquetas quede recogida en un único archivo
- ModelosSavgol_analisis_coincidentes.ipynb: Entrena los modelos con un dataset comun de pruebas para su posterior analisis de resultados de coincidencias
