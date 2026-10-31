"""
1-D CNN (multivariate) - multi-horizon (h1-h4) rolling-window forecasting.
Input: the last 6 months of the target and of the 13 climatic, socioeconomic
and macroeconomic covariates.

Target: Qtd_Focos_Local (monthly fire counts per municipality).
Validation: rolling chronological windows of 12, 24, 48 and 72 months.

Input : datasets/integrated_analysis.csv
Output: results/cnn1d_full/cnn1d_full_metrics.csv
"""
import os
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Force CPU to avoid memory crashes
tf.config.set_visible_devices([], 'GPU')

# =========================================================================
# 1. PATHS, CONFIGURATION AND DATA LOADING
# =========================================================================
REPO_ROOT = Path(__file__).resolve().parents[3]
arq = REPO_ROOT / "datasets" / "integrated_analysis.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "cnn1d_full"
caminho_saida = OUTPUT_DIR / "cnn1d_full_metrics.csv"
df = pd.read_csv(arq, sep=';', decimal=',')

features = [
    'Temperatura Media MENSAL(graus celsius)',
    'Precipitacao MENSAL(mm)',
    'Anomalia_ENOS',
    'Populacao_Estimada',
    'Vacas_Ordenhadas_Cabecas',
    'Rebanho_Bovino_Cabecas',
    'PIB_Mil_Reais',
    'Area_Soja_Ha',
    'Valor_Soja_MilReais',
    'Area_Milho_Ha',
    'Valor_Milho_MilReais',
    'Preco_Brent',
    'Tx_Cambio_Dolar'
]
alvo = "Qtd_Focos_Local"

df[alvo] = pd.to_numeric(df[alvo], errors='coerce')
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Data'] = pd.to_datetime(df['Mes/Ano'])

HORIZONTES = [1, 2, 3, 4]
COLUNAS_ALVO = [f'Focos_h{h}' for h in HORIZONTES]

# The model looks at the last 6 months
SEQ_LEN = 6
cols_input = [alvo] + features
n_features = len(cols_input)

# Column order must allow the reshape to work:
# (lag_5_alvo, lag_5_feat1..., lag_4_alvo, lag_4_feat1..., ..., lag_0_...)
LAG_COLS = []
for lag in range(SEQ_LEN-1, -1, -1):
    for col in cols_input:
        LAG_COLS.append(f'{col}_lag_{lag}')

# ---------------------------------------------------------
# 2. BUILD TARGETS (h1..h4) AND MULTIVARIATE INPUT SEQUENCES
# ---------------------------------------------------------
lista_df = []
for cidade, df_cidade in df.groupby("Match_Cidade"):
    df_cidade = df_cidade.sort_values('Data').copy()

    for h in HORIZONTES:
        df_cidade[f'Focos_h{h}'] = df_cidade[alvo].shift(-h)

    for lag in range(SEQ_LEN-1, -1, -1):
        for col in cols_input:
            col_name = f'{col}_lag_{lag}'
            if lag == 0:
                df_cidade[col_name] = df_cidade[col]
            else:
                df_cidade[col_name] = df_cidade[col].shift(lag)

    lista_df.append(df_cidade)

df_modelo = pd.concat(lista_df)
df_modelo = df_modelo[['Match_Cidade', 'Data'] + LAG_COLS + COLUNAS_ALVO].dropna()

datas_disponiveis = df_modelo['Data'].sort_values().unique()
tamanhos_janela = [12, 24, 48, 72]
resultados = []

# ---------------------------------------------------------
# 3. 1-D CNN ARCHITECTURE (MULTIVARIATE) AND OPTUNA
# ---------------------------------------------------------
def build_cnn1d_model(filters, kernel_size, dropout, learning_rate):
    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(SEQ_LEN, n_features)),
        Flatten(),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(len(HORIZONTES), activation='linear')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    return model

def rodar_optuna_bloco(treino_inicial, n_trials=10):
    X_raw = treino_inicial[LAG_COLS].values
    y = treino_inicial[COLUNAS_ALVO].values

    if len(X_raw) < 15:
        return {'filters': 32, 'kernel_size': 2, 'dropout': 0.2, 'learning_rate': 0.001}

    def objective(trial):
        filters = trial.suggest_categorical("filters", [16, 32, 64])
        kernel_size = trial.suggest_categorical("kernel_size", [2, 3])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        tscv = TimeSeriesSplit(n_splits=3)
        fold_maes = []

        for train_idx, val_idx in tscv.split(X_raw):
            X_train_raw_cv, y_train_cv = X_raw[train_idx], y[train_idx]
            X_val_raw_cv, y_val_cv = X_raw[val_idx], y[val_idx]

            # Scaling: flatten to 2D, apply the scaler, restore the 3D shape
            scaler = StandardScaler()
            X_train_2d = X_train_raw_cv.reshape(-1, n_features)
            X_train_s = scaler.fit_transform(X_train_2d).reshape(-1, SEQ_LEN, n_features)

            X_val_2d = X_val_raw_cv.reshape(-1, n_features)
            X_val_s = scaler.transform(X_val_2d).reshape(-1, SEQ_LEN, n_features)

            model = build_cnn1d_model(filters, kernel_size, dropout, learning_rate)
            cb = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

            model.fit(X_train_s, y_train_cv, validation_data=(X_val_s, y_val_cv),
                      epochs=40, batch_size=32, callbacks=cb, verbose=0)

            pred = model.predict(X_val_s, verbose=0)
            fold_maes.append(mean_absolute_error(y_val_cv, pred))
            tf.keras.backend.clear_session()

        return np.mean(fold_maes)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# ---------------------------------------------------------
# 4. MAIN LOOP (ROLLING WINDOWS)
# Hyperparameters are tuned only on the first window of each size and reused
# in all following rolling iterations.
# ---------------------------------------------------------
print("Starting rolling-window validation (h1 to h4, full 1-D CNN)...")

for janela in tamanhos_janela:
    print(f"\nProcessing window of {janela} months...")

    if len(datas_disponiveis) <= janela:
        continue

    datas_treino_inicial = datas_disponiveis[0:janela]
    treino_inicial = df_modelo[df_modelo['Data'].isin(datas_treino_inicial)]

    params_bloco = rodar_optuna_bloco(treino_inicial, n_trials=10)
    print(f"   1-D CNN hyperparameters fixed for the {janela}-month block: {params_bloco}")

    for i in range(len(datas_disponiveis) - janela):
        datas_treino = datas_disponiveis[i: i + janela]
        data_teste = datas_disponiveis[i + janela]

        treino = df_modelo[df_modelo['Data'].isin(datas_treino)]
        teste = df_modelo[df_modelo['Data'] == data_teste]

        if treino.empty or teste.empty:
            continue

        X_train_raw = treino[LAG_COLS].values
        y_train = treino[COLUNAS_ALVO].values
        X_test_raw = teste[LAG_COLS].values
        y_test = teste[COLUNAS_ALVO].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_raw.reshape(-1, n_features)).reshape(-1, SEQ_LEN, n_features)
        X_test_s = scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(-1, SEQ_LEN, n_features)

        resultado_mes = {
            'Tamanho_Janela (Meses)': janela,
            'Inicio_Treino': pd.to_datetime(datas_treino[0]).strftime('%Y-%m'),
            'Fim_Treino': pd.to_datetime(datas_treino[-1]).strftime('%Y-%m'),
            'Data_Teste (Base)': pd.to_datetime(data_teste).strftime('%Y-%m'),
            'Filters': params_bloco['filters'],
            'Kernel': params_bloco['kernel_size'],
            'Learning_Rate': params_bloco['learning_rate'],
        }

        model = build_cnn1d_model(params_bloco['filters'], params_bloco['kernel_size'],
                                  params_bloco['dropout'], params_bloco['learning_rate'])
        cb_final = [
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
        ]

        try:
            model.fit(X_train_s, y_train, validation_data=(X_test_s, y_test),
                      epochs=60, batch_size=32, callbacks=cb_final, verbose=0)
        except Exception as e:
            print(f"   [WARNING] Training failed at {resultado_mes['Data_Teste (Base)']}: {e}")
            continue

        pred_test = np.clip(model.predict(X_test_s, verbose=0), 0, None)
        pred_train = np.clip(model.predict(X_train_s, verbose=0), 0, None)

        mae_teste_h, mae_treino_h, r2_teste_h, r2_treino_h, rmse_teste_h, rmse_treino_h = [], [], [], [], [], []

        for idx_h, h in enumerate(HORIZONTES):
            yt_test = y_test[:, idx_h]
            yp_test = pred_test[:, idx_h]
            yt_train = y_train[:, idx_h]
            yp_train = pred_train[:, idx_h]

            mae_test = mean_absolute_error(yt_test, yp_test)
            mae_train = mean_absolute_error(yt_train, yp_train)
            r2_test = r2_score(yt_test, yp_test)
            rmse_test = np.sqrt(mean_squared_error(yt_test, yp_test))
            rmse_train = np.sqrt(mean_squared_error(yt_train, yp_train))

            resultado_mes[f'MAE_Teste_h{h}'] = mae_test
            resultado_mes[f'R2_Teste_h{h}'] = r2_test
            resultado_mes[f'RMSE_Teste_h{h}'] = rmse_test
            resultado_mes[f'Gap_MAE_h{h}'] = mae_test - mae_train

            mae_teste_h.append(mae_test)
            mae_treino_h.append(mae_train)
            r2_teste_h.append(r2_test)
            rmse_teste_h.append(rmse_test)
            rmse_treino_h.append(rmse_train)

        resultado_mes['MAE_Teste_Medio'] = np.mean(mae_teste_h)
        resultado_mes['MAE_Treino_Medio'] = np.mean(mae_treino_h)
        resultado_mes['R2_Teste_Medio'] = np.mean(r2_teste_h)
        resultado_mes['RMSE_Teste_Medio'] = np.mean(rmse_teste_h)
        resultado_mes['Gap_MAE_Medio'] = resultado_mes['MAE_Teste_Medio'] - resultado_mes['MAE_Treino_Medio']

        resultados.append(resultado_mes)

        # Free Keras memory
        tf.keras.backend.clear_session()

    print(f"Window of {janela} months finished!")

print("\nAll processing completed successfully!")

# ---------------------------------------------------------
# 5. EXPORT METRICS
# ---------------------------------------------------------
df_resultados = pd.DataFrame(resultados)
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_resultados.to_csv(caminho_saida, index=False, sep=';', decimal=',')
print(f"Metrics saved to:\n{caminho_saida}")

print("\n" + "="*70)
print("1-D CNN SUMMARY (with climate/IBGE features) - MAE, RMSE (mean) and R2 (median)")
print("="*70)

for janela in tamanhos_janela:
    subset = df_resultados[df_resultados['Tamanho_Janela (Meses)'] == janela]
    if subset.empty:
        continue
    print(f"\nWindow {janela} months ({len(subset)} iterations):")
    for h in HORIZONTES:
        mae_h = subset[f'MAE_Teste_h{h}'].mean()
        rmse_h = subset[f'RMSE_Teste_h{h}'].mean()
        r2_h = subset[f'R2_Teste_h{h}'].median()
        print(f"    h{h}: MAE={mae_h:.2f} | RMSE={rmse_h:.2f} | R2(median)={r2_h:.3f}")