"""
Random Forest - multi-horizon (h1-h4) rolling-window forecasting using ONLY
the fire history as feature (the current-month fire count predicts the next
4 months). One RandomForestRegressor per horizon via MultiOutputRegressor.

Target: Qtd_Focos_Local (monthly fire counts per municipality).
Validation: rolling chronological windows of 12, 24, 48 and 72 months.

Input : datasets/integrated_analysis.csv
Output: results/random_forest_fire_only/random_forest_fire_only_metrics.csv
"""
import os
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.multioutput import MultiOutputRegressor

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
arq = REPO_ROOT / "datasets" / "integrated_analysis.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "random_forest_fire_only"
caminho_saida = OUTPUT_DIR / "random_forest_fire_only_metrics.csv"

df = pd.read_csv(arq, sep=';', decimal=',')
df['Qtd_Focos_Local'] = pd.to_numeric(df['Qtd_Focos_Local'], errors='coerce')
df['Data'] = pd.to_datetime(df['Mes/Ano'])

# Fire history only
features = [
    'Qtd_Focos_Local'
]
alvo = "Qtd_Focos_Local"

# Forecast horizons (h1 to h4)
HORIZONTES = [1, 2, 3, 4]
COLUNAS_ALVO = [f'Focos_h{h}' for h in HORIZONTES]

# ---------------------------------------------------------
# 1. BUILD THE REGRESSION TARGETS FOR h1..h4
# One column per horizon (1 to 4 months ahead), per city, all as the actual
# number of fires.
# ---------------------------------------------------------
lista_df = []
for cidade, df_cidade in df.groupby("Match_Cidade"):
    df_cidade = df_cidade.sort_values('Data').copy()
    for h in HORIZONTES:
        df_cidade[f'Focos_h{h}'] = df_cidade[alvo].shift(-h)
    lista_df.append(df_cidade)

df_modelo = pd.concat(lista_df)
df_modelo = df_modelo[['Data'] + features + COLUNAS_ALVO].dropna()

datas_disponiveis = df_modelo['Data'].sort_values().unique()

tamanhos_janela = [12, 24, 48, 72]
resultados = []
lista_importancias = []

PARAMS_PADRAO = {'n_estimators': 100, 'max_depth': 8, 'min_samples_leaf': 5}


def rodar_optuna_bloco(treino_inicial, n_trials=20):
    """
    Runs Optuna ONCE per block (one window size), using the first available
    training window of that size as the optimization base. Returns the best
    hyperparameters for the RandomForestRegressor used inside the
    MultiOutputRegressor, optimizing the mean MAE across the 4 horizons
    """
    X = treino_inicial[features]
    y = treino_inicial[COLUNAS_ALVO]

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 150)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 20)

        base_modelo = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_samples=0.7,
            random_state=42,
            n_jobs=1,
        )
        modelo_temp = MultiOutputRegressor(base_modelo)

        tscv = TimeSeriesSplit(n_splits=5)
        try:
            scores = cross_val_score(
                modelo_temp, X, y,
                cv=tscv,
                scoring='neg_mean_absolute_error'
            )
            return scores.mean()
        except Exception:
            return -999.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


print("Starting rolling-window validation (h1 to h4, MultiOutputRegressor, fire history only)...")

for janela in tamanhos_janela:
    print(f"\nProcessing window of {janela} months...")

    if len(datas_disponiveis) <= janela:
        print(f"   (insufficient data for a {janela}-month window)")
        continue

    # -----------------------------------------------------------
    # 2. SINGLE TUNING FOR THE WHOLE BLOCK
    # -----------------------------------------------------------
    datas_treino_inicial = datas_disponiveis[0:janela]
    treino_inicial = df_modelo[df_modelo['Data'].isin(datas_treino_inicial)]

    params_bloco = rodar_optuna_bloco(treino_inicial, n_trials=20)
    print(f"   Hyperparameters fixed for the {janela}-month block: {params_bloco}")

    # -----------------------------------------------------------
    # 3. SLIDING THE WINDOW, WITHOUT RE-OPTIMIZING
    # -----------------------------------------------------------
    for i in range(len(datas_disponiveis) - janela):
        datas_treino = datas_disponiveis[i: i + janela]
        data_teste = datas_disponiveis[i + janela]

        treino = df_modelo[df_modelo['Data'].isin(datas_treino)]
        teste = df_modelo[df_modelo['Data'] == data_teste]

        if treino.empty or teste.empty:
            continue

        X_train = treino[features]
        X_test = teste[features]
        y_train = treino[COLUNAS_ALVO]
        y_test = teste[COLUNAS_ALVO]

        resultado_mes = {
            'Tamanho_Janela (Meses)': janela,
            'Inicio_Treino': pd.to_datetime(datas_treino[0]).strftime('%Y-%m'),
            'Fim_Treino': pd.to_datetime(datas_treino[-1]).strftime('%Y-%m'),
            'Data_Teste (Base)': pd.to_datetime(data_teste).strftime('%Y-%m'),
            'Best_Depth': params_bloco['max_depth'],
            'Best_Estimators': params_bloco['n_estimators'],
        }

        base_modelo_final = RandomForestRegressor(
            n_estimators=params_bloco['n_estimators'],
            max_depth=params_bloco['max_depth'],
            min_samples_leaf=params_bloco['min_samples_leaf'],
            random_state=42,
            n_jobs=1,
        )
        modelo_final = MultiOutputRegressor(base_modelo_final)

        modelo_final.fit(X_train, y_train)

        pred_test = modelo_final.predict(X_test)
        pred_train = modelo_final.predict(X_train)

        y_test_arr = y_test.to_numpy()
        y_train_arr = y_train.to_numpy()

        # --- METRICS PER HORIZON (MAE, R2, RMSE, test/train/gap) ---
        # Each metric becomes 4 columns (one per horizon), plus a mean summary
        mae_teste_h, mae_treino_h, r2_teste_h, r2_treino_h, rmse_teste_h, rmse_treino_h = [], [], [], [], [], []

        for idx_h, h in enumerate(HORIZONTES):
            yt_test = y_test_arr[:, idx_h]
            yp_test = pred_test[:, idx_h]
            yt_train = y_train_arr[:, idx_h]
            yp_train = pred_train[:, idx_h]

            mae_test = mean_absolute_error(yt_test, yp_test)
            mae_train = mean_absolute_error(yt_train, yp_train)
            r2_test = r2_score(yt_test, yp_test)
            r2_train = r2_score(yt_train, yp_train)
            rmse_test = np.sqrt(mean_squared_error(yt_test, yp_test))
            rmse_train = np.sqrt(mean_squared_error(yt_train, yp_train))

            resultado_mes[f'MAE_Teste_h{h}'] = mae_test
            resultado_mes[f'R2_Teste_h{h}'] = r2_test
            resultado_mes[f'RMSE_Teste_h{h}'] = rmse_test
            resultado_mes[f'MAE_Treino_h{h}'] = mae_train
            resultado_mes[f'R2_Treino_h{h}'] = r2_train
            resultado_mes[f'RMSE_Treino_h{h}'] = rmse_train
            resultado_mes[f'Gap_MAE_h{h}'] = mae_test - mae_train
            resultado_mes[f'Gap_R2_h{h}'] = r2_train - r2_test
            resultado_mes[f'Gap_RMSE_h{h}'] = rmse_test - rmse_train

            mae_teste_h.append(mae_test)
            mae_treino_h.append(mae_train)
            r2_teste_h.append(r2_test)
            r2_treino_h.append(r2_train)
            rmse_teste_h.append(rmse_test)
            rmse_treino_h.append(rmse_train)

        # --- MEAN SUMMARY ACROSS THE 4 HORIZONS ---
        resultado_mes['MAE_Teste_Medio'] = np.mean(mae_teste_h)
        resultado_mes['MAE_Treino_Medio'] = np.mean(mae_treino_h)
        resultado_mes['R2_Teste_Medio'] = np.mean(r2_teste_h)
        resultado_mes['R2_Treino_Medio'] = np.mean(r2_treino_h)
        resultado_mes['RMSE_Teste_Medio'] = np.mean(rmse_teste_h)
        resultado_mes['RMSE_Treino_Medio'] = np.mean(rmse_treino_h)
        resultado_mes['Gap_MAE_Medio'] = resultado_mes['MAE_Teste_Medio'] - resultado_mes['MAE_Treino_Medio']
        resultado_mes['Gap_R2_Medio'] = resultado_mes['R2_Treino_Medio'] - resultado_mes['R2_Teste_Medio']
        resultado_mes['Gap_RMSE_Medio'] = resultado_mes['RMSE_Teste_Medio'] - resultado_mes['RMSE_Treino_Medio']

        # Feature importance accumulation.
        # MultiOutputRegressor keeps one RandomForest per horizon in
        # modelo_final.estimators_ (same order as COLUNAS_ALVO). We take the
        # mean importance across the 4 horizons and also keep it per horizon.
        importancias_por_h = np.array([est.feature_importances_ for est in modelo_final.estimators_])
        importancia_media = importancias_por_h.mean(axis=0)

        linha_importancia = {
            "Janela_Meses": janela,
            "Data_Teste": resultado_mes['Data_Teste (Base)'],
        }
        linha_importancia.update(dict(zip(features, importancia_media)))
        for idx_h, h in enumerate(HORIZONTES):
            for f_idx, f_nome in enumerate(features):
                linha_importancia[f'{f_nome}_h{h}'] = importancias_por_h[idx_h, f_idx]

        lista_importancias.append(linha_importancia)

        resultados.append(resultado_mes)

    print(f"Window of {janela} months finished!")

print("\nAll processing completed successfully!")

# ---------------------------------------------------------------------------
# 4. EXPORT METRICS
# ---------------------------------------------------------------------------
df_resultados = pd.DataFrame(resultados)

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_resultados.to_csv(caminho_saida, index=False, sep=';', decimal=',')
print(f"Metrics saved to:\n{caminho_saida}")

# ---------------------------------------------------------------------------
# 5. SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SUMMARY - Mean MAE, R2 and Gap per window size (mean of h1-h4)")
print("="*70)

for janela in tamanhos_janela:
    subset = df_resultados[df_resultados['Tamanho_Janela (Meses)'] == janela]
    if subset.empty:
        continue
    mae_teste = subset['MAE_Teste_Medio'].mean()
    mae_treino = subset['MAE_Treino_Medio'].mean()
    r2_teste = subset['R2_Teste_Medio'].mean()
    gap_mae = subset['Gap_MAE_Medio'].mean()
    gap_r2 = subset['Gap_R2_Medio'].mean()
    print(f"\nWindow {janela} months ({len(subset)} iterations):")
    print(f"  MAE Test={mae_teste:.2f}  |  MAE Train={mae_treino:.2f}  |  Gap MAE={gap_mae:.2f}")
    print(f"  R2  Test={r2_teste:.3f}  |  Gap R2={gap_r2:.3f}")
    if gap_mae > mae_treino * 2:
        print(f"  WARNING: high gap - possible overfitting in this window")

    # Detail per horizon
    for h in HORIZONTES:
        mae_h = subset[f'MAE_Teste_h{h}'].mean()
        r2_h = subset[f'R2_Teste_h{h}'].mean()
        print(f"    h{h}: MAE={mae_h:.2f}  R2={r2_h:.3f}")