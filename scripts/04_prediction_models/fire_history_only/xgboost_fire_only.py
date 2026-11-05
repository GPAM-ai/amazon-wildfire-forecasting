"""
XGBoost - multi-horizon (h1-h4) rolling-window forecasting using ONLY the
fire history as feature (the current-month fire count predicts the next 4
months). One XGBRegressor per horizon via MultiOutputRegressor.

Target: Qtd_Focos_Local (monthly fire counts per municipality).
Validation: rolling chronological windows of 12, 24, 48 and 72 months.

Input : datasets/integrated_analysis.csv
Output: results/xgboost_fire_only/xgboost_fire_only_metrics.csv
"""
import os
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
optuna.logging.set_verbosity(optuna.logging.WARNING)

tempo_inicio = time.time()

# ---------------------------------------------------------------------------
# 0. PATHS AND DATA LOADING
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
arq = REPO_ROOT / "datasets" / "integrated_analysis.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "xgboost_fire_only"
caminho_saida = OUTPUT_DIR / "xgboost_fire_only_metrics.csv"

df = pd.read_csv(arq, sep=';', decimal=',')
df['Qtd_Focos_Local'] = pd.to_numeric(df['Qtd_Focos_Local'], errors='coerce')
df['Data'] = pd.to_datetime(df['Mes/Ano'])
df = df.sort_values(['Match_Cidade', 'Data'])

# Fire history only: the single feature is the fire count itself, which is
# also the base column used below to build Focos_h1..Focos_h4.
features = [
    'Qtd_Focos_Local'
]
alvo = "Qtd_Focos_Local"

# ---------------------------------------------------------------------------
# 1. BUILD THE 4 REGRESSION TARGETS (h1 to h4)
# ---------------------------------------------------------------------------
lista_df = []
for cidade, df_cidade in df.groupby("Match_Cidade"):
    df_cidade = df_cidade.sort_values('Data').copy()
    for h in range(1, 5):
        df_cidade[f'Focos_h{h}'] = df_cidade[alvo].shift(-h)
    lista_df.append(df_cidade)

df_modelo = pd.concat(lista_df)

colunas_alvo = [f'Focos_h{h}' for h in range(1, 5)]
df_modelo = df_modelo[['Data'] + features + colunas_alvo].dropna()

datas_disponiveis = df_modelo['Data'].sort_values().unique()

tamanhos_janela = [12, 24, 48, 72]
resultados = []


# ---------------------------------------------------------------------------
# 2. OPTUNA OPTIMIZATION (XGBoost + MultiOutput)
# One study per window size, tuned only on the first window of each size and
# reused in all following rolling iterations.
# ---------------------------------------------------------------------------
def rodar_optuna_bloco(treino_inicial, n_trials=20):
    treino_inicial = treino_inicial.sort_values('Data')
    X = treino_inicial[features]
    y = treino_inicial[colunas_alvo]

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)

        modelo_temp = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42,
                n_jobs=1
            )
        )

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


print("Starting rolling-window validation (XGBoost + MultiOutput, h1-h4, fire history only)...")

for janela in tamanhos_janela:
    print(f"\nProcessing window of {janela} months...")

    if len(datas_disponiveis) <= janela:
        print(f"   (insufficient data for a {janela}-month window)")
        continue

    # ---- Optuna study for this block (1 of the 4 studies) ----
    datas_treino_inicial = datas_disponiveis[0:janela]
    treino_inicial = df_modelo[df_modelo['Data'].isin(datas_treino_inicial)]

    params_bloco = rodar_optuna_bloco(treino_inicial, n_trials=20)
    print(f"   Hyperparameters fixed for the {janela}-month block: {params_bloco}")

    # ---- Sliding the window, without re-optimizing ----
    for i in range(len(datas_disponiveis) - janela):
        datas_treino = datas_disponiveis[i: i + janela]
        data_teste = datas_disponiveis[i + janela]

        treino = df_modelo[df_modelo['Data'].isin(datas_treino)]
        teste = df_modelo[df_modelo['Data'] == data_teste]

        if treino.empty or teste.empty:
            continue

        X_train = treino[features]
        X_test = teste[features]
        y_train = treino[colunas_alvo]
        y_test = teste[colunas_alvo]

        resultado_mes = {
            'Tamanho_Janela (Meses)': janela,
            'Inicio_Treino': pd.to_datetime(datas_treino[0]).strftime('%Y-%m'),
            'Fim_Treino': pd.to_datetime(datas_treino[-1]).strftime('%Y-%m'),
            'Data_Teste (Base)': pd.to_datetime(data_teste).strftime('%Y-%m'),
            'Best_Estimators': params_bloco['n_estimators'],
            'Best_MaxDepth': params_bloco['max_depth'],
            'Best_LearningRate': params_bloco['learning_rate'],
        }

        modelo_final = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=params_bloco['n_estimators'],
                max_depth=params_bloco['max_depth'],
                learning_rate=params_bloco['learning_rate'],
                subsample=params_bloco['subsample'],
                colsample_bytree=params_bloco['colsample_bytree'],
                random_state=42,
                n_jobs=1
            )
        )
        modelo_final.fit(X_train, y_train)

        pred_test = modelo_final.predict(X_test)
        pred_train = modelo_final.predict(X_train)

        # ---- Metrics PER HORIZON (h1..h4) ----
        for idx, h in enumerate(range(1, 5)):
            yt_test = y_test.iloc[:, idx]
            yp_test = np.clip(pred_test[:, idx], 0, None)
            yt_train = y_train.iloc[:, idx]
            yp_train = np.clip(pred_train[:, idx], 0, None)

            mae_teste = mean_absolute_error(yt_test, yp_test)
            r2_teste = r2_score(yt_test, yp_test)
            rmse_teste = np.sqrt(mean_squared_error(yt_test, yp_test))

            mae_treino = mean_absolute_error(yt_train, yp_train)
            r2_treino = r2_score(yt_train, yp_train)
            rmse_treino = np.sqrt(mean_squared_error(yt_train, yp_train))

            resultado_mes[f'MAE_Teste_h{h}'] = mae_teste
            resultado_mes[f'R2_Teste_h{h}'] = r2_teste
            resultado_mes[f'RMSE_Teste_h{h}'] = rmse_teste

            resultado_mes[f'MAE_Treino_h{h}'] = mae_treino
            resultado_mes[f'R2_Treino_h{h}'] = r2_treino
            resultado_mes[f'RMSE_Treino_h{h}'] = rmse_treino

            # Train-test gap (overfitting detection)
            resultado_mes[f'Gap_MAE_h{h}'] = mae_teste - mae_treino
            resultado_mes[f'Gap_R2_h{h}'] = r2_treino - r2_teste

        resultados.append(resultado_mes)

    print(f"Window of {janela} months finished!")

print("\nAll processing completed successfully!")

# ---------------------------------------------------------------------------
# 3. EXPORT METRICS
# ---------------------------------------------------------------------------
df_resultados = pd.DataFrame(resultados)

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_resultados.to_csv(caminho_saida, index=False, sep=';', decimal=',')
print(f"Metrics saved to:\n{caminho_saida}")

# ---------------------------------------------------------------------------
# 4. SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("SUMMARY - Mean MAE, RMSE and R2 per window and horizon (XGBoost)")
print("="*70)

for janela in tamanhos_janela:
    subset = df_resultados[df_resultados['Tamanho_Janela (Meses)'] == janela]
    if subset.empty:
        continue
    print(f"\nWindow {janela} months ({len(subset)} iterations):")
    for h in range(1, 5):
        mae = subset[f'MAE_Teste_h{h}'].mean()
        rmse = subset[f'RMSE_Teste_h{h}'].mean()
        r2 = subset[f'R2_Teste_h{h}'].mean()
        gap_mae = subset[f'Gap_MAE_h{h}'].mean()
        print(f"  h{h}: MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f} | Gap_MAE={gap_mae:.2f}")

# ---------------------------------------------------------------------------
# Total runtime
# ---------------------------------------------------------------------------
tempo_total_segundos = time.time() - tempo_inicio
minutos = int(tempo_total_segundos // 60)
segundos = tempo_total_segundos % 60
print(f"\nTotal runtime: {minutos} min {segundos:.1f} s "
      f"({tempo_total_segundos:.1f} seconds)")