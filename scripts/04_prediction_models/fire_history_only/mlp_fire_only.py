"""
MLP (multi-layer perceptron) - multi-horizon (h1-h4) rolling-window
forecasting using ONLY the fire history as feature (the current-month fire
count predicts the next 4 months). A single MLPRegressor with 4 outputs.

Target: Qtd_Focos_Local (monthly fire counts per municipality).
Validation: rolling chronological windows of 12, 24, 48 and 72 months.

Input : datasets/integrated_analysis.csv
Output: results/mlp_fire_only/mlp_fire_only_metrics.csv
"""
import os
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
arq = REPO_ROOT / "datasets" / "integrated_analysis.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "mlp_fire_only"
caminho_saida = OUTPUT_DIR / "mlp_fire_only_metrics.csv"

df = pd.read_csv(arq, sep=';', decimal=',')
df['Qtd_Focos_Local'] = pd.to_numeric(df['Qtd_Focos_Local'], errors='coerce')
df['Data'] = pd.to_datetime(df['Mes/Ano'])

# Fire history only: the single feature is the fire count itself, which is
# also the base column used below to build Focos_h1..Focos_h4.
features = [
    'Qtd_Focos_Local'
]
alvo = "Qtd_Focos_Local"

HORIZONTES = [1, 2, 3, 4]
COLUNAS_ALVO = [f'Focos_h{h}' for h in HORIZONTES]

# ---------------------------------------------------------
# 1. BUILD THE REGRESSION TARGETS FOR h1..h4
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

PARAMS_PADRAO = {
    'n_layers': 1,
    'n_units_l1': 64,
    'n_units_l2': None,
    'alpha': 1e-3,
    'learning_rate_init': 1e-3,
}


def construir_modelo(params):
    """
    Builds the Pipeline (scaler + MLP) wrapped in a TransformedTargetRegressor
    from a dictionary of Optuna hyperparameters.

    - MLPRegressor natively accepts a multi-column y (one output neuron per
      horizon), so MultiOutputRegressor is NOT used.
    - StandardScaler is applied to X (inside the Pipeline) and to y (via
      TransformedTargetRegressor), because the MLP is scale-sensitive. The
      metrics (MAE, R2, RMSE) are computed afterwards on the original scale,
      since TransformedTargetRegressor undoes the y normalization in the
      predictions automatically.
    """
    if params['n_layers'] == 1:
        hidden_layer_sizes = (params['n_units_l1'],)
    else:
        hidden_layer_sizes = (params['n_units_l1'], params['n_units_l2'])

    base_pipeline = Pipeline([
        ('scaler_x', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=params['alpha'],
            learning_rate_init=params['learning_rate_init'],
            activation='relu',
            solver='adam',
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.15,
            random_state=42,
        ))
    ])

    return TransformedTargetRegressor(regressor=base_pipeline, transformer=StandardScaler())


def rodar_optuna_bloco(treino_inicial, n_trials=20):
    """
    Runs Optuna ONCE per block (one window size), optimizing the architecture
    (1 or 2 hidden layers, number of neurons), the regularization (alpha) and
    the learning rate (learning_rate_init), using the mean MAE across the 4
    horizons (h1..h4) via TimeSeriesSplit.
    """
    X = treino_inicial[features]
    y = treino_inicial[COLUNAS_ALVO]

    if len(X) < 15:
        return PARAMS_PADRAO

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 2)
        n_units_l1 = trial.suggest_int("n_units_l1", 16, 128)
        n_units_l2 = trial.suggest_int("n_units_l2", 8, 64) if n_layers == 2 else None
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)

        params_trial = {
            'n_layers': n_layers,
            'n_units_l1': n_units_l1,
            'n_units_l2': n_units_l2,
            'alpha': alpha,
            'learning_rate_init': learning_rate_init,
        }

        modelo_temp = construir_modelo(params_trial)

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

    best = study.best_params
    return {
        'n_layers': best['n_layers'],
        'n_units_l1': best['n_units_l1'],
        'n_units_l2': best.get('n_units_l2'),
        'alpha': best['alpha'],
        'learning_rate_init': best['learning_rate_init'],
    }


print("Starting rolling-window validation (h1 to h4, multi-output MLP, fire history only)...")

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
    arquitetura_str = (
        f"({params_bloco['n_units_l1']},)" if params_bloco['n_layers'] == 1
        else f"({params_bloco['n_units_l1']}, {params_bloco['n_units_l2']})"
    )
    print(f"   Architecture fixed for the {janela}-month block: {arquitetura_str} "
          f"| alpha={params_bloco['alpha']:.5f} | lr={params_bloco['learning_rate_init']:.5f}")

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
            'Arquitetura': arquitetura_str,
            'Alpha': params_bloco['alpha'],
            'Learning_Rate': params_bloco['learning_rate_init'],
        }

        modelo_final = construir_modelo(params_bloco)

        try:
            modelo_final.fit(X_train, y_train)
        except Exception as e:
            print(f"   [WARNING] Training failed at {resultado_mes['Data_Teste (Base)']}: {e}")
            continue

        pred_test = modelo_final.predict(X_test)
        pred_train = modelo_final.predict(X_train)

        y_test_arr = y_test.to_numpy()
        y_train_arr = y_train.to_numpy()

        # --- METRICS PER HORIZON (MAE, R2, RMSE) ---
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

        resultado_mes['MAE_Teste_Medio'] = np.mean(mae_teste_h)
        resultado_mes['MAE_Treino_Medio'] = np.mean(mae_treino_h)
        resultado_mes['R2_Teste_Medio'] = np.mean(r2_teste_h)
        resultado_mes['R2_Treino_Medio'] = np.mean(r2_treino_h)
        resultado_mes['RMSE_Teste_Medio'] = np.mean(rmse_teste_h)
        resultado_mes['RMSE_Treino_Medio'] = np.mean(rmse_treino_h)
        resultado_mes['Gap_MAE_Medio'] = resultado_mes['MAE_Teste_Medio'] - resultado_mes['MAE_Treino_Medio']
        resultado_mes['Gap_R2_Medio'] = resultado_mes['R2_Treino_Medio'] - resultado_mes['R2_Teste_Medio']
        resultado_mes['Gap_RMSE_Medio'] = resultado_mes['RMSE_Teste_Medio'] - resultado_mes['RMSE_Treino_Medio']

        # --- FEATURE IMPORTANCE (permutation importance) ---
        # The MLP has no .feature_importances_ (there are no tree splits to
        # measure), so permutation_importance is used: it shuffles each
        # feature in the test set, one at a time, and measures how much the
        # R2 (mean across the 4 horizons) drops. The larger the drop, the
        # more important the feature. n_repeats=3 keeps the runtime
        # reasonable; increase it for a more stable estimate.
        try:
            resultado_perm = permutation_importance(
                modelo_final, X_test, y_test,
                scoring='r2', n_repeats=3, random_state=42, n_jobs=1
            )
            lista_importancias.append({
                "Janela_Meses": janela,
                "Data_Teste": resultado_mes['Data_Teste (Base)'],
                **dict(zip(features, resultado_perm.importances_mean))
            })
        except Exception:
            pass

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

    for h in HORIZONTES:
        mae_h = subset[f'MAE_Teste_h{h}'].mean()
        r2_h = subset[f'R2_Teste_h{h}'].mean()
        print(f"    h{h}: MAE={mae_h:.2f}  R2={r2_h:.3f}")