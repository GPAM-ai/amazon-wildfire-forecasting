"""
SARIMAX with exogenous covariates - multi-horizon (h1-h4) rolling-window
forecasting, fitted city by city, with Optuna-selected orders.

Target: Qtd_Focos_Local (monthly fire counts per municipality).
Validation: rolling chronological windows of 12, 24, 48 and 72 months.

Design notes
------------
1) No data leakage: only the exogenous features known at the "base month"
   (the first month after the end of training) are used. They are PERSISTED
   (repeated) over the 4 forecast steps, the same philosophy as the NB2 and
   XGBoost scripts (which use the features of one known month to predict
   h1..h4 ahead).
2) Horizons aligned with NB2/XGBoost: h1 = 1 month after the base month
   (Focos_h{n} = target.shift(-n) from the base-month row).
3) Aggregated RMSE/R2: each iteration yields a single forecast point per
   horizon, so RMSE == MAE at that level. The final summary therefore
   recomputes MAE, RMSE = sqrt(mean(error^2)) and a pooled R2 over ALL
   predictions of each (window, horizon).
4) SARIMAX is fitted per city (it is a univariate-series model by nature),
   unlike NB2/XGBoost, which pool all cities. The final metrics are
   comparable, but the training architecture differs by design.

Input : datasets/integrated_analysis.csv
Output: results/sarimax_full/sarimax_full_metrics.csv
        results/sarimax_full/sarimax_full_predictions.csv
"""
import os
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
optuna.logging.set_verbosity(optuna.logging.WARNING)

tempo_inicio = time.time()

# ---------------------------------------------------------------------------
# 0. CONFIGURATION (paths relative to the repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
arq = REPO_ROOT / "datasets" / "integrated_analysis.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "sarimax_full"
caminho_saida = OUTPUT_DIR / "sarimax_full_metrics.csv"
caminho_previsoes = OUTPUT_DIR / "sarimax_full_predictions.csv"

alvo = "Qtd_Focos_Local"

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

tamanhos_janela = [12, 24, 48, 72]
horizontes = [1, 2, 3, 4]

# --- performance knobs: reduce them if the run is too slow ---
N_TRIALS_OPTUNA = 15        # Optuna trials per block (city + window size)
N_CV_FOLDS = 3              # folds inside the initial block to score each trial
MULTIPLICADOR_SANIDADE = 3  # forecast cannot exceed N x the largest value seen

# ---------------------------------------------------------------------------
# 1. LOADING AND PREPARATION
# ---------------------------------------------------------------------------
df = pd.read_csv(arq, sep=';', decimal=',')
df[alvo] = pd.to_numeric(df[alvo], errors='coerce')
df['Data'] = pd.to_datetime(df['Mes/Ano'])

df_modelo = df[['Match_Cidade', 'Data', alvo] + features].dropna().sort_values(['Match_Cidade', 'Data'])

resultados = []
previsoes_detalhadas = []
cidades = df_modelo['Match_Cidade'].unique()

print(f"Starting SARIMAX per city with Optuna (h1-h4, no leakage). Total cities: {len(cidades)}")
print(f"Warning: with {N_TRIALS_OPTUNA} trials x {N_CV_FOLDS} folds x {len(cidades)} cities x "
      f"{len(tamanhos_janela)} windows, the run can take a long time. Adjust N_TRIALS_OPTUNA/"
      f"N_CV_FOLDS at the top of the script to speed it up.\n")


# ---------------------------------------------------------------------------
# 1.1 STATIONARITY DIAGNOSTIC
# ---------------------------------------------------------------------------
def diagnostico_estacionariedade(serie):
    try:
        adf_result = adfuller(serie)
        adf_estacionaria = adf_result[1] <= 0.05
    except Exception:
        adf_estacionaria = None

    try:
        kpss_result = kpss(serie, regression='c', nlags="auto")
        kpss_estacionaria = kpss_result[1] > 0.05
    except Exception:
        kpss_estacionaria = None

    if adf_estacionaria is None or kpss_estacionaria is None:
        veredito = "Indefinido"
    elif adf_estacionaria and kpss_estacionaria:
        veredito = "Estacionaria (ambos)"
    elif (not adf_estacionaria) and (not kpss_estacionaria):
        veredito = "NaoEstacionaria (ambos)"
    else:
        veredito = "Discordancia entre testes (tratado como nao-estacionaria)"

    return adf_estacionaria, kpss_estacionaria, veredito


# ---------------------------------------------------------------------------
# 2. SCORING EACH OPTUNA TRIAL
# Mini walk-forward INSIDE the initial block, using the SAME leakage-free
# h1-h4 logic that the final model uses.
# ---------------------------------------------------------------------------
def avaliar_trial(y, X, ordem, ordem_sazonal, n_folds=N_CV_FOLDS):
    n = len(y)
    fim_max = n - 5
    fim_min = max(6, n // 3)
    if fim_max < fim_min:
        return None

    fins_treino = np.unique(np.linspace(fim_min, fim_max, num=min(n_folds, fim_max - fim_min + 1), dtype=int))

    maes = []
    for fim_treino in fins_treino:
        y_treino, X_treino = y[:fim_treino], X[:fim_treino]
        X_base = X[fim_treino]
        y_reais = y[fim_treino + 1: fim_treino + 5]
        if len(y_reais) < 4:
            continue
        try:
            modelo_fit = SARIMAX(
                endog=y_treino, exog=X_treino,
                order=ordem, seasonal_order=ordem_sazonal,
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)

            exog_persistido = np.tile(X_base, (5, 1))
            previsao = modelo_fit.forecast(steps=5, exog=exog_persistido)
            previsao_h = previsao[1:5]

            maes.append(mean_absolute_error(y_reais, previsao_h))
        except Exception:
            continue

    return float(np.mean(maes)) if maes else None


def rodar_optuna_bloco(treino_inicial, m_sazonal, n_trials=N_TRIALS_OPTUNA):
    treino_inicial = treino_inicial.sort_values('Data')
    y = treino_inicial[alvo].values
    X = treino_inicial[features].values

    def objective(trial):
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 3)
        if m_sazonal > 1:
            P = trial.suggest_int("P", 0, 2)
            D = trial.suggest_int("D", 0, 1)
            Q = trial.suggest_int("Q", 0, 2)
        else:
            P = D = Q = 0

        ordem = (p, d, q)
        ordem_sazonal = (P, D, Q, m_sazonal)
        mae = avaliar_trial(y, X, ordem, ordem_sazonal)
        return mae if mae is not None else 1e6

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    bp = study.best_params
    ordem = (bp.get("p", 1), bp.get("d", 1), bp.get("q", 1))
    if m_sazonal > 1:
        ordem_sazonal = (bp.get("P", 0), bp.get("D", 0), bp.get("Q", 0), m_sazonal)
    else:
        ordem_sazonal = (0, 0, 0, 0)

    return ordem, ordem_sazonal


# ---------------------------------------------------------------------------
# 3. MAIN LOOP: PER CITY -> PER WINDOW SIZE -> SLIDING
# ---------------------------------------------------------------------------
for cidade in cidades:
    df_cidade = df_modelo[df_modelo['Match_Cidade'] == cidade].sort_values('Data').reset_index(drop=True)
    datas_disponiveis = df_cidade['Data'].sort_values().unique()

    print(f"\n===== City: {cidade} ({len(datas_disponiveis)} months) =====")

    serie_completa = df_cidade[alvo].values
    adf_est, kpss_est, veredito_estacionariedade = diagnostico_estacionariedade(serie_completa)
    maior_valor_historico = np.nanmax(serie_completa) if len(serie_completa) > 0 else 0

    for janela in tamanhos_janela:
        # needs at least: window (training) + 1 (base month) + 4 (horizons) months
        if len(datas_disponiveis) < janela + 5:
            print(f"   (insufficient data for a {janela}-month window)")
            continue

        print(f"   Window of {janela} months...")

        # m=12 seasonality only makes sense with ~2 full cycles; the 12-month
        # window therefore has no seasonal component.
        m_sazonal = 12 if janela >= 24 else 0

        datas_treino_inicial = datas_disponiveis[0:janela]
        treino_inicial = df_cidade[df_cidade['Data'].isin(datas_treino_inicial)]

        ordem, ordem_sazonal = rodar_optuna_bloco(treino_inicial, m_sazonal)
        print(f"      Order selected via Optuna: {ordem} seasonal {ordem_sazonal}")

        # 'i' goes up to the point where window + base month + h1..h4 still fit
        for i in range(len(datas_disponiveis) - janela - 4):
            datas_treino = datas_disponiveis[i: i + janela]
            idx_mes_base = i + janela
            idx_horizontes = [idx_mes_base + h for h in horizontes]

            if max(idx_horizontes) >= len(datas_disponiveis):
                continue

            treino = df_cidade[df_cidade['Data'].isin(datas_treino)]
            if treino.empty:
                continue

            y_train = treino[alvo].values
            X_train = treino[features].values

            data_mes_base = datas_disponiveis[idx_mes_base]
            linha_mes_base = df_cidade[df_cidade['Data'] == data_mes_base]
            if linha_mes_base.empty:
                continue
            X_base = linha_mes_base[features].values[0]

            datas_alvo = datas_disponiveis[idx_horizontes]
            linhas_alvo = df_cidade[df_cidade['Data'].isin(datas_alvo)].sort_values('Data')
            if len(linhas_alvo) < 4:
                continue
            y_reais = linhas_alvo[alvo].values

            resultado_mes = {
                'Cidade': cidade,
                'Estacionariedade_ADF': adf_est,
                'Estacionariedade_KPSS': kpss_est,
                'Estacionariedade_Veredito': veredito_estacionariedade,
                'Tamanho_Janela': janela,
                'Inicio_Treino': pd.to_datetime(datas_treino[0]).strftime('%Y-%m'),
                'Fim_Treino': pd.to_datetime(datas_treino[-1]).strftime('%Y-%m'),
                'Data_MesBase': pd.to_datetime(data_mes_base).strftime('%Y-%m'),
                'Ordem': str(ordem),
                'Ordem_Sazonal': str(ordem_sazonal),
            }

            try:
                modelo_fit = SARIMAX(
                    endog=y_train, exog=X_train,
                    order=ordem, seasonal_order=ordem_sazonal,
                    enforce_stationarity=False, enforce_invertibility=False
                ).fit(disp=False)

                # NO LEAKAGE: the only features known "today" are the
                # base-month ones. They are repeated over the 5 future steps
                # (base month + h1..h4); the 1st position (the base month
                # itself) is discarded.
                exog_persistido = np.tile(X_base, (5, 1))
                previsao_completa = modelo_fit.forecast(steps=5, exog=exog_persistido)
                previsao_h = previsao_completa[1:5]

                limite_superior = max(MULTIPLICADOR_SANIDADE * maior_valor_historico, 1)
                previsao_corrigida = np.clip(previsao_h, 0, limite_superior)
                n_truncados = int(np.sum(previsao_h > limite_superior))

                for h, prev, real in zip(horizontes, previsao_corrigida, y_reais):
                    erro_abs = abs(real - prev)
                    resultado_mes[f'Previsto_h{h}'] = prev
                    resultado_mes[f'Real_h{h}'] = real
                    resultado_mes[f'MAE_h{h}'] = erro_abs
                    resultado_mes[f'RMSE_h{h}'] = erro_abs

                    previsoes_detalhadas.append({
                        'Cidade': cidade,
                        'Tamanho_Janela': janela,
                        'Horizonte': f'h{h}',
                        'Real': real,
                        'Previsto': prev,
                    })

                resultado_mes['Previsoes_Truncadas'] = n_truncados

            except Exception as e:
                for h in horizontes:
                    resultado_mes[f'Previsto_h{h}'] = np.nan
                    resultado_mes[f'Real_h{h}'] = np.nan
                    resultado_mes[f'MAE_h{h}'] = np.nan
                    resultado_mes[f'RMSE_h{h}'] = np.nan
                resultado_mes['Previsoes_Truncadas'] = np.nan
                resultado_mes['Erro'] = str(e)

            resultados.append(resultado_mes)

    print(f"City {cidade} finished!")

print("\nAll processing completed successfully!")

# ---------------------------------------------------------------------------
# 4. EXPORT SPREADSHEETS
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv(caminho_saida, index=False, sep=';', decimal=',')
print(f"Metrics saved to:\n{caminho_saida}")

df_previsoes = pd.DataFrame(previsoes_detalhadas)
df_previsoes.to_csv(caminho_previsoes, index=False, sep=';', decimal=',')
print(f"Predictions (actual x predicted) saved to:\n{caminho_previsoes}")

# ---------------------------------------------------------------------------
# 5. SUMMARY WITH POOLED MAE / RMSE / R2
# RMSE = sqrt(mean(error^2)) over ALL predictions of a (window, h); pooled R2
# over the same set (it makes no sense for a single point at a time).
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY - pooled MAE, RMSE and R2 per window and horizon")
print("=" * 80)

for janela in tamanhos_janela:
    subset = df_previsoes[df_previsoes['Tamanho_Janela'] == janela]
    if subset.empty:
        continue

    print(f"\n--- Window of {janela} months ---")
    for h in horizontes:
        sh = subset[subset['Horizonte'] == f'h{h}'].dropna()
        if sh.empty:
            print(f"  h{h}: no valid predictions")
            continue
        mae = mean_absolute_error(sh['Real'], sh['Previsto'])
        rmse = np.sqrt(mean_squared_error(sh['Real'], sh['Previsto']))
        r2 = r2_score(sh['Real'], sh['Previsto'])
        print(f"  h{h}: MAE={mae:.2f} | RMSE={rmse:.2f} | R2(pooled)={r2:.3f} | n={len(sh)}")

# ---------------------------------------------------------------------------
# Total runtime
# ---------------------------------------------------------------------------
tempo_total_segundos = time.time() - tempo_inicio
minutos = int(tempo_total_segundos // 60)
segundos = tempo_total_segundos % 60
print(f"\nTotal runtime: {minutos} min {segundos:.1f} s ({tempo_total_segundos:.1f} seconds)")