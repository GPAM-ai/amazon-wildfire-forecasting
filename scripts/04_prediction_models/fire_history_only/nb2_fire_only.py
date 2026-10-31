"""
Negative Binomial (NB2) GLM - multi-horizon (h1-h4) rolling-window forecasting
using ONLY the fire history as feature (univariate/endogenous setup: the
current-month fire count predicts the next 4 months).

Target: Qtd_Focos_Local (monthly fire counts per municipality).
Validation: rolling chronological windows of 12, 24, 48 and 72 months.

Input : datasets/integrated_analysis.csv
Output: results/nb2_fire_only/nb2_fire_only_metrics.csv
"""
import os
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
optuna.logging.set_verbosity(optuna.logging.WARNING)

tempo_inicio = time.time()

# ---------------------------------------------------------------------------
# PATHS (relative to the repository root, so the script runs on any machine)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
arq = REPO_ROOT / "datasets" / "integrated_analysis.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "nb2_fire_only"
caminho_saida = OUTPUT_DIR / "nb2_fire_only_metrics.csv"

alvo = "Qtd_Focos_Local"

# The dataset is a semicolon-separated CSV with decimal commas.
df = pd.read_csv(arq, sep=';', decimal=',')
df[alvo] = pd.to_numeric(df[alvo], errors='coerce')
df['Data'] = pd.to_datetime(df['Mes/Ano'])
df = df.sort_values(['Match_Cidade', 'Data'])

# ---------------------------------------------------------------------------
# OVERDISPERSION DIAGNOSTIC
# Poisson assumes Variance = Mean. A Var/Mean ratio >> 1 means Poisson
# underestimates uncertainty, which motivates the NB2 family.
# ---------------------------------------------------------------------------
media_alvo = df[alvo].mean()
var_alvo = df[alvo].var()
razao_disp = var_alvo / media_alvo
pct_zeros = (df[alvo] == 0).mean() * 100

print("=" * 60)
print(f"OVERDISPERSION DIAGNOSTIC ({alvo})")
print("=" * 60)
print(f"Mean: {media_alvo:.2f}  |  Variance: {var_alvo:.2f}")
print(f"Variance/Mean ratio: {razao_disp:.1f}  (Poisson assumes = 1)")
print(f"% zeros: {pct_zeros:.1f}%")
if razao_disp > 2:
    print("-> Relevant overdispersion detected. Negative Binomial (NB2) is "
          "more adequate than Poisson for this target.")
print("=" * 60 + "\n")

# Fire history only: the single feature is the fire count itself
features = [
    "Qtd_Focos_Local"
]

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
lista_coeficientes = []


# ---------------------------------------------------------------------------
# 2. SCIKIT-LEARN WRAPPER FOR THE STATSMODELS NB2 GLM
# Var = mu + alpha * mu^2. Here 'alpha' is the NB2 DISPERSION parameter, not
# a regularization strength (hence the wide Optuna range, 0.01 to 30).
# The canonical/default link for NegativeBinomial in statsmodels is log.
#
# Numerical safeguards (the unpenalized MLE blew up in a few test months of
# the 12-month window, with MAE reaching ~3.5e10):
#   1) reg_alpha: L2 (ridge) penalty via fit_regularized. It shrinks the
#      coefficients toward zero, reducing variance at the cost of some bias.
#   2) eta_clip: clipping of the linear predictor before exp(), so overflow
#      can never happen even when regularization alone is not enough.
# ---------------------------------------------------------------------------
class NegativeBinomialRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, max_iter=200, reg_alpha=0.1, eta_clip=15.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.reg_alpha = reg_alpha
        self.eta_clip = eta_clip

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        X_ = sm.add_constant(X, has_constant='add')
        family = sm.families.NegativeBinomial(alpha=self.alpha)
        self.model_ = sm.GLM(y, X_, family=family)
        if self.reg_alpha and self.reg_alpha > 0:
            self.results_ = self.model_.fit_regularized(
                alpha=self.reg_alpha, L1_wt=0.0, maxiter=self.max_iter
            )
        else:
            self.results_ = self.model_.fit(maxiter=self.max_iter)
        self.intercept_ = self.results_.params[0]
        self.coef_ = self.results_.params[1:]
        return self

    def predict(self, X):
        X = np.asarray(X)
        X_ = sm.add_constant(X, has_constant='add')
        eta = X_ @ np.asarray(self.results_.params)
        eta = np.clip(eta, -self.eta_clip, self.eta_clip)
        return np.exp(eta)


def nb_deviance(y_true, y_pred, alpha):
    """
    NB2 deviance (analogous to mean_poisson_deviance, but correct for the NB2
    family). Extra metric, since MAE/RMSE/R2 do not capture how well the
    model respects the actual count distribution.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    r = 1.0 / alpha
    termo1 = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    termo2 = (y_true + r) * np.log((y_true + r) / (y_pred + r))
    dev = 2 * (termo1 - termo2)
    return np.mean(dev)


# ---------------------------------------------------------------------------
# 3. OPTUNA - ONE STUDY PER WINDOW SIZE (4 studies in total)
# Hyperparameters are tuned only on the first window of each size and reused
# in all following rolling iterations.
# ---------------------------------------------------------------------------
def rodar_optuna_bloco(treino_inicial, n_trials=20):
    treino_inicial = treino_inicial.sort_values('Data')
    X = treino_inicial[features]
    y = treino_inicial[colunas_alvo]

    scaler_temp = StandardScaler()
    X_scaled = scaler_temp.fit_transform(X)

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 30.0, log=True)
        max_iter = trial.suggest_int("max_iter", 50, 500)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True)

        modelo_temp = MultiOutputRegressor(
            NegativeBinomialRegressor(alpha=alpha, max_iter=max_iter, reg_alpha=reg_alpha)
        )

        tscv = TimeSeriesSplit(n_splits=5)
        try:
            scores = cross_val_score(
                modelo_temp, X_scaled, y,
                cv=tscv,
                scoring='neg_mean_absolute_error'
            )
            return scores.mean()
        except Exception:
            return -999.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


print("Starting rolling-window validation (NB + MultiOutput, h1-h4, fire history only)...")

for janela in tamanhos_janela:
    print(f"\nProcessing window of {janela} months...")

    if len(datas_disponiveis) <= janela:
        print(f"   (insufficient data for a {janela}-month window)")
        continue

    datas_treino_inicial = datas_disponiveis[0:janela]
    treino_inicial = df_modelo[df_modelo['Data'].isin(datas_treino_inicial)]

    params_bloco = rodar_optuna_bloco(treino_inicial, n_trials=20)
    print(f"   Hyperparameters fixed for the {janela}-month block: {params_bloco}")

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

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        resultado_mes = {
            'Tamanho_Janela (Meses)': janela,
            'Inicio_Treino': pd.to_datetime(datas_treino[0]).strftime('%Y-%m'),
            'Fim_Treino': pd.to_datetime(datas_treino[-1]).strftime('%Y-%m'),
            'Data_Teste (Base)': pd.to_datetime(data_teste).strftime('%Y-%m'),
            'Best_Alpha_Dispersao': params_bloco['alpha'],
            'Best_MaxIter': params_bloco['max_iter'],
            'Best_Reg_Alpha': params_bloco['reg_alpha'],
        }

        try:
            modelo_final = MultiOutputRegressor(
                NegativeBinomialRegressor(
                    alpha=params_bloco['alpha'],
                    max_iter=params_bloco['max_iter'],
                    reg_alpha=params_bloco['reg_alpha'],
                )
            )
            modelo_final.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"   Failed to fit NB at {resultado_mes['Data_Teste (Base)']}: {e}")
            continue

        pred_test = modelo_final.predict(X_test_scaled)
        pred_train = modelo_final.predict(X_train_scaled)

        for idx, h in enumerate(range(1, 5)):
            yt_test = y_test.iloc[:, idx]
            yp_test = pred_test[:, idx]
            yt_train = y_train.iloc[:, idx]
            yp_train = pred_train[:, idx]

            mae_teste = mean_absolute_error(yt_test, yp_test)
            r2_teste = r2_score(yt_test, yp_test)
            rmse_teste = np.sqrt(mean_squared_error(yt_test, yp_test))
            deviance_teste = nb_deviance(yt_test, yp_test, params_bloco['alpha'])

            mae_treino = mean_absolute_error(yt_train, yp_train)
            r2_treino = r2_score(yt_train, yp_train)
            rmse_treino = np.sqrt(mean_squared_error(yt_train, yp_train))

            resultado_mes[f'MAE_Teste_h{h}'] = mae_teste
            resultado_mes[f'R2_Teste_h{h}'] = r2_teste
            resultado_mes[f'RMSE_Teste_h{h}'] = rmse_teste
            resultado_mes[f'Deviance_Teste_h{h}'] = deviance_teste

            resultado_mes[f'MAE_Treino_h{h}'] = mae_treino
            resultado_mes[f'R2_Treino_h{h}'] = r2_treino
            resultado_mes[f'RMSE_Treino_h{h}'] = rmse_treino

            resultado_mes[f'Gap_MAE_h{h}'] = mae_teste - mae_treino
            resultado_mes[f'Gap_R2_h{h}'] = r2_treino - r2_teste

            estimator_h = modelo_final.estimators_[idx]
            lista_coeficientes.append({
                "Janela_Meses": janela,
                "Data_Teste": resultado_mes['Data_Teste (Base)'],
                "Horizonte": f"h{h}",
                **dict(zip(features, estimator_h.coef_))
            })

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
# 5. SUMMARY (mean AND median: R2 is sensitive to outliers in low-variance
# months, so both are worth inspecting)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY - MAE, RMSE, R2 (mean/median) and Deviance per window/horizon")
print("=" * 70)

for janela in tamanhos_janela:
    subset = df_resultados[df_resultados['Tamanho_Janela (Meses)'] == janela]
    if subset.empty:
        continue
    print(f"\nWindow {janela} months ({len(subset)} iterations):")
    for h in range(1, 5):
        mae = subset[f'MAE_Teste_h{h}'].mean()
        rmse = subset[f'RMSE_Teste_h{h}'].mean()
        r2_media = subset[f'R2_Teste_h{h}'].mean()
        r2_mediana = subset[f'R2_Teste_h{h}'].median()
        pct_r2_pos = (subset[f'R2_Teste_h{h}'] > 0).mean() * 100
        dev = subset[f'Deviance_Teste_h{h}'].mean()
        print(f"  h{h}: MAE={mae:.2f} | RMSE={rmse:.2f} | "
              f"R2(mean)={r2_media:.3f} | R2(median)={r2_mediana:.3f} | "
              f"%R2>0={pct_r2_pos:.1f}% | Deviance={dev:.3f}")

# ---------------------------------------------------------------------------
# Total runtime
# ---------------------------------------------------------------------------
tempo_total_segundos = time.time() - tempo_inicio
minutos = int(tempo_total_segundos // 60)
segundos = tempo_total_segundos % 60
print(f"\nTotal runtime: {minutos} min {segundos:.1f} s "
      f"({tempo_total_segundos:.1f} seconds)")