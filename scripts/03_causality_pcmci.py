"""
Lagged conditional-dependence analysis with PCMCI (Tigramite), run city by
city, with three conditional-independence tests:

    ParCorr   : partial correlation (linear dependence)
    GPDC      : Gaussian-process distance correlation (nonlinear dependence)
    Spearman  : ParCorr applied to the RANKS of the series (monotonic
                dependence, the rank-based analogue of Spearman correlation)

Variables per municipality: monthly precipitation, monthly mean temperature,
fire count (Qtd_Focos_Local) and ENSO anomaly. PCMCI is run with tau_max = 4
months and pc_alpha = 0.05. Every link with p-value <= 0.05 is exported
(lag 0 to 4 months; lag-0 self links are skipped).

Input : datasets/integrated_analysis.csv
Output (in results/causality_pcmci/):
    CausasPARCORR.csv, CausasGPDC.csv, CausasSPEARMAN.csv   one per method
    CausasPCMCI_AllMethods.csv                              all methods together
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tigramite.plotting as tp
from tigramite import data_processing as pp
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
METODOS_DISPONIVEIS = ['ParCorr', 'GPDC', 'Spearman']
METODOS = sys.argv[1:] or METODOS_DISPONIVEIS
for m in METODOS:
    if m not in METODOS_DISPONIVEIS:
        raise SystemExit(f"Unknown method '{m}'. Choose among: {METODOS_DISPONIVEIS}")

SALVAR_GRAFOS = False

REPO_ROOT = Path(__file__).resolve().parents[1]
arq = REPO_ROOT / "datasets" / "integrated_analysis.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "causality_pcmci"
ARQUIVOS_SAIDA = {
    'ParCorr': "CausasPARCORR.csv",
    'GPDC': "CausasGPDC.csv",
    'Spearman': "CausasSPEARMAN.csv",
}

df = pd.read_csv(arq, sep=';', decimal=',')
df['Qtd_Focos_Local'] = pd.to_numeric(df['Qtd_Focos_Local'], errors='coerce')
df['Data'] = pd.to_datetime(df['Mes/Ano'])

colunas = [
    'Precipitacao MENSAL(mm)',
    'Temperatura Media MENSAL(graus celsius)',
    'Qtd_Focos_Local',
    'Anomalia_ENOS'
]

cidades = df['Match_Cidade'].dropna().unique()

print(f"Total cities found: {len(cidades)}")


def rodar_pcmci(metodo):
    """Runs PCMCI city by city with the given test; returns the sorted
    DataFrame of significant links (p <= 0.05)."""
    lista_results = []

    for cidade in cidades:
        print(f"\n--RUNNING {metodo.upper()} FOR THE CITY OF {cidade}--")

        dfCidade = df[df['Match_Cidade'] == cidade].copy()
        dfCidade = dfCidade.sort_values('Data')

        dfAnalise = dfCidade[colunas].reset_index(drop=True)

        if metodo == 'Spearman':
            dados = dfAnalise.rank()   # ParCorr on the ranks = Spearman
        else:
            dados = dfAnalise

        dfTigramite = pp.DataFrame(
            dados.values,
            datatime=np.arange(len(dfAnalise)),
            var_names=dfAnalise.columns
        )

        if metodo == 'GPDC':
            teste = GPDC(significance='analytic')
        else:
            teste = ParCorr(significance='analytic')
        pcmci = PCMCI(
            dataframe=dfTigramite,
            cond_ind_test=teste,
            verbosity=1
        )
        results = pcmci.run_pcmci(tau_max=4, pc_alpha=0.05)

        if SALVAR_GRAFOS:
            plt.figure(figsize=(10, 10))

            tp.plot_graph(
                val_matrix=results['val_matrix'],
                graph=results['graph'],
                var_names=dfAnalise.columns,
                link_colorbar_label='Causalidade',
                node_size=0.3,
                node_label_size=10,
                arrow_linewidth=2.5,
                arrowhead_size=15,
                curved_radius=0.15,
                link_label_fontsize=10
            )

            plt.title(f"Teste {metodo} - {cidade} (tau_max=4)", fontsize=16)

            pasta_grafos = OUTPUT_DIR / "graphs"
            os.makedirs(pasta_grafos, exist_ok=True)
            plt.savefig(pasta_grafos / f"Grafo_Causalidade_{metodo}_{cidade}.png", bbox_inches='tight')
            plt.close()

        var = dfAnalise.columns
        for i, var_origem in enumerate(var):
            for j, var_destino in enumerate(var):
                for tau in range(5):

                    if tau == 0 and i == j:
                        continue

                    p_valor = results['p_matrix'][i, j, tau]

                    if p_valor <= 0.05:
                        valor_corr = results['val_matrix'][i, j, tau]

                        lista_results.append({
                            'Cidade': cidade,
                            'Metodo': metodo,
                            'Var_ORIGEM': var_origem,
                            'Var_DESTINO': var_destino,
                            'Lag_Meses': tau,
                            'PVal': round(p_valor, 5),
                            'Val': round(valor_corr, 4)
                        })

    df_final = pd.DataFrame(lista_results)

    df_final['ModuloDoVal'] = df_final['Val'].abs()
    df_final = df_final.sort_values(by=['Cidade', 'ModuloDoVal'], ascending=[True, False]).drop(columns=['ModuloDoVal'])
    return df_final


# ---------------------------------------------------------------------------
# MAIN: one PCMCI pass per method, then all methods concatenated
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
resultados_metodos = []

for metodo in METODOS:
    df_final = rodar_pcmci(metodo)

    print(f"\n--- FINAL DATAFRAME ({metodo}) ---")
    print(df_final)

    caminho = OUTPUT_DIR / ARQUIVOS_SAIDA[metodo]
    df_final.to_csv(caminho, sep=';', decimal=',', index=False)
    print(f"\nDataset saved to: {caminho}")
    resultados_metodos.append(df_final)

df_todos = pd.concat(resultados_metodos, ignore_index=True)
caminho_todos = OUTPUT_DIR / "CausasPCMCI_AllMethods.csv"
df_todos.to_csv(caminho_todos, sep=';', decimal=',', index=False)
print(f"\nAll methods concatenated ({len(df_todos)} rows) saved to: {caminho_todos}")