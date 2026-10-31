"""
Annual aggregation and SPEARMAN correlation matrix (all variables vs. all
variables) for the integrated panel dataset.

Monthly data (34 municipalities) are aggregated per year and municipality:
fire counts are SUMMED, all other variables are AVERAGED. 2025 is excluded
(incomplete year). The Spearman correlation matrix is then drawn as a heatmap.

Input : datasets/integrated_analysis.csv
Output: results/correlation/spearman_correlation_heatmap.png
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# PATHS (relative to the repository root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
arq = REPO_ROOT / "datasets" / "integrated_analysis.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "correlation"
caminho_figura = OUTPUT_DIR / "spearman_correlation_heatmap.png"

df = pd.read_csv(arq, sep=';', decimal=',')

# --- 1. ANNUAL AGGREGATION (per year and municipality) ---
df['Ano'] = pd.to_datetime(df['Mes/Ano']).dt.year
df = df[df['Ano'] <= 2024]
df_anual = df.groupby(['Ano', 'Match_Cidade']).agg({
    'Qtd_Focos_Local': 'sum',
    'Precipitacao MENSAL(mm)': 'mean',
    'Temperatura Media MENSAL(graus celsius)': 'mean',
    'Anomalia_ENOS': 'mean',
    'Populacao_Estimada': 'mean',
    'Vacas_Ordenhadas_Cabecas': 'mean',
    'Rebanho_Bovino_Cabecas': 'mean',
    'PIB_Mil_Reais': 'mean',
    'Area_Soja_Ha': 'mean',
    'Valor_Soja_MilReais': 'mean',
    'Area_Milho_Ha': 'mean',
    'Valor_Milho_MilReais': 'mean',
    'Preco_Brent': 'mean',
    'Tx_Cambio_Dolar': 'mean'
}).reset_index()

# --- 2. DISPLAY NAMES (figure labels) ---
nomes_bonitos = {
    'Precipitacao MENSAL(mm)': 'Precipitação',
    'Anomalia_ENOS': 'Índice ONI',
    'Qtd_Focos_Local': 'Focos de Incêndio',
    'Temperatura Media MENSAL(graus celsius)': 'Temperatura',
    'Populacao_Estimada': 'População',
    'Vacas_Ordenhadas_Cabecas': 'Vacas Ord.',
    'Rebanho_Bovino_Cabecas': 'Rebanho Bovino',
    'PIB_Mil_Reais': 'PIB Mil Reais',
    'Area_Soja_Ha': 'Ár. Soja Ha',
    'Valor_Soja_MilReais': 'Vl. Soja Mil R$',
    'Area_Milho_Ha': 'Ár. Milho Ha',
    'Valor_Milho_MilReais': 'Vl. Milho Mil R$',
    'Preco_Brent': 'Vl. Petróleo',
    'Tx_Cambio_Dolar': 'Preço Dolar'
}

df_final = df_anual.rename(columns=nomes_bonitos)

# Keep only numeric columns (the correlation only accepts numbers)
colunas_para_matriz = ['Precipitação', 'Índice ONI', 'Focos de Incêndio', 'Temperatura',
                       'População', 'Vacas Ord.', 'Rebanho Bovino', 'PIB Mil Reais',
                       'Ár. Soja Ha', 'Vl. Soja Mil R$', 'Ár. Milho Ha', 'Vl. Milho Mil R$',
                       'Vl. Petróleo', 'Preço Dolar']

df_calculo = df_final[colunas_para_matriz]

# --- 3. SPEARMAN CORRELATION MATRIX ---
corr = df_calculo.corr(method='spearman')

# --- 4. HEATMAP ---
plt.figure(figsize=(10, 8))

sns.heatmap(corr,
            annot=True,
            fmt=".3f",
            cmap='coolwarm',
            vmin=-1, vmax=1,
            linewidths=1,
            square=True)

plt.xticks(fontsize=10, rotation=45, ha='right')
plt.yticks(fontsize=10, rotation=0)

plt.title('Matriz de Correlação SPEARMAN: Todos vs Todos\n(Dados Anuais Agrupados)', fontsize=16, pad=20)
plt.tight_layout()

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(caminho_figura, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to:\n{caminho_figura}")

plt.show()