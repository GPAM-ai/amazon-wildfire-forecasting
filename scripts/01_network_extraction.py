"""
Chronological network (chronnet) extraction and topological metrics from
NASA MODIS active-fire detections over the Amazon region.

For each month of each year, fire events (confidence > 70) are mapped to a
0.5 x 0.5 degree grid and connected in chronological order, producing a
directed weighted network that is then simplified to an undirected weighted
one. Monthly metrics: number of nodes/edges, fire count, mean degree, degree
entropy, max k-core, mean betweenness, number of communities and modularity.

Input data
---------------------------------------------
MODIS Collection 6.1 active fire archive, downloaded per year from NASA FIRMS
(Fire Information for Resource Management System):
    https://firms.modaps.eosdis.nasa.gov/download/
Files are named like "2025.fire_archive_M-C61_<request_id>.csv"
Required columns: latitude, longitude, acq_date, acq_time, confidence.

Usage (from anywhere):
    python scripts/01_network_extraction.py              # all years, 2003-2025
    python scripts/01_network_extraction.py 2024 2025    # only the given years

Output (in results/network_extraction/):
    NOVO_Metricas_AMZ_Ano<YEAR>.csv   one file per year
    Metricas_AMZ_AllYears.csv         all processed years concatenated
"""
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# Years to process: command-line arguments if given, otherwise 2003 to 2025.
ANOS = [int(a) for a in sys.argv[1:]] or list(range(2003, 2026))

# Regions as (lat_min, lat_max, lon_min, lon_max), based on
# "Graph-based semi-supervised classification for similar wildfire dynamics"
regioes = {
    "AMZ": (-15, 5, -70, -50)
}

# Paths relative to the repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
NASA_DIR = REPO_ROOT / "raw_data" / "nasa_firms"
caminhoSaidaPanda = REPO_ROOT / "results" / "network_extraction"

# Grid division on each region in a way that the grid cells
# correspond to rectangles of size 0.5 x 0.5 degrees, producing 1600 grid cells
tamCelula = 0.5


def processar_ano(ano):
    """Runs the full chronnet extraction for one year.
    Returns {region name: DataFrame of monthly metrics}, or None if the
    year's FIRMS file is not available."""
    # Picks the FIRMS file of the year (the request id in the name varies)
    arquivos_ano = sorted(NASA_DIR.glob(f"{ano}.fire_archive_*.csv"))
    if not arquivos_ano:
        print(f"\n[WARNING] No file '{ano}.fire_archive_*.csv' in {NASA_DIR}. Skipping {ano}.")
        return None
    arquivo = arquivos_ano[0]
    nomeArquivo = os.path.basename(arquivo)  # file name only
    Ano = nomeArquivo.split('.')[0]          # first token of the name (the year)

    print(f"\n########## Year {Ano} ({nomeArquivo}) ##########")

    Graph = pd.read_csv(arquivo)
    Graph['acq_datetime'] = pd.to_datetime(
        Graph['acq_date'] + ' ' + Graph['acq_time'].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M"
    )
    Graph['month'] = Graph['acq_datetime'].dt.month   # used to filter events by month
    Graph = Graph[Graph['confidence'] > 70].copy()    # only detections with confidence > 70

    listaMetricasMensais = []
    dfs_regioes = {}

    for nome_regiao, (lat_min, lat_max, lon_min, lon_max) in regioes.items():
        print(f"\nRegion: {nome_regiao}")

        num_celulas_x = int((lon_max - lon_min) / tamCelula)
        num_celulas_y = int((lat_max - lat_min) / tamCelula)
        total_celulas_possiveis = num_celulas_x * num_celulas_y

        barreira = Graph[
            (Graph['latitude'] >= lat_min) & (Graph['latitude'] <= lat_max) &
            (Graph['longitude'] >= lon_min) & (Graph['longitude'] <= lon_max)
        ].copy()

        if barreira.empty:
            print(f"{nome_regiao} has no data")
            continue

        celulasAtivasMes = set()

        for mes in range(1, 13):
            DataFr_mes = barreira[barreira['month'] == mes]
            DataFr_mes = DataFr_mes.sort_values(by='acq_datetime').reset_index(drop=True)

            if DataFr_mes.empty:
                print(f"- Month {mes} has no data")
                continue

            DataFr_mes['x_Celula'] = np.floor((DataFr_mes['longitude'] - lon_min) / tamCelula).astype(int)
            DataFr_mes['y_Celula'] = np.floor((DataFr_mes['latitude'] - lat_min) / tamCelula).astype(int)

            numFirecount = len(DataFr_mes)

            GridCelula = list(zip(DataFr_mes['x_Celula'], DataFr_mes['y_Celula']))

            celulasAtivasMes.update(GridCelula)

            G_CelulaDir = nx.DiGraph()

            for i in range(len(GridCelula) - 1):
                origem = GridCelula[i]
                destino = GridCelula[i + 1]
                if origem != destino:
                    if G_CelulaDir.has_edge(origem, destino):
                        G_CelulaDir[origem][destino]['weight'] += 1
                    else:
                        G_CelulaDir.add_edge(origem, destino, weight=1)

            print(f"- Month {mes}")
            print(G_CelulaDir)

            numNos = G_CelulaDir.number_of_nodes()
            numArest = G_CelulaDir.number_of_edges()

            G_NaoDirec = G_CelulaDir.to_undirected()
            for u in G_CelulaDir:
                for v in nx.neighbors(G_CelulaDir, u):
                    if u in nx.neighbors(G_CelulaDir, v):
                        peso_total = G_CelulaDir[u][v]['weight'] + G_CelulaDir[v][u]['weight']
                        G_NaoDirec[u][v]['weight'] = peso_total

            if numNos > 0:
                GrauDeCadaNo = G_NaoDirec.degree(weight='weight')
                listaValoresGraus = [grau for no, grau in GrauDeCadaNo]
                mediaGrau = np.mean(listaValoresGraus)
            else:
                mediaGrau = 0

            if numNos <= 1:
                entropia = 0
            else:
                graus = [d for no, d in GrauDeCadaNo]
                grausUnicos, NosComGrauK = np.unique(graus, return_counts=True)
                if len(grausUnicos) <= 1:
                    entropia = 0
                else:
                    Pk = NosComGrauK / numNos
                    entropia = -np.sum(Pk * np.log(Pk)) / np.log(numNos)

            if numNos > 0:
                kCoreMax = max(nx.core_number(G_NaoDirec).values())
            else:
                kCoreMax = 0

            if numNos > 0:
                betweenness = nx.betweenness_centrality(G_NaoDirec, k=None, normalized=True, weight='weight', endpoints=False, seed=None)
                mediaBetweenness = np.mean(list(betweenness.values()))
            else:
                mediaBetweenness = 0

            if numNos > 0:
                comunidades = nx.community.greedy_modularity_communities(G_NaoDirec, weight='weight', resolution=1, cutoff=1, best_n=None)  # Clauset algorithm
                numComunidades = len(comunidades)

                modularidade = nx.community.modularity(G_NaoDirec, comunidades, weight='weight', resolution=1)  # (whole network, communities to compare)
            else:
                numComunidades = 0
                modularidade = 0

            mesLabel = f"{mes:02d}/{Ano}"
            listaMetricasMensais.append({
                'Mes/Ano': mesLabel,
                'Nos': numNos,
                'Arestas': numArest,
                'FireCount': numFirecount,
                'Grau Medio Nos(K)': mediaGrau,
                'Entropia': entropia,
                'KCore': kCoreMax,
                'Betweenness': mediaBetweenness,
                'Comunidades': numComunidades,
                'Modularidade': modularidade
            })

        print("Process completed")

        df_final = pd.DataFrame(listaMetricasMensais)
        os.makedirs(caminhoSaidaPanda, exist_ok=True)
        NomeArqTabelaFinal = f"NOVO_Metricas_{nome_regiao}_Ano{Ano}.csv"
        nome_arquivo_final = os.path.join(caminhoSaidaPanda, NomeArqTabelaFinal)

        df_final.to_csv(nome_arquivo_final, index=False, sep=';', decimal=',')
        dfs_regioes[nome_regiao] = df_final

        print(f"\nSUCCESS! DataFrame saved to: {nome_arquivo_final}")

    return dfs_regioes


# ---------------------------------------------------------------------------
# MAIN: one pass per year, then concatenation of all processed years
# ---------------------------------------------------------------------------
resultados_anos = []
for ano in ANOS:
    dfs = processar_ano(ano)
    if dfs:
        resultados_anos.append(dfs)

if not resultados_anos:
    raise FileNotFoundError(
        f"No FIRMS file found in {NASA_DIR} for the requested years. "
        "Download them from NASA FIRMS (MODIS C6.1) and place them there."
    )

for nome_regiao in regioes:
    dfs_todos = [r[nome_regiao] for r in resultados_anos if nome_regiao in r]
    if not dfs_todos:
        continue
    df_todos = pd.concat(dfs_todos, ignore_index=True)
    caminho_todos = os.path.join(caminhoSaidaPanda, f"Metricas_{nome_regiao}_AllYears.csv")
    df_todos.to_csv(caminho_todos, index=False, sep=';', decimal=',')
    print(f"\nAll years concatenated ({len(dfs_todos)} years, {len(df_todos)} rows) saved to: {caminho_todos}")