"""Gera log-retornos de 1 e 5 pregões e seus gráficos descritivos.

Execute após 01_baixar_dados.py, com ambos os scripts na raiz do projeto.
Instalação: python -m pip install numpy pandas matplotlib
Execução: python 02_log_retornos.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


# Os caminhos seguem a estrutura criada pelo primeiro script.
RAIZ = Path(__file__).resolve().parent
ENTRADA = RAIZ / "data" / "trabalho" / "sp500.csv"
SAIDA = RAIZ / "data" / "trabalho" / "sp500_log_retornos.csv"
FIGURAS = RAIZ / "outputs" / "figuras"
RESUMO = RAIZ / "outputs" / "resumo_log_retornos.txt"


def main():
    # Lemos a cópia de trabalho sem modificar os arquivos do primeiro script.
    if not ENTRADA.exists():
        raise FileNotFoundError("Execute primeiro o script 01_baixar_dados.py.")
    dados = pd.read_csv(ENTRADA, na_values=["."])
    dados.columns = ["data", "SP500"]
    dados["data"] = pd.to_datetime(dados["data"], errors="raise")
    dados["SP500"] = pd.to_numeric(dados["SP500"], errors="raise")

    # Verificamos a ordem antes de retirar linhas sem fechamento.
    if dados["data"].isna().any() or dados["data"].duplicated().any():
        raise ValueError("Há datas ausentes ou duplicadas.")
    if not dados["data"].is_monotonic_increasing:
        raise ValueError("As datas não estão em ordem crescente.")
    ausentes = int(dados["SP500"].isna().sum())
    precos = dados.set_index("data")["SP500"].dropna()
    if len(precos) < 6 or not (np.isfinite(precos) & (precos > 0)).all():
        raise ValueError("São necessários pelo menos seis preços positivos e finitos.")

    # As defasagens contam fechamentos disponíveis, sem preencher feriados.
    # Se faltar um pregão verdadeiro na fonte, será preciso investigá-lo à parte.
    # O retorno em t termina em t: ainda não estamos construindo alvos futuros.
    resultado = precos.to_frame()
    resultado["log_retorno_1d"] = np.log(precos / precos.shift(1))
    resultado["log_retorno_5d"] = np.log(precos / precos.shift(5))

    # O acumulado logarítmico equivale à soma de cinco log-retornos diários.
    # Ele é diferente do retorno simples acumulado: exp(log_retorno_5d) - 1.
    # Mantemos os valores iniciais indefinidos: um para 1d e cinco para 5d.
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(SAIDA, index_label="data", date_format="%Y-%m-%d")

    # O resumo usa toda a amostra, sem padronização ou divisão temporal.
    colunas = ["log_retorno_1d", "log_retorno_5d"]
    series = resultado[colunas]
    estatisticas = series.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    estatisticas.loc["assimetria"] = series.skew()
    estatisticas.loc["curtose_excesso"] = series.kurt()
    texto = (
        f"Período: {precos.index.min().date()} a {precos.index.max().date()}\n"
        f"Linhas sem preço retiradas apenas nesta base derivada: {ausentes}\n"
        "1d = ln(P_t / P_{t-1}); 5d = ln(P_t / P_{t-5}).\n"
        "Data da linha = fim do período; defasagens em fechamentos válidos.\n"
        "Valores em unidades decimais; assimetria e curtose são adimensionais.\n"
        "Curtose reportada em excesso: normal = 0.\n"
        "Janelas de cinco dias se sobrepõem.\n\n"
        + estatisticas.to_string(float_format=lambda x: f"{x:.6f}")
        + "\n"
    )
    RESUMO.write_text(texto, encoding="utf-8")

    # Cada horizonte recebe uma figura com série temporal, histograma e boxplot.
    # Os gráficos exibem os log-retornos em escala percentual (100 vezes o valor).
    for coluna, horizonte in zip(colunas, [1, 5]):
        serie = resultado[coluna].dropna()
        fig, eixos = plt.subplots(3, 1, figsize=(11, 9))
        eixos[0].plot(serie.index, serie, linewidth=0.7)
        eixos[0].axhline(0, color="gray", linewidth=0.7)
        eixos[0].set(xlabel="Data", ylabel="Log-retorno (%)")
        eixos[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        eixos[1].hist(serie, bins=60, edgecolor="white")
        eixos[1].set(xlabel="Log-retorno (%)", ylabel="Frequência")
        eixos[1].xaxis.set_major_formatter(PercentFormatter(xmax=1))
        eixos[2].boxplot(serie.to_numpy(), vert=False)
        eixos[2].set(xlabel="Log-retorno (%)", yticks=[])
        eixos[2].xaxis.set_major_formatter(PercentFormatter(xmax=1))
        for eixo in eixos:
            eixo.grid(alpha=0.2)
        fig.suptitle(f"S&P 500 — log-retorno acumulado de {horizonte} pregão(ões)")
        fig.tight_layout()
        fig.savefig(FIGURAS / f"log_retornos_{horizonte}d.png", dpi=150)
        plt.close(fig)

    print(texto)
    print(f"Base derivada: {SAIDA}")
    print(f"Gráficos: {FIGURAS}")
    print(f"Resumo: {RESUMO}")


if __name__ == "__main__":
    main()
