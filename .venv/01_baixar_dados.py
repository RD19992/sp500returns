"""Baixa e confere o SP500 do FRED, preserva a fonte e gera três gráficos.

Instalação: python -m pip install pandas matplotlib
Execução: python 01_baixar_dados.py
"""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter


# Os caminhos partem da pasta deste script, independentemente do terminal usado.
# Coloque este arquivo na raiz de sp500returns, ao lado do README.md.
RAIZ = Path(__file__).resolve().parent
REFERENCIA = RAIZ / "data" / "referencia" / "sp500_fred.csv"
TRABALHO = RAIZ / "data" / "trabalho" / "sp500.csv"
FIGURAS = RAIZ / "outputs" / "figuras"
RELATORIO = RAIZ / "outputs" / "verificacao_dados.txt"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"


def main():
    # Uma execução posterior reutiliza a referência para manter a mesma amostra.
    # A cópia de trabalho pode ser recriada; a referência nunca é sobrescrita.
    if REFERENCIA.exists():
        conteudo = REFERENCIA.read_bytes()
        origem = "Referência existente reutilizada; nenhum novo download."
    else:
        requisicao = Request(URL, headers={"User-Agent": "MAE5920-SP500/1.0"})
        with urlopen(requisicao, timeout=60) as resposta:
            conteudo = resposta.read()
        origem = "Dados baixados do FRED nesta execução."

    # O FRED pode representar valores ausentes por ponto ou campo vazio.
    # A leitura é feita em memória; os arquivos salvos manterão os bytes originais.
    dados = pd.read_csv(BytesIO(conteudo), na_values=["."])
    if len(dados.columns) != 2 or dados.columns[1] != "SP500":
        raise ValueError("Formato inesperado: eram esperadas data e coluna SP500.")
    dados.columns = ["data", "SP500"]
    dados["data"] = pd.to_datetime(dados["data"], errors="raise")
    dados["SP500"] = pd.to_numeric(dados["SP500"], errors="raise")

    # Interrompemos em problemas estruturais, sem ordenar ou corrigir a fonte.
    if dados.empty or dados["data"].isna().any():
        raise ValueError("A base está vazia ou contém datas ausentes.")
    if dados["data"].duplicated().any():
        raise ValueError("Há datas duplicadas na fonte.")
    if not dados["data"].is_monotonic_increasing:
        raise ValueError("As datas da fonte não estão em ordem crescente.")
    if (dados["data"] > pd.Timestamp.now().normalize()).any():
        raise ValueError("A fonte contém datas futuras.")
    valores = dados["SP500"].dropna()
    if len(valores) < 2 or not ((valores > 0) & (valores < float("inf"))).all():
        raise ValueError("São necessários dois preços válidos, positivos e finitos.")

    # Esta comparação detecta dias de segunda a sexta sem linha no CSV.
    # Não é um calendário de pregões: feriados também podem aparecer nas lacunas.
    dias_uteis = pd.bdate_range(dados["data"].min(), dados["data"].max())
    sem_linha = dias_uteis.difference(pd.DatetimeIndex(dados["data"]))
    ausentes = dados.loc[dados["SP500"].isna(), "data"]
    fins_de_semana = int((dados["data"].dt.dayofweek >= 5).sum())

    # Removemos ausências apenas da série temporária usada nos gráficos.
    # Não preenchemos feriados, não aplicamos log e não alteramos a base salva.
    precos = dados.set_index("data")["SP500"].dropna()
    retorno_diario = precos.pct_change(fill_method=None)
    retorno_acumulado = precos / precos.iloc[0] - 1

    # O hash permite reconhecer a cópia exata da fonte usada no projeto.
    resumo = [
        origem,
        f"Fonte: {URL}",
        f"Verificação (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"SHA-256: {sha256(conteudo).hexdigest()}",
        f"Linhas: {len(dados)}; preços válidos: {len(precos)}",
        f"Datas da fonte: {dados['data'].min().date()} a {dados['data'].max().date()}",
        f"Preços válidos: {precos.index.min().date()} a {precos.index.max().date()}",
        "Datas válidas, únicas e em ordem crescente: OK",
        f"Preços ausentes: {len(ausentes)}",
        "Datas com preço ausente: " + ", ".join(ausentes.dt.strftime("%Y-%m-%d")),
        f"Dias de segunda a sexta sem linha: {len(sem_linha)}",
        "Datas sem linha: " + ", ".join(sem_linha.strftime("%Y-%m-%d")),
        f"Linhas em fins de semana: {fins_de_semana}",
        "Ausências e lacunas exigem conferência; feriados não são erros por si só.",
        "Sem calendário da bolsa, esta verificação não certifica todos os pregões.",
        "Retorno diário: variação simples entre fechamentos válidos consecutivos.",
        "Acumulado: variação simples desde o primeiro fechamento válido.",
        "O índice não inclui dividendos; nenhum log-retorno foi calculado.",
        "As duas cópias CSV mantêm o conteúdo original, inclusive ausências.",
    ]

    # Só salvamos após as verificações estruturais. O modo xb impede sobrescrita.
    REFERENCIA.parent.mkdir(parents=True, exist_ok=True)
    TRABALHO.parent.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    if not REFERENCIA.exists():
        with REFERENCIA.open("xb") as arquivo:
            arquivo.write(conteudo)
        REFERENCIA.with_suffix(".sha256").write_text(
            sha256(conteudo).hexdigest() + "\n", encoding="utf-8"
        )
        REFERENCIA.with_suffix(".txt").write_text(
            "\n".join(resumo) + "\n", encoding="utf-8"
        )
    else:
        hash_original = REFERENCIA.with_suffix(".sha256").read_text().strip()
        if sha256(conteudo).hexdigest() != hash_original:
            raise ValueError("A referência foi alterada desde sua criação.")
    TRABALHO.write_bytes(conteudo)
    RELATORIO.write_text("\n".join(resumo) + "\n", encoding="utf-8")

    # Cada gráfico é salvo separadamente para facilitar inspeção e uso posterior.
    graficos = [
        (precos, "S&P 500 — fechamento diário", "Pontos", "01_indice.png"),
        (retorno_diario, "S&P 500 — retorno diário simples", "Retorno (%)", "02_retorno_diario.png"),
        (retorno_acumulado, "S&P 500 — retorno acumulado desde o início", "Retorno (%)", "03_retorno_acumulado.png"),
    ]
    for serie, titulo, eixo_y, nome in graficos:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(serie.index, serie, linewidth=0.8)
        ax.set(title=titulo, xlabel="Data", ylabel=eixo_y)
        if eixo_y == "Retorno (%)":
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.axhline(0, color="gray", linewidth=0.6)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURAS / nome, dpi=150)
        plt.close(fig)

    print("\n".join(resumo))
    print(f"\nCópia de trabalho: {TRABALHO}")
    print(f"Gráficos: {FIGURAS}")
    print(f"Relatório: {RELATORIO}")


if __name__ == "__main__":
    main()
