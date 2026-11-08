"""Separa os dados cronologicamente e descreve os conjuntos, sem treinar modelos.

Execute após 02_log_retornos.py, com os scripts na raiz do projeto.
Execução: python 03_dividir_dados.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# EDITE AQUI: limites inclusivos, sempre em ordem cronológica.
# O início opcional limita as origens; preços anteriores continuam disponíveis
# para calcular retornos históricos. None utiliza toda a cobertura disponível.
INICIO = None
FIM_TREINO = "2022-12-31"
FIM_VALIDACAO = "2024-12-31"
FIM_TESTE = None
HORIZONTES = [1, 5]

# A separação é por data de origem da previsão, com purga nas fronteiras.
# Não há sorteio, padronização, construção de janelas ou ajuste de modelos.
RAIZ = Path(__file__).resolve().parent
ENTRADA = RAIZ / "data" / "trabalho" / "sp500_log_retornos.csv"
PASTA_SAIDA = RAIZ / "data" / "conjuntos"
PASTA_RESUMOS = RAIZ / "outputs" / "conjuntos"


def main():
    # Mantemos todos os fechamentos para alinhar o futuro antes de fazer cortes.
    dados = pd.read_csv(ENTRADA, parse_dates=["data"])
    if dados.empty or dados["data"].isna().any():
        raise ValueError("A base está vazia ou contém datas ausentes.")
    if dados["data"].duplicated().any() or not dados["data"].is_monotonic_increasing:
        raise ValueError("As datas devem ser únicas e estar em ordem crescente.")
    if not (np.isfinite(dados["SP500"]) & (dados["SP500"] > 0)).all():
        raise ValueError("Os preços devem ser positivos, finitos e sem ausências.")
    if not HORIZONTES or any(h not in [1, 5] for h in HORIZONTES):
        raise ValueError("Escolha horizontes 1 e/ou 5, conforme a especificação.")

    inicio = pd.Timestamp(INICIO) if INICIO else dados["data"].min()
    fim_treino = pd.Timestamp(FIM_TREINO)
    fim_validacao = pd.Timestamp(FIM_VALIDACAO)
    fim_teste = pd.Timestamp(FIM_TESTE) if FIM_TESTE else dados["data"].max()
    if not inicio <= fim_treino < fim_validacao < fim_teste:
        raise ValueError("Os limites devem satisfazer início <= treino < validação < teste.")

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    PASTA_RESUMOS.mkdir(parents=True, exist_ok=True)
    contagens = []
    estatisticas = []

    for h in HORIZONTES:
        base = dados.copy()

        # O script 02 datou os retornos pelo fim do período. Aqui a linha t
        # passa a ter o alvo futuro ln(P_{t+h}/P_t), conhecido só em t+h.
        # A coluna data_alvo permite conferir quando o alvo se realiza.
        base["data_alvo"] = dados["data"].shift(-h)
        base["alvo"] = np.log(dados["SP500"].shift(-h) / dados["SP500"])

        # Os limites das origens são comuns aos dois horizontes.
        # A purga exige que o alvo também termine dentro do próprio conjunto.
        cortes = [
            ("treino", base["data"].between(inicio, fim_treino), fim_treino),
            ("validacao", (base["data"] > fim_treino) & (base["data"] <= fim_validacao), fim_validacao),
            ("teste", (base["data"] > fim_validacao) & (base["data"] <= fim_teste), fim_teste),
        ]
        for nome, mascara, limite in cortes:
            candidatos = base.loc[mascara]
            conjunto = candidatos.loc[candidatos["data_alvo"] <= limite].copy()
            if conjunto.empty:
                raise ValueError(f"Conjunto vazio: {nome}, horizonte {h}. Reveja os cortes.")
            conjunto.to_csv(
                PASTA_SAIDA / f"{nome}_{h}d.csv", index=False, date_format="%Y-%m-%d"
            )

            # As estatísticas descrevem os alvos futuros, em unidades decimais.
            # Os primeiros retornos históricos ausentes não são preenchidos:
            # serão tratados quando construirmos as janelas de entrada.
            resumo = conjunto["alvo"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
            resumo["assimetria"] = conjunto["alvo"].skew()
            resumo["curtose_excesso"] = conjunto["alvo"].kurt()
            resumo["conjunto"] = nome
            resumo["horizonte"] = h
            estatisticas.append(resumo)
            contagens.append({
                "conjunto": nome, "horizonte": h,
                "origens_antes_da_purga": len(candidatos),
                "excluidas_por_fronteira_ou_falta_de_futuro": len(candidatos) - len(conjunto),
                "observacoes": len(conjunto),
                "primeira_origem": conjunto["data"].min().date(),
                "ultima_origem": conjunto["data"].max().date(),
                "ultimo_alvo": conjunto["data_alvo"].max().date(),
                "ausentes_log_retorno_1d": int(conjunto["log_retorno_1d"].isna().sum()),
                "ausentes_log_retorno_5d": int(conjunto["log_retorno_5d"].isna().sum()),
            })

    # Registramos os cortes para permitir reproduzir e explicar a separação.
    tabela = pd.DataFrame(contagens)
    descricao = pd.DataFrame(estatisticas).set_index(["conjunto", "horizonte"])
    tabela.to_csv(PASTA_RESUMOS / "contagens.csv", index=False)
    descricao.to_csv(PASTA_RESUMOS / "estatisticas_alvos.csv")
    texto = (
        f"Início das origens: {inicio.date()}\n"
        f"Fim do treino: {fim_treino.date()}\n"
        f"Fim da validação: {fim_validacao.date()}\n"
        f"Fim do teste: {fim_teste.date()}\n"
        "Purga: origem e realização do alvo devem pertencer ao mesmo conjunto.\n"
        "Horizontes contados em fechamentos disponíveis, não dias corridos.\n"
        "Alvo = ln(P_{t+h}/P_t); retornos históricos terminam em t.\n"
        "Estatísticas em unidades decimais; curtose em excesso (normal = 0).\n"
        "As janelas de cinco dias se sobrepõem dentro de cada conjunto.\n"
        "Não usar estatísticas do teste para escolher modelos ou transformações.\n\n"
        + tabela.to_string(index=False) + "\n\n"
        + descricao.to_string(float_format=lambda x: f"{x:.6f}") + "\n"
    )
    (PASTA_RESUMOS / "resumo_divisao.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Conjuntos salvos em: {PASTA_SAIDA}")
    print(f"Resumos salvos em: {PASTA_RESUMOS}")


if __name__ == "__main__":
    main()
