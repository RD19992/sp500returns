"""AR direto para 1 e 5 pregões, com seleção por BIC apenas no treino.

Execute ao lado de 04_metricas.py: python 06_baseline_ar.py
Não abre os conjuntos de teste. Requer somente numpy e pandas.
"""

import importlib.util
import json
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import pandas as pd


# EDITE AQUI: a ordem p é escolhida entre zero e ORDEM_MAXIMA.
# p=0 estima apenas uma constante; não há reestimação na validação.
RAIZ = Path(__file__).resolve().parent
PASTA_CONJUNTOS = RAIZ / "data" / "conjuntos"
ARQUIVO_HISTORICO = RAIZ / "data" / "trabalho" / "sp500_log_retornos.csv"
PASTA_RESULTADOS = RAIZ / "outputs" / "baseline_ar"
HORIZONTES = [1, 5]
ORDEM_MAXIMA = 10

# O nome numérico de 04_metricas.py exige importação pelo caminho do arquivo.
especificacao = importlib.util.spec_from_file_location("metricas", RAIZ / "04_metricas.py")
metricas = importlib.util.module_from_spec(especificacao)
especificacao.loader.exec_module(metricas)


def ler_conjunto(nome, h):
    """Confere as datas e os alvos antes de construir as entradas."""
    dados = pd.read_csv(PASTA_CONJUNTOS / f"{nome}_{h}d.csv", parse_dates=["data", "data_alvo"])
    if dados.empty or dados[["data", "data_alvo"]].isna().any().any():
        raise ValueError(f"Conjunto {nome} vazio ou com datas ausentes.")
    if dados.data.duplicated().any() or not dados.data.is_monotonic_increasing:
        raise ValueError(f"Datas inválidas no conjunto {nome}.")
    if not np.isfinite(dados.alvo).all() or not (dados.data_alvo > dados.data).all():
        raise ValueError(f"Alvos inválidos no conjunto {nome}.")
    return dados


def main():
    if not isinstance(ORDEM_MAXIMA, int) or ORDEM_MAXIMA < 0:
        raise ValueError("ORDEM_MAXIMA deve ser um inteiro não negativo.")
    pasta = PASTA_RESULTADOS / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True, exist_ok=False)
    ambiente = {
        "sistema": platform.platform(), "cpu": platform.processor() or "Não identificada",
        "dispositivo": "CPU; GPU não utilizada", "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    tabelas = []

    for h in HORIZONTES:
        inicio = datetime.now().astimezone()
        inicio_total, inicio_cpu = perf_counter(), process_time()
        treino = ler_conjunto("treino", h)
        validacao = ler_conjunto("validacao", h)
        if treino.data_alvo.max() >= validacao.data.min():
            raise ValueError("Alvos de treino atravessam a fronteira da validação.")

        # A escala da MASE é exatamente a mesma usada no script 05.
        coluna = f"log_retorno_{h}d"
        referencia = treino.dropna(subset=["alvo", coluna])
        escala = metricas.escala_mase(referencia.alvo, referencia[coluna])

        # O histórico completo permite usar retornos anteriores à fronteira na
        # primeira previsão da validação, inclusive datas removidas pela purga.
        # Descartamos datas posteriores à última origem de validação.
        historico = pd.read_csv(ARQUIVO_HISTORICO, parse_dates=["data"])
        historico = historico.loc[historico.data <= validacao.data.max()].copy()
        if historico.data.isna().any() or historico.data.duplicated().any() or not historico.data.is_monotonic_increasing:
            raise ValueError("Histórico com datas ausentes, duplicadas ou fora de ordem.")
        entradas = historico[["data"]].copy()
        nomes = []
        for j in range(ORDEM_MAXIMA):
            # x1 é r_t, já conhecido ao fechamento; x2 é r_{t-1}, etc.
            nome = f"x{j + 1}"
            entradas[nome] = historico.log_retorno_1d.shift(j)
            nomes.append(nome)

        conjuntos = {}
        excluidas = {}
        for nome, dados in [("treino", treino), ("validacao", validacao)]:
            base = dados.merge(entradas, on="data", how="left", validate="one_to_one")
            # Todas as ordens usam as mesmas origens para comparar o BIC.
            base = base.dropna(subset=nomes).copy()
            if base.empty or not np.isfinite(base[nomes].to_numpy()).all():
                raise ValueError(f"Entradas insuficientes ou inválidas em {nome}.")
            conjuntos[nome] = base
            excluidas[nome] = len(dados) - len(base)
        treino_ar = conjuntos["treino"]
        y = treino_ar.alvo.to_numpy()
        if len(y) <= ORDEM_MAXIMA + 1:
            raise ValueError("Poucos exemplos para a ordem máxima escolhida.")
        tempo_preparacao = perf_counter() - inicio_total

        candidatos = []
        coeficientes = []
        for p in range(ORDEM_MAXIMA + 1):
            # Cada ajuste candidato também tem métricas, horários e custo.
            inicio_candidato = datetime.now().astimezone()
            relogio, cpu = perf_counter(), process_time()
            x = np.column_stack([np.ones(len(y)), treino_ar[nomes[:p]].to_numpy()])
            inicio_ajuste = perf_counter()
            beta, _, posto, _ = np.linalg.lstsq(x, y, rcond=None)
            tempo_ajuste = perf_counter() - inicio_ajuste
            if posto != p + 1:
                raise ValueError(f"Regressores linearmente dependentes para p={p}.")
            inicio_previsao = perf_counter()
            previsto = x @ beta
            tempo_previsao = perf_counter() - inicio_previsao
            desempenho = metricas.calcular_metricas(y, previsto, escala)
            # BIC gaussiano, omitindo constantes comuns a todas as ordens.
            # p+1 conta os coeficientes da média; a variância acrescentaria
            # a mesma penalização a todos os candidatos e não muda a escolha.
            mse = float(np.mean((y - previsto) ** 2))
            if mse <= 0:
                raise ValueError("Ajuste perfeito: BIC não é finito nesta amostra.")
            bic = len(y) * np.log(mse) + (p + 1) * np.log(len(y))
            candidatos.append({
                "ordem": p, "BIC": float(bic), "observacoes": len(y),
                "inicio": inicio_candidato.isoformat(),
                "fim": datetime.now().astimezone().isoformat(),
                **desempenho, "conjunto": "treino",
                "tempo_ajuste_s": tempo_ajuste, "tempo_previsao_s": tempo_previsao,
                "tempo_total_s": perf_counter() - relogio, "tempo_cpu_s": process_time() - cpu,
            })
            coeficientes.append(beta)

        # A validação não participa da seleção da ordem nem dos coeficientes.
        melhor = min(candidatos, key=lambda c: c["BIC"])
        p = melhor["ordem"]
        beta = coeficientes[p]
        resultados, previsoes = [], []
        for nome, base in conjuntos.items():
            x = np.column_stack([np.ones(len(base)), base[nomes[:p]].to_numpy()])
            relogio = perf_counter()
            previsto = x @ beta
            tempo_previsao = perf_counter() - relogio
            relogio = perf_counter()
            desempenho = metricas.calcular_metricas(base.alvo, previsto, escala)
            tempo_metricas = perf_counter() - relogio
            resultados.append({
                "modelo": "ar_direto", "horizonte": h, "ordem": p,
                "conjunto": nome, "observacoes": len(base), **desempenho,
                "primeira_origem": base.data.min().isoformat(),
                "ultima_origem": base.data.max().isoformat(),
                "tempo_previsao_s": tempo_previsao, "tempo_metricas_s": tempo_metricas,
            })
            saida = base[["data", "data_alvo", "alvo"]].copy()
            saida["conjunto"], saida["previsao"] = nome, previsto
            previsoes.append(saida)

        tempo_total, tempo_cpu = perf_counter() - inicio_total, process_time() - inicio_cpu
        fim = datetime.now().astimezone()
        registro = {
            "modelo": "ar_direto", "horizonte": h, "ordem_escolhida": p,
            "ordem_maxima": ORDEM_MAXIMA, "inicio": inicio.isoformat(), "fim": fim.isoformat(),
            "politica": "MQO e seleção BIC no treino; coeficientes fixos na validação",
            "formula": "alvo_t = ln(P_{t+h}/P_t); entradas r_t, ..., r_{t-p+1}",
            "coeficientes": dict(zip(["intercepto"] + nomes[:p], beta.tolist())),
            "parametros_estimados_media": p + 1, "semente": None, "epocas": None,
            "escala_mase_treino": escala, "observacoes_escala_mase": len(referencia),
            "origens_excluidas_por_falta_de_historico": excluidas,
            "tempo_preparacao_s": tempo_preparacao,
            "tempo_ajuste_escolhido_s": melhor["tempo_ajuste_s"],
            "tempo_ajuste_todos_candidatos_s": sum(c["tempo_ajuste_s"] for c in candidatos),
            "tempo_total_modelo_s": tempo_total, "tempo_cpu_modelo_s": tempo_cpu,
            "escopo_tempo_total": "Leitura, preparação, busca BIC, previsões e métricas; sem gravação final",
            "ambiente": ambiente, "candidatos_treino": candidatos, "resultados": resultados,
        }
        (pasta / f"ar_direto_{h}d.json").write_text(
            json.dumps(registro, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        pd.DataFrame(candidatos).to_csv(pasta / f"busca_bic_{h}d.csv", index=False)
        pd.concat(previsoes, ignore_index=True).to_csv(pasta / f"ar_direto_{h}d_previsoes.csv", index=False)
        for resultado in resultados:
            tabelas.append({**resultado, "inicio": inicio.isoformat(), "fim": fim.isoformat(),
                            "tempo_total_modelo_s": tempo_total, "tempo_cpu_modelo_s": tempo_cpu})

    tabela = pd.DataFrame(tabelas)
    tabela.to_csv(pasta / "metricas.csv", index=False)
    print(tabela.to_string(index=False))
    print(f"\nResultados: {pasta}")
    print("MAE/RMSE em decimais; MASE adimensional. Treino é ajuste dentro da amostra.")
    print("Tempos totais incluem busca BIC e se repetem entre treino/validação; não somar essas linhas.")


if __name__ == "__main__":
    main()
