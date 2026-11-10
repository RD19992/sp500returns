"""Avalia retorno zero e média histórica no treino e na validação, para 1d e 5d.

Coloque ao lado de 04_metricas.py e execute: python 05_baselines_simples.py
A média é estimada apenas no treino e fica fixa na validação.
"""

import importlib.util
import json
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import pandas as pd


# EDITE AQUI se mudar a localização dos conjuntos ou os horizontes avaliados.
# O caminho abaixo segue o script 03 mostrado na sua captura de tela.
RAIZ = Path(__file__).resolve().parent
PASTA_CONJUNTOS = RAIZ / "data" / "conjuntos"
PASTA_RESULTADOS = RAIZ / "outputs" / "baselines_simples"
HORIZONTES = [1, 5]

# Arquivos iniciados por números precisam ser importados desta forma.
especificacao = importlib.util.spec_from_file_location("metricas", RAIZ / "04_metricas.py")
metricas = importlib.util.module_from_spec(especificacao)
especificacao.loader.exec_module(metricas)


def ler_conjunto(nome, horizonte):
    """Lê e confere somente o conjunto solicitado, sem abrir arquivos de teste."""
    caminho = PASTA_CONJUNTOS / f"{nome}_{horizonte}d.csv"
    dados = pd.read_csv(caminho, parse_dates=["data", "data_alvo"])
    if dados.empty or dados[["data", "data_alvo"]].isna().any().any():
        raise ValueError(f"Datas ausentes ou conjunto vazio: {caminho}")
    if dados["data"].duplicated().any() or not dados["data"].is_monotonic_increasing:
        raise ValueError(f"Datas duplicadas ou fora de ordem: {caminho}")
    if not np.isfinite(dados["alvo"]).all():
        raise ValueError(f"Alvos ausentes ou infinitos: {caminho}")
    if not (dados["data_alvo"] > dados["data"]).all():
        raise ValueError(f"O alvo deve se realizar depois da origem: {caminho}")
    return dados


def main():
    # Uma pasta nova preserva os resultados de cada chamada do script.
    identificador = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    pasta_execucao = PASTA_RESULTADOS / identificador
    pasta_execucao.mkdir(parents=True, exist_ok=False)
    linhas = []
    ambiente = {
        "sistema": platform.platform(),
        "cpu": platform.processor() or "Identificação não disponível",
        "dispositivo": "CPU; GPU não utilizada",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    for h in HORIZONTES:
        # A preparação compartilhada é medida separadamente do custo dos modelos.
        inicio_preparacao = perf_counter()
        treino = ler_conjunto("treino", h)
        validacao = ler_conjunto("validacao", h)
        if treino["data_alvo"].max() >= validacao["data"].min():
            raise ValueError("Há alvos de treino atravessando a fronteira da validação.")
        coluna_persistencia = f"log_retorno_{h}d"
        referencia = treino.dropna(subset=["alvo", coluna_persistencia])
        escala = metricas.escala_mase(referencia["alvo"], referencia[coluna_persistencia])
        tempo_preparacao = perf_counter() - inicio_preparacao

        for modelo in ["retorno_zero", "media_historica"]:
            # Não há otimização iterativa: zero não tem ajuste, e média estima
            # uma constante no treino. Não calculamos a média da validação.
            inicio = datetime.now().astimezone()
            inicio_total = perf_counter()
            inicio_cpu = process_time()
            if modelo == "media_historica":
                inicio_ajuste = perf_counter()
                constante = float(treino["alvo"].mean())
                tempo_ajuste = perf_counter() - inicio_ajuste
            else:
                constante = 0.0
                tempo_ajuste = None

            previsoes = []
            resultados = []
            for nome, dados in [("treino", treino), ("validacao", validacao)]:
                inicio_previsao = perf_counter()
                previstos = np.full(len(dados), constante)
                tempo_previsao = perf_counter() - inicio_previsao
                inicio_avaliacao = perf_counter()
                desempenho = metricas.calcular_metricas(dados["alvo"], previstos, escala)
                tempo_avaliacao = perf_counter() - inicio_avaliacao
                resultados.append({
                    "conjunto": nome, "observacoes": len(dados),
                    "primeira_origem": dados["data"].min().isoformat(),
                    "ultima_origem": dados["data"].max().isoformat(),
                    **desempenho,
                    "tempo_previsao_s": tempo_previsao,
                    "tempo_metricas_s": tempo_avaliacao,
                })
                previsao = dados[["data", "data_alvo", "alvo"]].copy()
                previsao["conjunto"] = nome
                previsao["previsao"] = previstos
                previsoes.append(previsao)

            # Tempo de parede e tempo de CPU são diferentes e ambos são salvos.
            # Os totais abaixo excluem preparação compartilhada e gravação final.
            tempo_cpu = process_time() - inicio_cpu
            tempo_total = perf_counter() - inicio_total
            fim = datetime.now().astimezone()
            registro = {
                "modelo": modelo, "horizonte": h,
                "inicio": inicio.isoformat(), "fim": fim.isoformat(),
                "constante_prevista": constante,
                "politica": "Constante estimada no treino e congelada na validação",
                "parametros_estimados": int(modelo == "media_historica"),
                "semente": None, "epocas": None,
                "escala_mase_treino": escala,
                "observacoes_escala_mase": len(referencia),
                "pasta_dados": str(PASTA_CONJUNTOS),
                "tempo_preparacao_compartilhada_s": tempo_preparacao,
                "tempo_ajuste_s": tempo_ajuste,
                "tempo_previsao_s": sum(r["tempo_previsao_s"] for r in resultados),
                "tempo_total_modelo_s": tempo_total, "tempo_cpu_modelo_s": tempo_cpu,
                "escopo_tempo_total": "Cálculo do modelo, previsão, métricas e montagem das tabelas; sem leitura e gravação",
                "ambiente": ambiente, "resultados": resultados,
            }
            nome_arquivo = f"{modelo}_{h}d"
            (pasta_execucao / f"{nome_arquivo}.json").write_text(
                json.dumps(registro, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
            )
            pd.concat(previsoes, ignore_index=True).to_csv(
                pasta_execucao / f"{nome_arquivo}_previsoes.csv", index=False
            )
            for resultado in resultados:
                linhas.append({
                    "modelo": modelo, "horizonte": h, **resultado,
                    "inicio": inicio.isoformat(), "fim": fim.isoformat(),
                    "tempo_ajuste_s": tempo_ajuste,
                    "tempo_total_modelo_s": tempo_total,
                    "tempo_cpu_modelo_s": tempo_cpu,
                })

    # Os tempos totais se repetem nas duas linhas do mesmo modelo/horizonte:
    # são custos da execução conjunta, não devem ser somados entre conjuntos.
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(pasta_execucao / "metricas.csv", index=False)
    print(tabela.to_string(index=False))
    print(f"\nResultados salvos em: {pasta_execucao}")
    print("MAE e RMSE em log-retornos decimais; MASE adimensional.")
    print("Tempos muito curtos variam entre execuções; não arredondamos os arquivos.")


if __name__ == "__main__":
    main()
