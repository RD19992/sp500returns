"""Avalia no teste os três baselines já ajustados pelos scripts 05 e 06.

Execute após congelar as escolhas: python 07_avaliar_baselines_teste.py
Coloque ao lado de 04_metricas.py. Não ajusta nem seleciona modelos.
"""

import importlib.util
import json
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parent
PASTA_CONJUNTOS = RAIZ / "data" / "conjuntos"
ARQUIVO_HISTORICO = RAIZ / "data" / "trabalho" / "sp500_log_retornos.csv"
PASTA_RESULTADOS = RAIZ / "outputs" / "baselines_teste"
HORIZONTES = [1, 5]

# EDITE AQUI: informe o nome da pasta de execução para fixar os modelos usados.
# None seleciona a pasta mais recente que contém todos os modelos necessários.
# As pastas escolhidas serão exibidas e registradas nos resultados.
EXECUCAO_SIMPLES = None
EXECUCAO_AR = None

especificacao = importlib.util.spec_from_file_location("metricas", RAIZ / "04_metricas.py")
metricas = importlib.util.module_from_spec(especificacao)
especificacao.loader.exec_module(metricas)


def escolher_execucao(pasta, nome, arquivos):
    """Localiza uma execução completa, sem combinar modelos de pastas distintas."""
    candidatas = [pasta / nome] if nome else sorted(pasta.glob("*"), reverse=True)
    for candidata in candidatas:
        if all((candidata / arquivo).is_file() for arquivo in arquivos):
            return candidata
    raise FileNotFoundError(f"Execução completa não encontrada em {pasta}. Rode o script correspondente.")


def main():
    simples = escolher_execucao(
        RAIZ / "outputs" / "baselines_simples", EXECUCAO_SIMPLES,
        [f"{modelo}_{h}d.json" for h in HORIZONTES for modelo in ["retorno_zero", "media_historica"]],
    )
    ar = escolher_execucao(
        RAIZ / "outputs" / "baseline_ar", EXECUCAO_AR,
        [f"ar_direto_{h}d.json" for h in HORIZONTES],
    )
    print(f"Modelos simples: {simples}\nModelo AR: {ar}")
    pasta = PASTA_RESULTADOS / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True, exist_ok=False)
    ambiente = {
        "sistema": platform.platform(), "cpu": platform.processor() or "Não identificada",
        "dispositivo": "CPU; GPU não utilizada", "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    linhas = []

    for h in HORIZONTES:
        for modelo in ["retorno_zero", "media_historica", "ar_direto"]:
            inicio = datetime.now().astimezone()
            relogio, cpu = perf_counter(), process_time()
            origem = (ar if modelo == "ar_direto" else simples) / f"{modelo}_{h}d.json"
            ajuste = json.loads(origem.read_text(encoding="utf-8"))
            if ajuste["modelo"] != modelo or ajuste["horizonte"] != h:
                raise ValueError(f"Identificação incorreta no modelo salvo: {origem}")
            teste = pd.read_csv(PASTA_CONJUNTOS / f"teste_{h}d.csv", parse_dates=["data", "data_alvo"])
            if teste.empty or teste[["data", "data_alvo"]].isna().any().any():
                raise ValueError("Teste vazio ou com datas ausentes.")
            if teste.data.duplicated().any() or not teste.data.is_monotonic_increasing:
                raise ValueError("Datas de teste duplicadas ou fora de ordem.")
            if not np.isfinite(teste.alvo).all() or not (teste.data_alvo > teste.data).all():
                raise ValueError("Alvos de teste inválidos.")
            # O teste deve começar depois das origens usadas na validação.
            ultima_validacao = max(r["ultima_origem"] for r in ajuste["resultados"] if r["conjunto"] == "validacao")
            if teste.data.min() <= pd.Timestamp(ultima_validacao):
                raise ValueError("O teste se sobrepõe às origens de validação do modelo salvo.")
            escala = ajuste["escala_mase_treino"]

            # Preparamos apenas entradas conhecidas em cada origem de teste.
            # A série histórica é necessária para recuperar as defasagens do AR.
            if modelo == "ar_direto":
                p = ajuste["ordem_escolhida"]
                historico = pd.read_csv(ARQUIVO_HISTORICO, parse_dates=["data"])
                historico = historico.loc[historico.data <= teste.data.max()].copy()
                if historico.data.isna().any() or historico.data.duplicated().any() or not historico.data.is_monotonic_increasing:
                    raise ValueError("Datas inválidas no histórico do AR.")
                entradas = historico[["data"]].copy()
                nomes = [f"x{j + 1}" for j in range(p)]
                for j, nome in enumerate(nomes):
                    entradas[nome] = historico.log_retorno_1d.shift(j)
                base = teste.merge(entradas, on="data", how="left", validate="one_to_one")
                if not np.isfinite(base[nomes].to_numpy()).all():
                    raise ValueError("Faltam retornos históricos para avaliar o AR nas mesmas datas.")
                x = np.column_stack([np.ones(len(base)), base[nomes].to_numpy()])
                beta = np.array([ajuste["coeficientes"][nome] for nome in ["intercepto"] + nomes])
                if not np.isfinite(beta).all():
                    raise ValueError("Coeficientes salvos inválidos.")
            tempo_preparacao = perf_counter() - relogio

            inicio_previsao = perf_counter()
            previstos = x @ beta if modelo == "ar_direto" else np.full(len(teste), ajuste["constante_prevista"])
            tempo_previsao = perf_counter() - inicio_previsao
            inicio_avaliacao = perf_counter()
            desempenho = metricas.calcular_metricas(teste.alvo, previstos, escala)
            tempo_metricas = perf_counter() - inicio_avaliacao
            tempo_total, tempo_cpu = perf_counter() - relogio, process_time() - cpu
            fim = datetime.now().astimezone()
            registro = {
                "modelo": modelo, "horizonte": h, "conjunto": "teste",
                "observacoes": len(teste), "primeira_origem": teste.data.min().isoformat(),
                "ultima_origem": teste.data.max().isoformat(), **desempenho,
                "inicio": inicio.isoformat(), "fim": fim.isoformat(),
                "tempo_ajuste_s": None, "tempo_preparacao_s": tempo_preparacao,
                "tempo_previsao_s": tempo_previsao, "tempo_metricas_s": tempo_metricas,
                "tempo_total_modelo_s": tempo_total, "tempo_cpu_modelo_s": tempo_cpu,
                "escopo_tempo_total": "Leitura, preparação, previsão e métricas; sem gravação final",
                "arquivo_modelo_origem": str(origem), "escala_mase_treino": escala,
                "ordem": ajuste.get("ordem_escolhida"),
                "politica": "Modelo salvo congelado; sem ajuste, busca ou atualização no teste",
                "ambiente": ambiente,
            }
            (pasta / f"{modelo}_{h}d.json").write_text(
                json.dumps(registro, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
            )
            previsoes = teste[["data", "data_alvo", "alvo"]].copy()
            previsoes["previsao"] = previstos
            previsoes.to_csv(pasta / f"{modelo}_{h}d_previsoes.csv", index=False)
            linhas.append({chave: valor for chave, valor in registro.items() if chave != "ambiente"})

    tabela = pd.DataFrame(linhas)
    tabela.to_csv(pasta / "metricas_teste.csv", index=False)
    print(tabela[["modelo", "horizonte", "observacoes", "MAE", "RMSE", "MASE", "tempo_total_modelo_s"]].to_string(index=False))
    print(f"\nResultados: {pasta}")
    print("MAE e RMSE em decimais; MASE adimensional. Custo de ajuste consta nas execuções de origem.")
    print("Não utilizar estes resultados para ajustar modelos e reapresentar o mesmo teste como independente.")


if __name__ == "__main__":
    main()
