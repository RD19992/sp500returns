"""Experimentos controlados de CNN para retornos de 1 e 5 pregões, com treino e validação.

Coloque ao lado de 04_metricas.py. Execute com python 10_experimentos_cnn.py.
Instale numpy, pandas, matplotlib e a versão do PyTorch adequada à sua GPU.
"""

import copy
import importlib.util
import json
import platform
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


# EDITE AQUI. A configuração inteira é salva em cada run e no checkpoint.
CONFIG = {
    "horizontes": [1, 5],
    "janela": 5,
    "profundidade": 2,
    "filtros": 8,
    "kernel": 3,
    "dilatacoes": [1, 1],
    "padding": "causal",                 # Opções: causal ou valid.
    "inicializacao": "he",               # Opções: he ou xavier.
    "otimizador": "adam",                # Opções: adam, adamw ou sgd.
    "taxa_aprendizagem": 0.001,
    "weight_decay": 0.0,
    "batch_size": 256,
    "epocas": 200,
    "paciencia": 20,
    "semente": 42,
    "dispositivo": "cuda",               # Opções: auto, cuda ou cpu.
    "threads_cpu": 4,
}
# Cada linha modifica a configuração de referência. Com leitura na última posição,
# aumentar a janela além do campo receptivo não acrescenta informação à previsão.
SEMENTES = [42, 43, 44]
EXPERIMENTOS = {
    "minima": {},
    "profundidade_1": {"profundidade": 1, "dilatacoes": [1]},
    "profundidade_4": {"profundidade": 4, "dilatacoes": [1, 1, 1, 1]},
    "kernel_1": {"kernel": 1},
    "kernel_5": {"kernel": 5},
    "largura_4": {"filtros": 4},
    "largura_32": {"filtros": 32},
    "dilatacao_2": {"dilatacoes": [1, 2]},
    "inicializacao_xavier": {"inicializacao": "xavier"},
    "sgd_momentum": {"otimizador": "sgd"},
    "adam_com_penalizacao": {"weight_decay": 0.01},
    "adamw_com_penalizacao": {"otimizador": "adamw", "weight_decay": 0.01},
    "taxa_menor": {"taxa_aprendizagem": 0.0003},
    "pooling_medio": {"leitura": "media"},
}
CONFIG["leitura"] = "ultima"

RAIZ = Path(__file__).resolve().parent
PASTA_CONJUNTOS = RAIZ / "data" / "conjuntos"
ARQUIVO_HISTORICO = RAIZ / "data" / "trabalho" / "sp500_log_retornos.csv"
PASTA_RESULTADOS = RAIZ / "outputs" / "experimentos_cnn"

# Importamos as mesmas métricas usadas nos baselines.
especificacao = importlib.util.spec_from_file_location("metricas", RAIZ / "04_metricas.py")
metricas = importlib.util.module_from_spec(especificacao)
especificacao.loader.exec_module(metricas)


def registrar(texto, pasta):
    linha = f"{datetime.now().astimezone().isoformat()} | {texto}"
    print(linha, flush=True)
    with (pasta / "run.log").open("a", encoding="utf-8") as arquivo:
        arquivo.write(linha + "\n")


def salvar_json(caminho, conteudo):
    # A troca do marcador é atômica, evitando um último run parcialmente gravado.
    temporario = caminho.with_suffix(".tmp")
    temporario.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporario.replace(caminho)


def sincronizar(dispositivo):
    # CUDA trabalha de forma assíncrona. Sincronizamos para medir tempos reais.
    if dispositivo.type == "cuda":
        torch.cuda.synchronize(dispositivo)


class CNN(nn.Module):
    """Convoluções temporais com ReLU e leitura na última posição."""

    def __init__(self, config):
        super().__init__()
        self.padding = config["padding"]
        self.leitura = config["leitura"]
        self.convolucoes = nn.ModuleList()
        canais = 1
        for dilatacao in config["dilatacoes"]:
            camada = nn.Conv1d(canais, config["filtros"], config["kernel"], dilation=dilatacao)
            if config["inicializacao"] == "he":
                nn.init.kaiming_normal_(camada.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(camada.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(camada.bias)
            self.convolucoes.append(camada)
            canais = config["filtros"]
        self.saida = nn.Linear(canais, 1)
        nn.init.xavier_normal_(self.saida.weight)
        nn.init.zeros_(self.saida.bias)

    def forward(self, entradas):
        for camada in self.convolucoes:
            if self.padding == "causal":
                esquerda = (camada.kernel_size[0] - 1) * camada.dilation[0]
                entradas = F.pad(entradas, (esquerda, 0))
            entradas = F.relu(camada(entradas))
        resumo = entradas[:, :, -1] if self.leitura == "ultima" else entradas.mean(dim=-1)
        return self.saida(resumo).squeeze(-1)


def treinar(h, pasta, dispositivo, config, janela_comum):
    inicio = datetime.now().astimezone()
    inicio_total, inicio_cpu = perf_counter(), process_time()
    registrar(f"Início h={h}, dispositivo={dispositivo}", pasta)
    random.seed(config["semente"])
    np.random.seed(config["semente"])
    torch.manual_seed(config["semente"])
    if dispositivo.type == "cuda":
        torch.cuda.manual_seed_all(config["semente"])
        torch.cuda.reset_peak_memory_stats(dispositivo)
    # Evitamos seleção automática de algoritmos que varie durante a execução.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    conjuntos = {
        nome: pd.read_csv(PASTA_CONJUNTOS / f"{nome}_{h}d.csv", parse_dates=["data", "data_alvo"])
        for nome in ["treino", "validacao"]
    }
    treino, validacao = conjuntos["treino"], conjuntos["validacao"]
    for nome, dados in conjuntos.items():
        if dados.empty or dados[["data", "data_alvo"]].isna().any().any():
            raise ValueError(f"Datas ausentes ou conjunto vazio em {nome}.")
        if dados.data.duplicated().any() or not dados.data.is_monotonic_increasing:
            raise ValueError(f"Datas duplicadas ou fora de ordem em {nome}.")
        if not np.isfinite(dados.alvo).all() or not (dados.data_alvo > dados.data).all():
            raise ValueError(f"Alvos inválidos em {nome}.")
    if treino.data_alvo.max() >= validacao.data.min():
        raise ValueError("Há alvos de treino atravessando a validação.")
    referencia = treino.dropna(subset=["alvo", f"log_retorno_{h}d"])
    escala_mase = metricas.escala_mase(referencia.alvo, referencia[f"log_retorno_{h}d"])

    historico = pd.read_csv(ARQUIVO_HISTORICO, parse_dates=["data"])
    historico = historico.loc[historico.data <= validacao.data.max()].copy()
    if historico.data.isna().any() or historico.data.duplicated().any() or not historico.data.is_monotonic_increasing:
        raise ValueError("Datas inválidas no histórico.")
    # Cada janela termina na origem t. A primeira validação usa o passado disponível.
    retornos = historico.set_index("data").log_retorno_1d
    entradas = pd.concat([retornos.shift(j) for j in range(janela_comum - 1, -1, -1)], axis=1)
    escalas = retornos.loc[(retornos.index >= treino.data.min()) & (retornos.index <= treino.data.max())].dropna()
    media_x, desvio_x = float(escalas.mean()), float(escalas.std(ddof=0))
    media_y, desvio_y = float(treino.alvo.mean()), float(treino.alvo.std(ddof=0))
    if not np.isfinite([media_x, desvio_x, media_y, desvio_y]).all() or min(desvio_x, desvio_y) <= 0:
        raise ValueError("Escalas de treino inválidas.")
    tensores, excluidas = {}, {}
    for nome, dados in conjuntos.items():
        valores = entradas.reindex(dados.data).to_numpy()
        disponiveis = np.isfinite(valores).all(axis=1)
        conjuntos[nome] = dados.loc[disponiveis].reset_index(drop=True)
        excluidas[nome] = int((~disponiveis).sum())
        if not disponiveis.any():
            raise ValueError(f"Sem janelas completas em {nome}.")
        # Todos os candidatos usam as mesmas origens; cada CNN vê só a própria janela.
        x = ((valores[disponiveis, -config["janela"]:] - media_x) / desvio_x).astype("float32")
        y = ((conjuntos[nome].alvo.to_numpy() - media_y) / desvio_y).astype("float32")
        # Toda a base pequena é transferida uma vez. Os lotes são recortados na GPU.
        tensores[nome] = (torch.from_numpy(x[:, None, :]).to(dispositivo), torch.from_numpy(y).to(dispositivo))
    modelo = CNN(config).to(dispositivo)
    otimizadores = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    argumentos = {"lr": config["taxa_aprendizagem"], "weight_decay": config["weight_decay"]}
    if config["otimizador"] == "sgd":
        argumentos["momentum"] = 0.9
    otimizador = otimizadores[config["otimizador"]](modelo.parameters(), **argumentos)
    perda = nn.MSELoss()
    sincronizar(dispositivo)
    tempo_preparacao = perf_counter() - inicio_total
    registrar(f"Parâmetros={sum(p.numel() for p in modelo.parameters())}, origens excluídas={excluidas}", pasta)

    curvas, melhor_rmse, melhor_epoca, sem_melhora = [], float("inf"), 0, 0
    # A ordem dos lotes é pareada inclusive quando muda o número de pesos.
    gerador = torch.Generator(device=dispositivo).manual_seed(config["semente"])
    inicio_ajuste = perf_counter()
    for epoca in range(1, config["epocas"] + 1):
        sincronizar(dispositivo)
        inicio_epoca = perf_counter()
        modelo.train()
        x_treino, y_treino = tensores["treino"]
        ordem = torch.randperm(len(y_treino), device=dispositivo, generator=gerador)
        for lote in ordem.split(config["batch_size"]):
            otimizador.zero_grad(set_to_none=True)
            erro = perda(modelo(x_treino[lote]), y_treino[lote])
            erro.backward()
            otimizador.step()
        modelo.eval()
        linha = {"epoca": epoca}
        with torch.inference_mode():
            for nome, (x, y) in tensores.items():
                previstos = modelo(x).cpu().numpy().astype(float) * desvio_y + media_y
                medidas = metricas.calcular_metricas(conjuntos[nome].alvo, previstos, escala_mase)
                linha.update({f"{nome}_{chave}": valor for chave, valor in medidas.items()})
        if linha["validacao_RMSE"] < melhor_rmse:
            melhor_rmse, melhor_epoca, sem_melhora = linha["validacao_RMSE"], epoca, 0
            melhores_pesos = copy.deepcopy(modelo.state_dict())
        else:
            sem_melhora += 1
        sincronizar(dispositivo)
        linha["tempo_epoca_s"] = perf_counter() - inicio_epoca
        curvas.append(linha)
        # O CSV incremental preserva o histórico se uma execução for interrompida.
        pd.DataFrame([linha]).to_csv(pasta / "historico_epocas.csv", mode="a", header=(epoca == 1), index=False)
        if epoca == 1 or epoca % 10 == 0:
            registrar(f"Época {epoca}, RMSE treino={linha['treino_RMSE']:.6f}, validação={linha['validacao_RMSE']:.6f}", pasta)
        if sem_melhora >= config["paciencia"]:
            break
    sincronizar(dispositivo)
    tempo_ajuste = perf_counter() - inicio_ajuste
    modelo.load_state_dict(melhores_pesos)
    modelo.eval()
    resultados, previsoes = [], []
    for nome, (x, y) in tensores.items():
        sincronizar(dispositivo)
        inicio_previsao = perf_counter()
        with torch.inference_mode():
            previstos = modelo(x).cpu().numpy().astype(float) * desvio_y + media_y
        sincronizar(dispositivo)
        tempo_previsao = perf_counter() - inicio_previsao
        medidas = metricas.calcular_metricas(conjuntos[nome].alvo, previstos, escala_mase)
        resultados.append({
            "conjunto": nome, "observacoes": len(x), **medidas,
            "primeira_origem": conjuntos[nome].data.min().isoformat(),
            "ultima_origem": conjuntos[nome].data.max().isoformat(),
            "tempo_previsao_s": tempo_previsao,
        })
        saida = conjuntos[nome][["data", "data_alvo", "alvo"]].copy()
        saida["conjunto"], saida["previsao"] = nome, previstos
        previsoes.append(saida)

    normalizacao = {"media_x": media_x, "desvio_x": desvio_x, "media_y": media_y, "desvio_y": desvio_y}
    # O checkpoint inclui tudo de que uma avaliação posterior precisa.
    torch.save({"pesos": {k: v.cpu() for k, v in melhores_pesos.items()}, "config": config,
                "horizonte": h, "melhor_epoca": melhor_epoca, "normalizacao": normalizacao,
                "escala_mase_treino": escala_mase}, pasta / "pesos_melhor.pt")
    pd.concat(previsoes, ignore_index=True).to_csv(pasta / "previsoes.csv", index=False)
    pd.DataFrame(resultados).to_csv(pasta / "metricas.csv", index=False)
    figura, eixo = plt.subplots(figsize=(9, 4))
    for nome in ["treino", "validacao"]:
        eixo.plot([c["epoca"] for c in curvas], [c[f"{nome}_RMSE"] for c in curvas], label=nome)
    eixo.axvline(melhor_epoca, color="gray", linestyle="--", label="Melhor época")
    eixo.set(xlabel="Época", ylabel="RMSE do log-retorno", title=f"{config['experimento']} — horizonte de {h} pregões")
    eixo.legend()
    figura.tight_layout()
    figura.savefig(pasta / "curva_aprendizagem.png", dpi=150)
    plt.close(figura)
    sincronizar(dispositivo)
    registro = {
        "status": "concluido", "modelo": config["experimento"], "janela_comum_selecao": janela_comum, "horizonte": h, "config": config,
        "inicio": inicio.isoformat(), "fim": datetime.now().astimezone().isoformat(),
        "melhor_epoca": melhor_epoca, "epocas_executadas": len(curvas),
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "campo_receptivo": 1 + (config["kernel"] - 1) * sum(config["dilatacoes"]),
        "normalizacao": normalizacao, "escala_mase_treino": escala_mase,
        "origens_excluidas": excluidas, "tempo_preparacao_s": tempo_preparacao,
        "tempo_ajuste_s": tempo_ajuste, "tempo_previsao_s": sum(r["tempo_previsao_s"] for r in resultados),
        "tempo_total_s": perf_counter() - inicio_total, "tempo_cpu_s": process_time() - inicio_cpu,
        "escopo_tempos": "Ajuste inclui avaliação por época e CSV incremental. Total inclui leitura, preparação, checkpoint e gráficos até o resumo final.",
        "ambiente": {"sistema": platform.platform(), "cpu": platform.processor(),
                     "dispositivo": str(dispositivo), "gpu": torch.cuda.get_device_name(dispositivo) if dispositivo.type == "cuda" else None,
                     "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "torch": str(torch.__version__), "cuda": torch.version.cuda,
                     "memoria_gpu_maxima_bytes": torch.cuda.max_memory_allocated(dispositivo) if dispositivo.type == "cuda" else None},
        "resultados": resultados,
    }
    salvar_json(pasta / "resumo.json", registro)
    registrar(f"Concluído. Melhor época={melhor_epoca}, RMSE validação={melhor_rmse:.6f}", pasta)
    print(pd.DataFrame(resultados).to_string(index=False))
    validacao = resultados[1]
    return {"horizonte": h, "experimento": config["experimento"], "semente": config["semente"],
            **validacao, "parametros": registro["parametros"], "melhor_epoca": melhor_epoca,
            "tempo_ajuste_s": tempo_ajuste, "tempo_total_s": registro["tempo_total_s"],
            "pesos": str((pasta / "pesos_melhor.pt").relative_to(RAIZ))}


def configuracoes():
    # Ajustamos apenas a janela mínima necessária para acomodar cada campo receptivo.
    if EXPERIMENTOS.get("minima") != {}:
        raise ValueError("A referência minima deve usar a configuração base.")
    configs = []
    for nome, mudancas in EXPERIMENTOS.items():
        if mudancas.keys() - CONFIG.keys():
            raise ValueError(f"Opção desconhecida em {nome}.")
        c = {**CONFIG, **mudancas, "experimento": nome}
        if len(c["dilatacoes"]) != c["profundidade"] or not c["dilatacoes"]:
            raise ValueError(f"Revise a profundidade e as dilatações de {nome}.")
        if min(c["dilatacoes"] + [c[k] for k in ("kernel","janela","filtros","batch_size","epocas","paciencia")]) < 1:
            raise ValueError(f"Dimensão ou limite inválido em {nome}.")
        if (c["padding"] not in ("causal", "valid") or c["leitura"] not in ("ultima", "media")
                or c["inicializacao"] not in ("xavier", "he") or c["otimizador"] not in ("adam", "adamw", "sgd")):
            raise ValueError(f"Opção inválida em {nome}.")
        campo = 1 + (c["kernel"] - 1) * sum(c["dilatacoes"])
        c["janela"] = max(c["janela"], campo)
        configs.append(c)
    return configs


def comparar(linhas, pasta):
    # Comparamos cada resultado com a CNN mínima da mesma semente e horizonte.
    tabela = pd.DataFrame(linhas)
    chaves = ["horizonte", "semente"]
    medidas = ["MAE", "RMSE", "MASE"]
    referencia = tabela.loc[tabela.experimento == "minima", chaves + medidas]
    pares = tabela.merge(referencia, on=chaves, suffixes=("", "_minima"), validate="many_to_one")
    for medida in medidas:
        pares[f"delta_{medida}"] = pares[medida] - pares[f"{medida}_minima"]
    pares["venceu_minima"] = pares.delta_RMSE < 0
    pares.to_csv(pasta / "comparacao_pareada.csv", index=False)
    resumo = pares.groupby(["horizonte", "experimento"], as_index=False).agg(
        sementes=("semente", "size"), MAE=("MAE", "mean"), RMSE=("RMSE", "mean"),
        MASE=("MASE", "mean"), delta_MAE=("delta_MAE", "mean"),
        delta_RMSE=("delta_RMSE", "mean"), delta_MASE=("delta_MASE", "mean"),
        desvio_delta_RMSE=("delta_RMSE", "std"), vitorias_RMSE=("venceu_minima", "sum"),
        parametros=("parametros", "first"), tempo_ajuste_medio_s=("tempo_ajuste_s", "mean"))
    resumo = resumo.sort_values(["horizonte", "RMSE", "parametros", "experimento"])
    resumo.to_csv(pasta / "ranking_validacao.csv", index=False)
    return resumo


def main():
    configs = configuracoes()
    horizontes = CONFIG["horizontes"]
    if not horizontes or len(set(horizontes)) != len(horizontes) or set(horizontes) - {1, 5}:
        raise ValueError("Escolha horizontes distintos entre 1 e 5.")
    if not SEMENTES or len(set(SEMENTES)) != len(SEMENTES):
        raise ValueError("Escolha pelo menos uma semente, sem repetições.")
    escolhido = CONFIG["dispositivo"]
    if escolhido == "auto":
        escolhido = "cuda" if torch.cuda.is_available() else "cpu"
    dispositivo = torch.device(escolhido)
    if dispositivo.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível neste interpretador.")
    torch.set_num_threads(CONFIG["threads_cpu"])
    janela_comum = max(c["janela"] for c in configs)
    pasta = PASTA_RESULTADOS / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True, exist_ok=False)
    salvar_json(pasta / "plano.json", {"configuracoes": configs, "sementes": SEMENTES,
                                       "janela_comum_selecao": janela_comum})
    linhas = []
    for h in horizontes:
        for c in configs:
            for semente in SEMENTES:
                config = {**c, "semente": semente}
                destino = pasta / f"h{h}" / c["experimento"] / f"semente_{semente}"
                destino.mkdir(parents=True)
                try:
                    linha = treinar(h, destino, dispositivo, config, janela_comum)
                except Exception as erro:
                    salvar_json(destino / "falha.json", {"config": config, "erro": str(erro),
                                                       "fim": datetime.now().astimezone().isoformat()})
                    raise
                linhas.append(linha)
                pd.DataFrame([linha]).to_csv(pasta / "runs.csv", mode="a", header=len(linhas)==1, index=False)
    ranking = comparar(linhas, pasta)
    marcadores = []
    for h in horizontes:
        melhor = ranking.loc[ranking.horizonte == h].iloc[0].experimento
        # Preservamos todos os pesos da configuração selecionada por RMSE médio.
        pesos = [r for r in linhas if r["horizonte"] == h and r["experimento"] == melhor]
        marcador = {"pasta": str(pasta.relative_to(RAIZ)), "horizonte": h, "experimento": melhor,
                    "pesos_por_semente": {str(r["semente"]): r["pesos"] for r in pesos}}
        salvar_json(PASTA_RESULTADOS / f"ultimo_run_{h}d.json", marcador)
        marcadores.append(marcador)
    salvar_json(PASTA_RESULTADOS / "ultimo_run.json", {"pasta": str(pasta.relative_to(RAIZ)), "selecionados": marcadores})
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
