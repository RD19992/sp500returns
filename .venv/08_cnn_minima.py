"""CNN mínima para retornos de 1 e 5 pregões, com treino e validação.

Coloque ao lado de 04_metricas.py. Execute com python 08_cnn_minima.py.
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
RAIZ = Path(__file__).resolve().parent
PASTA_CONJUNTOS = RAIZ / "data" / "conjuntos"
ARQUIVO_HISTORICO = RAIZ / "data" / "trabalho" / "sp500_log_retornos.csv"
PASTA_RESULTADOS = RAIZ / "outputs" / "cnn_minima"

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
        return self.saida(entradas[:, :, -1]).squeeze(-1)


def treinar(h, pasta, dispositivo):
    inicio = datetime.now().astimezone()
    inicio_total, inicio_cpu = perf_counter(), process_time()
    registrar(f"Início h={h}, dispositivo={dispositivo}", pasta)
    random.seed(CONFIG["semente"])
    np.random.seed(CONFIG["semente"])
    torch.manual_seed(CONFIG["semente"])
    if dispositivo.type == "cuda":
        torch.cuda.manual_seed_all(CONFIG["semente"])
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
    entradas = pd.concat([retornos.shift(j) for j in range(CONFIG["janela"] - 1, -1, -1)], axis=1)
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
        x = ((valores[disponiveis] - media_x) / desvio_x).astype("float32")
        y = ((conjuntos[nome].alvo.to_numpy() - media_y) / desvio_y).astype("float32")
        # Toda a base pequena é transferida uma vez. Os lotes são recortados na GPU.
        tensores[nome] = (torch.from_numpy(x[:, None, :]).to(dispositivo), torch.from_numpy(y).to(dispositivo))
    modelo = CNN(CONFIG).to(dispositivo)
    otimizadores = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    argumentos = {"lr": CONFIG["taxa_aprendizagem"], "weight_decay": CONFIG["weight_decay"]}
    if CONFIG["otimizador"] == "sgd":
        argumentos["momentum"] = 0.9
    otimizador = otimizadores[CONFIG["otimizador"]](modelo.parameters(), **argumentos)
    perda = nn.MSELoss()
    sincronizar(dispositivo)
    tempo_preparacao = perf_counter() - inicio_total
    registrar(f"Parâmetros={sum(p.numel() for p in modelo.parameters())}, origens excluídas={excluidas}", pasta)

    curvas, melhor_rmse, melhor_epoca, sem_melhora = [], float("inf"), 0, 0
    inicio_ajuste = perf_counter()
    for epoca in range(1, CONFIG["epocas"] + 1):
        sincronizar(dispositivo)
        inicio_epoca = perf_counter()
        modelo.train()
        x_treino, y_treino = tensores["treino"]
        ordem = torch.randperm(len(y_treino), device=dispositivo)
        for lote in ordem.split(CONFIG["batch_size"]):
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
        pd.DataFrame(curvas).to_csv(pasta / "historico_epocas.csv", index=False)
        if epoca == 1 or epoca % 10 == 0:
            registrar(f"Época {epoca}, RMSE treino={linha['treino_RMSE']:.6f}, validação={linha['validacao_RMSE']:.6f}", pasta)
        if sem_melhora >= CONFIG["paciencia"]:
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
    torch.save({"pesos": {k: v.cpu() for k, v in melhores_pesos.items()}, "config": CONFIG,
                "horizonte": h, "melhor_epoca": melhor_epoca, "normalizacao": normalizacao,
                "escala_mase_treino": escala_mase}, pasta / "pesos_melhor.pt")
    pd.concat(previsoes, ignore_index=True).to_csv(pasta / "previsoes.csv", index=False)
    pd.DataFrame(resultados).to_csv(pasta / "metricas.csv", index=False)
    figura, eixo = plt.subplots(figsize=(9, 4))
    for nome in ["treino", "validacao"]:
        eixo.plot([c["epoca"] for c in curvas], [c[f"{nome}_RMSE"] for c in curvas], label=nome)
    eixo.axvline(melhor_epoca, color="gray", linestyle="--", label="Melhor época")
    eixo.set(xlabel="Época", ylabel="RMSE do log-retorno", title=f"CNN mínima — horizonte de {h} pregões")
    eixo.legend()
    figura.tight_layout()
    figura.savefig(pasta / "curva_aprendizagem.png", dpi=150)
    plt.close(figura)
    sincronizar(dispositivo)
    registro = {
        "status": "concluido", "modelo": "cnn_minima", "horizonte": h, "config": CONFIG,
        "inicio": inicio.isoformat(), "fim": datetime.now().astimezone().isoformat(),
        "melhor_epoca": melhor_epoca, "epocas_executadas": len(curvas),
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "campo_receptivo": 1 + (CONFIG["kernel"] - 1) * sum(CONFIG["dilatacoes"]),
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
    return {"horizonte": h, "pasta": str(pasta.relative_to(RAIZ)), "pesos": str((pasta / 'pesos_melhor.pt').relative_to(RAIZ))}


def main():
    if CONFIG["profundidade"] != len(CONFIG["dilatacoes"]) or CONFIG["profundidade"] < 1:
        raise ValueError("Profundidade deve coincidir com a quantidade de dilatações.")
    if CONFIG["padding"] not in ["causal", "valid"] or CONFIG["inicializacao"] not in ["he", "xavier"]:
        raise ValueError("Padding ou inicialização inválidos.")
    if CONFIG["otimizador"] not in ["adam", "adamw", "sgd"]:
        raise ValueError("Otimizador inválido.")
    if min(CONFIG[k] for k in ["janela", "filtros", "kernel", "batch_size", "epocas", "paciencia", "threads_cpu"]) < 1 or min(CONFIG["dilatacoes"]) < 1:
        raise ValueError("Dimensões, épocas e paciência devem ser positivas.")
    campo = 1 + (CONFIG["kernel"] - 1) * sum(CONFIG["dilatacoes"])
    if CONFIG["janela"] < campo:
        raise ValueError(f"Use janela >= campo receptivo ({campo}) para evitar contexto artificial.")
    if CONFIG["dispositivo"] not in ["auto", "cuda", "cpu"]:
        raise ValueError("Dispositivo inválido.")
    escolhido = "cuda" if CONFIG["dispositivo"] == "auto" and torch.cuda.is_available() else CONFIG["dispositivo"]
    dispositivo = torch.device("cpu" if escolhido == "auto" else escolhido)
    if dispositivo.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível. Confira a instalação do PyTorch com suporte à sua GPU.")
    torch.set_num_threads(CONFIG["threads_cpu"])
    pasta = PASTA_RESULTADOS / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True, exist_ok=False)
    salvar_json(pasta / "config.json", CONFIG)
    runs = []
    for h in CONFIG["horizontes"]:
        if h not in [1, 5]:
            raise ValueError("Os horizontes do projeto são 1 e 5.")
        destino = pasta / f"h{h}"
        destino.mkdir()
        try:
            run = treinar(h, destino, dispositivo)
        except Exception as erro:
            salvar_json(destino / "falha.json", {"status": "falhou", "erro": str(erro), "config": CONFIG})
            raise
        runs.append(run)
        salvar_json(PASTA_RESULTADOS / f"ultimo_run_{h}d.json", run)
    salvar_json(PASTA_RESULTADOS / "ultimo_run.json", {"pasta": str(pasta.relative_to(RAIZ)), "runs": runs})
    print(f"\nÚltimo run atualizado em {PASTA_RESULTADOS / 'ultimo_run.json'}")


if __name__ == "__main__":
    main()
