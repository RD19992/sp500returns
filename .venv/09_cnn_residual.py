"""CNN residual dilatada para retornos de 1 e 5 pregões, com treino e validação.

Coloque ao lado de 04_metricas.py. Execute com python 09_cnn_residual.py.
Instale numpy, pandas, matplotlib e a versão do PyTorch adequada à sua GPU.

Referências que orientam esta adaptação de TCN:
Bai, Kolter e Koltun (2018): https://arxiv.org/abs/1803.01271
He et al. (2016), resíduos: https://arxiv.org/abs/1512.03385
He et al. (2015), inicialização: https://arxiv.org/abs/1502.01852
Loshchilov e Hutter (2019), AdamW: https://arxiv.org/abs/1711.05101
As dimensões e os hiperparâmetros abaixo são escolhas experimentais do projeto.
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
    "janela": 125,
    "blocos": 5,                          # Duas convoluções por bloco: profundidade 10.
    "filtros": [32, 32, 32, 32, 32],
    "kernels": [3, 3, 3, 3, 3],
    "dilatacoes": [1, 2, 4, 8, 16],
    "padding": "causal",                 # Opções: causal ou valid.
    "residuos": True,
    "leitura": "ultima",                 # Opções: ultima ou media (pooling global).
    "dropout": 0.10,
    "inicializacao": "he",               # Opções: he ou xavier.
    "relu_antes_da_soma": True,         # False permite correções residuais negativas.
    "desvio_inicial_saida": 0.001,       # Inicialização da cabeça para o ajuste da CNN.
    "otimizador": "adamw",               # Opções: adam, adamw ou sgd.
    "taxa_aprendizagem": 0.0003,
    "weight_decay": 0.001,
    "batch_size": 128,
    "epocas": 1000,
    "paciencia": 100,
    "paciencia_lr": 25,
    "fator_lr": 0.5,
    "lr_minima": 0.000001,
    "clip_gradiente": 1.0,
    "tempo_maximo_por_horizonte_s": 600,  # Verificado ao final de cada época.
    "semente": 42,
    "dispositivo": "cuda",               # Opções: auto, cuda ou cpu.
    "threads_cpu": 4,
}


def campo_receptivo(config):
    # Cada bloco contém duas convoluções de mesmo kernel e dilatação.
    return 1 + 2 * sum((k - 1) * d for k, d in zip(config["kernels"], config["dilatacoes"]))

RAIZ = Path(__file__).resolve().parent
PASTA_CONJUNTOS = RAIZ / "data" / "conjuntos"
ARQUIVO_HISTORICO = RAIZ / "data" / "trabalho" / "sp500_log_retornos.csv"
PASTA_RESULTADOS = RAIZ / "outputs" / "cnn_residual"

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


class BlocoResidual(nn.Module):
    """Duas convoluções dilatadas e uma conexão que preserva a entrada."""

    def __init__(self, canais_entrada, canais_saida, kernel, dilatacao, config):
        super().__init__()
        self.padding = config["padding"]
        self.residuos = config["residuos"]
        # Checkpoints da primeira versão usavam ReLU nas duas convoluções.
        self.relu_antes_da_soma = config.get("relu_antes_da_soma", True)
        self.esquerda = (kernel - 1) * dilatacao
        self.conv1 = nn.Conv1d(canais_entrada, canais_saida, kernel, dilation=dilatacao)
        self.conv2 = nn.Conv1d(canais_saida, canais_saida, kernel, dilation=dilatacao)
        self.dropout = nn.Dropout1d(config["dropout"])
        # A projeção 1x1 ajusta os canais sem misturar posições temporais.
        self.atalho = nn.Identity()
        if self.residuos and canais_entrada != canais_saida:
            self.atalho = nn.Conv1d(canais_entrada, canais_saida, 1)

    def forward(self, entradas):
        saida = entradas
        for camada in (self.conv1, self.conv2):
            if self.padding == "causal":
                saida = F.pad(saida, (self.esquerda, 0))
            saida = camada(saida)
            if camada is self.conv1 or self.relu_antes_da_soma:
                saida = F.relu(saida)
            saida = self.dropout(saida)
        if self.residuos:
            atalho = self.atalho(entradas)
            # No modo valid, alinhamos o atalho às mesmas posições finais.
            atalho = atalho[:, :, -saida.shape[-1]:]
            saida = saida + atalho
        return F.relu(saida)


class CNN(nn.Module):
    """TCN adaptada com blocos residuais e saída escalar por janela."""

    def __init__(self, config):
        super().__init__()
        self.leitura = config["leitura"]
        blocos = []
        canais = 1
        for filtros, kernel, dilatacao in zip(config["filtros"], config["kernels"], config["dilatacoes"]):
            blocos.append(BlocoResidual(canais, filtros, kernel, dilatacao, config))
            canais = filtros
        self.blocos = nn.Sequential(*blocos)
        self.saida = nn.Linear(canais, 1)
        for camada in self.modules():
            if isinstance(camada, nn.Conv1d):
                if config["inicializacao"] == "he":
                    nn.init.kaiming_normal_(camada.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(camada.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(camada.bias)
        nn.init.normal_(self.saida.weight, std=config["desvio_inicial_saida"])
        nn.init.zeros_(self.saida.bias)

    def forward(self, entradas):
        saida = self.blocos(entradas)
        representacao = saida[:, :, -1] if self.leitura == "ultima" else saida.mean(dim=-1)
        return self.saida(representacao).squeeze(-1)


@torch.inference_mode()
def prever(modelo, x, media_y, desvio_y):
    # Limitamos as ativações temporárias ao tamanho de um lote também na avaliação.
    modelo.eval()
    lotes = [modelo(lote) for lote in x.split(CONFIG["batch_size"])]
    return torch.cat(lotes).cpu().numpy().astype(float) * desvio_y + media_y


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
    agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="min", factor=CONFIG["fator_lr"],
        patience=CONFIG["paciencia_lr"], min_lr=CONFIG["lr_minima"],
    )
    perda = nn.MSELoss()
    sincronizar(dispositivo)
    tempo_preparacao = perf_counter() - inicio_total
    registrar(f"Parâmetros={sum(p.numel() for p in modelo.parameters())}, origens excluídas={excluidas}", pasta)

    # A época zero representa a média histórica, candidata explícita à seleção.
    curvas, melhor_rmse, melhor_epoca, sem_melhora = [], float("inf"), 0, 0
    motivo_parada = "limite_epocas"
    inicio_ajuste = perf_counter()
    for epoca in range(CONFIG["epocas"] + 1):
        inicio_epoca = perf_counter()
        if epoca > 0:
            modelo.train()
            x_treino, y_treino = tensores["treino"]
            ordem = torch.randperm(len(y_treino), device=dispositivo)
            for lote in ordem.split(CONFIG["batch_size"]):
                otimizador.zero_grad(set_to_none=True)
                erro = perda(modelo(x_treino[lote]), y_treino[lote])
                erro.backward()
                nn.utils.clip_grad_norm_(modelo.parameters(), CONFIG["clip_gradiente"], error_if_nonfinite=True)
                otimizador.step()
        linha = {"epoca": epoca, "taxa_aprendizagem": otimizador.param_groups[0]["lr"]}
        for nome, (x, _) in tensores.items():
            previstos = np.full(len(x), media_y) if epoca == 0 else prever(modelo, x, media_y, desvio_y)
            medidas = metricas.calcular_metricas(conjuntos[nome].alvo, previstos, escala_mase)
            linha.update({f"{nome}_{chave}": valor for chave, valor in medidas.items()})
        if linha["validacao_RMSE"] < melhor_rmse:
            melhor_rmse, melhor_epoca, sem_melhora = linha["validacao_RMSE"], epoca, 0
            melhores_pesos = copy.deepcopy(modelo.state_dict())
            if epoca == 0:
                # Zero na escala padronizada reproduz a média; só alteramos a cópia.
                melhores_pesos["saida.weight"].zero_()
                melhores_pesos["saida.bias"].zero_()
        else:
            sem_melhora += 1
        if epoca > 0:
            agendador.step(linha["validacao_RMSE"])
        sincronizar(dispositivo)
        linha["tempo_epoca_s"] = perf_counter() - inicio_epoca
        curvas.append(linha)
        # O CSV incremental preserva o histórico se uma execução for interrompida.
        pd.DataFrame([linha]).to_csv(pasta / "historico_epocas.csv", mode="a", header=(epoca == 0), index=False)
        if epoca == 1 or epoca % 10 == 0:
            registrar(f"Época {epoca}, RMSE treino={linha['treino_RMSE']:.6f}, validação={linha['validacao_RMSE']:.6f}", pasta)
        if sem_melhora >= CONFIG["paciencia"]:
            motivo_parada = "parada_antecipada"
            break
        if perf_counter() - inicio_ajuste >= CONFIG["tempo_maximo_por_horizonte_s"]:
            motivo_parada = "orcamento_tempo"
            break
    sincronizar(dispositivo)
    tempo_ajuste = perf_counter() - inicio_ajuste
    modelo.load_state_dict(melhores_pesos)
    modelo.eval()
    resultados, previsoes = [], []
    for nome, (x, y) in tensores.items():
        sincronizar(dispositivo)
        inicio_previsao = perf_counter()
        previstos = prever(modelo, x, media_y, desvio_y)
        sincronizar(dispositivo)
        tempo_previsao = perf_counter() - inicio_previsao
        medidas = metricas.calcular_metricas(conjuntos[nome].alvo, previstos, escala_mase)
        # Usamos a mesma média do script 05 e exatamente as origens desta CNN.
        previstos_media = np.full(len(x), media_y)
        rmse_media = metricas.rmse(conjuntos[nome].alvo, previstos_media)
        resultados.append({
            "rmse_media_historica": rmse_media,
            "ganho_rmse_sobre_media": rmse_media - medidas["RMSE"],
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
    eixo.set(xlabel="Época", ylabel="RMSE do log-retorno", title=f"CNN residual dilatada — horizonte de {h} pregões")
    eixo.legend()
    figura.tight_layout()
    figura.savefig(pasta / "curva_aprendizagem.png", dpi=150)
    plt.close(figura)
    sincronizar(dispositivo)
    registro = {
        "status": "concluido", "modelo": "cnn_residual", "horizonte": h, "config": CONFIG,
        "inicio": inicio.isoformat(), "fim": datetime.now().astimezone().isoformat(),
        "melhor_epoca": melhor_epoca, "epocas_executadas": curvas[-1]["epoca"],
        "previsor_selecionado": "media_historica" if melhor_epoca == 0 else "cnn",
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "campo_receptivo": campo_receptivo(CONFIG),
        "profundidade_convolucional": 2 * CONFIG["blocos"], "motivo_parada": motivo_parada,
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
    if CONFIG["blocos"] < 1 or any(len(CONFIG[k]) != CONFIG["blocos"] for k in ["filtros", "kernels", "dilatacoes"]):
        raise ValueError("Filtros, kernels e dilatações precisam de um valor por bloco.")
    if min(CONFIG["filtros"] + CONFIG["kernels"] + CONFIG["dilatacoes"]) < 1:
        raise ValueError("Filtros, kernels e dilatações devem ser positivos.")
    if CONFIG["padding"] not in ["causal", "valid"] or CONFIG["leitura"] not in ["ultima", "media"]:
        raise ValueError("Padding ou leitura inválidos.")
    if CONFIG["inicializacao"] not in ["he", "xavier"] or CONFIG["otimizador"] not in ["adam", "adamw", "sgd"]:
        raise ValueError("Inicialização ou otimizador inválidos.")
    if not 0 <= CONFIG["dropout"] < 1 or not 0 < CONFIG["fator_lr"] < 1:
        raise ValueError("Dropout ou fator de redução da taxa inválidos.")
    if min(CONFIG[k] for k in ["janela", "batch_size", "epocas", "paciencia", "threads_cpu", "tempo_maximo_por_horizonte_s", "clip_gradiente", "taxa_aprendizagem",  "lr_minima"]) <= 0:
        raise ValueError("Dimensões, taxas e orçamento de tempo devem ser positivos.")
    if CONFIG["desvio_inicial_saida"] < 0:
        raise ValueError("O desvio inicial da saída deve ser não negativo.")
    campo = campo_receptivo(CONFIG)
    if CONFIG["janela"] < campo:
        raise ValueError(f"Use janela >= campo receptivo ({campo}).")
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
