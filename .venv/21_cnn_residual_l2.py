"""Grid de L2 explícita na CNN principal do 09. Execute este arquivo e depois o 22."""
import copy
import importlib
import json
import platform
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

original = importlib.import_module("09_cnn_residual")
CNN = original.CNN
campo_receptivo = original.campo_receptivo
sincronizar = original.sincronizar
registrar = original.registrar
salvar_json = original.salvar_json
metricas = original.metricas
RAIZ = original.RAIZ
PASTA_CONJUNTOS = original.PASTA_CONJUNTOS
ARQUIVO_HISTORICO = original.ARQUIVO_HISTORICO
MODELO = "cnn_residual_l2"
PASTA_RESULTADOS = RAIZ / "outputs" / MODELO

# MSE média dos alvos padronizados + lambda_l2 * soma(W²)/(2P).
# P conta cada peso convolucional ou linear uma vez, incluindo projeções.
# Não se penalizam os vieses. O controle usa o weight_decay original do 09.
CONFIG = {**original.CONFIG, "sementes": [42, 43, 44],
          "lambdas_l2": [0.0, 0.3, 3.0, 30.0], "dispositivo": "auto"}


@torch.inference_mode()
def prever(modelo, x, media_y, desvio_y, lote):
    modelo.eval()
    saidas = [modelo(parte) for parte in x.split(lote)]
    return torch.cat(saidas).cpu().numpy().astype(float) * desvio_y + media_y


def treinar(h, pasta, dispositivo, config):
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
    entradas = pd.concat([retornos.shift(j) for j in range(config["janela"] - 1, -1, -1)], axis=1)
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
    modelo = CNN(config).to(dispositivo)
    otimizadores = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    argumentos = {"lr": config["taxa_aprendizagem"], "weight_decay": config["weight_decay"]}
    if config["otimizador"] == "sgd":
        argumentos["momentum"] = 0.9
    otimizador = otimizadores[config["otimizador"]](modelo.parameters(), **argumentos)
    agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="min", factor=config["fator_lr"],
        patience=config["paciencia_lr"], min_lr=config["lr_minima"],
    )
    perda = nn.MSELoss()
    pesos = [m.weight for m in modelo.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]
    numero_pesos = sum(w.numel() for w in pesos)
    coeficiente = config["lambda_l2"] / (2 * numero_pesos)
    sincronizar(dispositivo)
    tempo_preparacao = perf_counter() - inicio_total
    registrar(f"Parâmetros={sum(p.numel() for p in modelo.parameters())}, origens excluídas={excluidas}", pasta)

    # A época zero representa a média histórica, candidata explícita à seleção.
    curvas, melhor_rmse, melhor_epoca, sem_melhora = [], float("inf"), 0, 0
    motivo_parada = "limite_epocas"
    inicio_ajuste = perf_counter()
    for epoca in range(config["epocas"] + 1):
        inicio_epoca = perf_counter()
        if epoca > 0:
            modelo.train()
            x_treino, y_treino = tensores["treino"]
            ordem = torch.randperm(len(y_treino), device=dispositivo)
            for lote in ordem.split(config["batch_size"]):
                otimizador.zero_grad(set_to_none=True)
                erro = perda(modelo(x_treino[lote]), y_treino[lote])
                if coeficiente:
                    soma_quadrados = torch.stack([w.square().sum() for w in pesos]).sum()
                    erro = erro + coeficiente * soma_quadrados
                erro.backward()
                nn.utils.clip_grad_norm_(modelo.parameters(), config["clip_gradiente"], error_if_nonfinite=True)
                otimizador.step()
        linha = {"epoca": epoca, "taxa_aprendizagem": otimizador.param_groups[0]["lr"]}
        for nome, (x, _) in tensores.items():
            previstos = np.full(len(x), media_y) if epoca == 0 else prever(modelo, x, media_y, desvio_y, config["batch_size"])
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
        if sem_melhora >= config["paciencia"]:
            motivo_parada = "parada_antecipada"
            break
        if perf_counter() - inicio_ajuste >= config["tempo_maximo_por_horizonte_s"]:
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
        previstos = prever(modelo, x, media_y, desvio_y, config["batch_size"])
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
    torch.save({"pesos": {k: v.cpu() for k, v in melhores_pesos.items()}, "config": config,
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
        "status": "concluido", "modelo": MODELO, "pesos_regularizados": numero_pesos, "horizonte": h, "config": config,
        "inicio": inicio.isoformat(), "fim": datetime.now().astimezone().isoformat(),
        "melhor_epoca": melhor_epoca, "epocas_executadas": curvas[-1]["epoca"],
        "previsor_selecionado": "media_historica" if melhor_epoca == 0 else "cnn",
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "campo_receptivo": campo_receptivo(config),
        "profundidade_convolucional": 2 * config["blocos"], "motivo_parada": motivo_parada,
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


def main():
    grade = CONFIG["lambdas_l2"]
    sementes = CONFIG["sementes"]
    if not sementes or len(set(sementes)) != len(sementes):
        raise ValueError("Informe sementes distintas.")
    if not grade or 0 not in grade or len(set(grade)) != len(grade) or not all(
            np.isfinite(v) and v >= 0 for v in grade):
        raise ValueError("A grade deve incluir zero e valores finitos, positivos e distintos.")
    alvo = CONFIG["dispositivo"]
    if alvo == "auto":
        alvo = "cuda" if torch.cuda.is_available() else "cpu"
    dispositivo = torch.device(alvo)
    torch.set_num_threads(CONFIG["threads_cpu"])
    pasta = PASTA_RESULTADOS / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True, exist_ok=False)
    salvar_json(pasta / "config.json", CONFIG)
    candidatos = [(f"l2_{i:02d}", v, 0.0) for i, v in enumerate(grade)]
    candidatos.append(("controle_09", 0.0, CONFIG["weight_decay"]))
    linhas, escolhidos, runs = [], {}, []
    for h in CONFIG["horizontes"]:
        for nome, valor, decay in candidatos:
            for s in sementes:
                destino = pasta / f"h{h}" / nome / f"semente_{s}"
                destino.mkdir(parents=True)
                config = {**CONFIG, "semente": s, "lambda_l2": valor, "weight_decay": decay}
                print(f"h={h}, {nome}, lambda={valor:g}, decay={decay:g}, semente={s}", flush=True)
                treinar(h, destino, dispositivo, config)
                resumo = json.loads((destino / "resumo.json").read_text(encoding="utf-8"))
                val = next(r for r in resumo["resultados"] if r["conjunto"] == "validacao")
                linhas.append(dict(horizonte=h, candidato=nome, lambda_l2=valor, weight_decay=decay,
                                   semente=s, RMSE=val["RMSE"], melhor_epoca=resumo["melhor_epoca"],
                                   tempo_ajuste_s=resumo["tempo_ajuste_s"], motivo_parada=resumo["motivo_parada"]))
                pd.DataFrame(linhas).to_csv(pasta / "grade_por_ajuste.csv", index=False)
        dados = pd.DataFrame(linhas)
        ranking = dados.groupby(["horizonte", "candidato", "lambda_l2", "weight_decay"], as_index=False).agg(
            RMSE_medio=("RMSE", "mean"), RMSE_dp=("RMSE", "std"), ajustes=("RMSE", "size"))
        ranking.to_csv(pasta / "grade_validacao.csv", index=False)
        disputa = ranking[(ranking.horizonte == h) & (ranking.candidato != "controle_09")]
        melhor = disputa.sort_values(["RMSE_medio", "lambda_l2"]).iloc[0]
        nome = str(melhor.candidato)
        escolhidos[str(h)] = dict(candidato=nome, lambda_l2=float(melhor.lambda_l2),
                                  RMSE_medio=float(melhor.RMSE_medio))
        for s in sementes:
            runs.append(dict(horizonte=h, semente=s, candidato=nome))
        print(f"Selecionado por validação, h={h}: {escolhidos[str(h)]}", flush=True)
    salvar_json(pasta / "execucao.json", {"selecionados": escolhidos, "runs": runs})
    salvar_json(PASTA_RESULTADOS / "ultimo_run.json", {"pasta": str(pasta.relative_to(RAIZ))})
    print(f"Grade e checkpoints: {pasta}")


if __name__ == "__main__":
    main()
