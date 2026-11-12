"""Avalia no teste os modelos congelados dos scripts 05, 06, 08, 09 e 10.

Execute na pasta dos scripts: python 11_avaliar_modelos_teste.py
O erro assinado é o realizado menos o previsto. Retornos em escala log decimal.
"""
import importlib
import json
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# EDITE AQUI. None usa o último run completo. Para fixar, informe a pasta temporal.
RUNS = dict(baselines_simples=None, baseline_ar=None, cnn_minima=None,
            cnn_residual=None, experimentos_cnn=None)
HORIZONTES = [1, 5]
DISPOSITIVO = "auto"  # auto, cuda ou cpu
LOTE = 128
RAIZ = Path(__file__).resolve().parent
OUTPUTS = RAIZ / "outputs"
metricas = importlib.import_module("04_metricas")
baselines = ["retorno_zero", "media_historica", "ar_direto"]
scripts = {"cnn_minima": "08_cnn_minima", "cnn_residual": "09_cnn_residual",
           "experimentos_cnn": "10_experimentos_cnn"}


def ler_json(arquivo):
    return json.loads(arquivo.read_text(encoding="utf-8"))


def caminho_salvo(texto):
    # Os marcadores podem ter sido produzidos no Windows ou no Linux.
    return RAIZ / str(texto).replace(chr(92), "/")


def inventario():
    # Resolvemos a seleção feita na validação antes de abrir qualquer alvo de teste.
    modelos = []
    for familia, nome in RUNS.items():
        raiz = OUTPUTS / familia
        if familia in scripts:
            pasta = raiz / nome if nome else caminho_salvo(ler_json(raiz / "ultimo_run.json")["pasta"])
        else:
            nomes = baselines[:2] if familia == "baselines_simples" else baselines[2:]
            arquivos = [f"{m}_{h}d.json" for h in HORIZONTES for m in nomes]
            escolher = importlib.import_module("07_avaliar_baselines_teste").escolher_execucao
            pasta = escolher(raiz, nome, arquivos)
        if familia == "experimentos_cnn":
            plano = ler_json(pasta / "plano.json")
            ranking = pd.read_csv(pasta / "ranking_validacao.csv")
        for h in HORIZONTES:
            if familia == "experimentos_cnn":
                vencedor = ranking.loc[ranking.horizonte == h, "experimento"].iloc[0]
                for config in plano["configuracoes"]:
                    exp = config["experimento"]
                    for s in plano["sementes"]:
                        arquivo = pasta / f"h{h}" / exp / f"semente_{s}" / "pesos_melhor.pt"
                        modelos.append((familia, exp, h, s, exp == vencedor, arquivo))
            elif familia in scripts:
                modelos.append((familia, familia, h, None, False, pasta / f"h{h}" / "pesos_melhor.pt"))
            else:
                for m in nomes:
                    modelos.append((familia, m, h, None, False, pasta / f"{m}_{h}d.json"))
    for *_, arquivo in modelos:
        if not arquivo.is_file():
            raise FileNotFoundError(f"Modelo ausente na execução escolhida: {arquivo}")
    return modelos


def janelas(serie, datas, tamanho):
    # A coluna final corresponde ao retorno conhecido na origem da previsão.
    x = pd.concat([serie.shift(i) for i in reversed(range(tamanho))], axis=1)
    valores = x.reindex(datas).to_numpy()
    if not np.isfinite(valores).all():
        raise ValueError("Histórico insuficiente para prever em todas as datas do teste.")
    return valores


def avaliar(item, teste, historico, dispositivo):
    familia, modelo, h, semente, selecionado, arquivo = item
    inicio = datetime.now().astimezone()
    relogio, cpu = perf_counter(), process_time()
    neural = familia in scripts
    gpu = neural and dispositivo.type == "cuda"
    if gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    resumo = ler_json(arquivo.parent / "resumo.json" if neural else arquivo)
    if resumo["horizonte"] != h or resumo["modelo"] != modelo:
        raise ValueError(f"Modelo ou horizonte divergente em {arquivo}")
    fim_val = next(r["ultima_origem"] for r in resumo["resultados"] if r["conjunto"] == "validacao")
    if pd.Timestamp(fim_val) >= teste.data.min():
        raise ValueError(f"Sobreposição com a validação em {arquivo}")
    escala = resumo["escala_mase_treino"]
    if neural:
        ck = torch.load(arquivo, map_location="cpu", weights_only=True)
        cfg, norm = ck["config"], ck["normalizacao"]
        if ck["horizonte"] != h or cfg != resumo["config"] or ck["escala_mase_treino"] != escala:
            raise ValueError(f"Checkpoint e resumo incompatíveis: {arquivo}")
        if semente is not None and (cfg["semente"] != semente or cfg["experimento"] != modelo):
            raise ValueError(f"Configuração experimental divergente em {arquivo}")
        semente = cfg["semente"]
        rede = importlib.import_module(scripts[familia]).CNN(cfg)
        rede.load_state_dict(ck["pesos"])
        rede.to(dispositivo).eval()
        valores = janelas(historico, teste.data, cfg["janela"])
        x = torch.as_tensor((valores - norm["media_x"]) / norm["desvio_x"],
                            dtype=torch.float32, device=dispositivo).unsqueeze(1)
    elif modelo == "ar_direto":
        p = resumo["ordem_escolhida"]
        coef = resumo["coeficientes"]
        # O AR usa x1 = retorno atual, x2 = retorno anterior e assim por diante.
        if p:
            x = janelas(historico, teste.data, p)[:, ::-1]
            beta = np.array([coef[f"x{i}"] for i in range(1, p + 1)])
    if gpu:
        torch.cuda.synchronize()
    t = perf_counter()
    if neural:
        with torch.inference_mode():
            bruto = torch.cat([rede(lote) for lote in x.split(LOTE)]).cpu().numpy()
        previstos = bruto.astype(float) * norm["desvio_y"] + norm["media_y"]
    elif modelo == "ar_direto":
        previstos = np.full(len(teste), coef["intercepto"])
        if p:
            previstos = previstos + x @ beta
    else:
        previstos = np.full(len(teste), resumo["constante_prevista"])
    if gpu:
        torch.cuda.synchronize()
    tempo_previsao = perf_counter() - t
    erros = teste[["data", "data_alvo", "alvo"]].copy()
    erros["previsao"] = previstos
    erros["erro"] = erros.alvo - previstos
    erros["erro_absoluto"] = erros.erro.abs()
    erros["erro_quadratico"] = erros.erro ** 2
    chave = dict(familia=familia, modelo=modelo, horizonte=h, semente=semente,
                 selecionado_na_validacao_10=selecionado)
    linha = dict(**chave, **metricas.calcular_metricas(teste.alvo, previstos, escala),
                 observacoes=len(teste), vies=erros.erro.mean(), p95_erro_absoluto=erros.erro_absoluto.quantile(.95),
                 escala_mase_treino=escala, arquivo_modelo=str(arquivo.relative_to(RAIZ)),
                 inicio=inicio.isoformat(), fim=datetime.now().astimezone().isoformat(),
                 dispositivo=str(dispositivo) if neural else "cpu", tempo_previsao_s=tempo_previsao,
                 tempo_total_avaliacao_s=perf_counter()-relogio, tempo_cpu_s=process_time()-cpu,
                 pico_gpu_bytes=torch.cuda.max_memory_allocated() if gpu else None)
    return linha, erros.assign(**chave)


def main():
    itens = inventario()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if DISPOSITIVO == "auto" else DISPOSITIVO
    dispositivo = torch.device(device)
    inicio = datetime.now().astimezone()
    pasta = OUTPUTS / "avaliacao_teste" / inicio.strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True, exist_ok=False)
    historico = pd.read_csv(RAIZ / "data/trabalho/sp500_log_retornos.csv", parse_dates=["data"]).set_index("data")
    if historico.index.hasnans or not historico.index.is_unique or not historico.index.is_monotonic_increasing:
        raise ValueError("Histórico com datas inválidas.")
    linhas, detalhes = [], []
    for h in HORIZONTES:
        teste = pd.read_csv(RAIZ / f"data/conjuntos/teste_{h}d.csv", parse_dates=["data", "data_alvo"])
        if (teste.empty or teste.data.duplicated().any() or not teste.data.is_monotonic_increasing
                or not (teste.data_alvo > teste.data).all() or not np.isfinite(teste.alvo).all()):
            raise ValueError(f"Teste de {h} pregões inválido.")
        # O alvo da última previsão de validação deve ser anterior à primeira origem de teste.
        val = pd.read_csv(RAIZ / f"data/conjuntos/validacao_{h}d.csv", parse_dates=["data_alvo"])
        if val.empty or val.data_alvo.isna().any() or val.data_alvo.max() >= teste.data.min():
            raise ValueError("O corte temporal entre validação e teste é inválido.")
        for item in itens:
            if item[2] == h:
                linha, erros = avaliar(item, teste, historico.log_retorno_1d, dispositivo)
                linhas.append(linha)
                detalhes.append(erros)
                print(h, item[1], item[3], f"RMSE={linha['RMSE']:.6f}", flush=True)
        escalas = [r["escala_mase_treino"] for r in linhas if r["horizonte"] == h]
        if not np.allclose(escalas, escalas[0], rtol=1e-10, atol=0):
            raise ValueError("Escalas de MASE divergentes: confira as execuções de origem.")

    tabela = pd.DataFrame(linhas)
    erros = pd.concat(detalhes, ignore_index=True)
    # A média histórica congelada serve de referência na mesma amostra de teste.
    referencia = tabela[tabela.modelo == "media_historica"].set_index("horizonte").RMSE
    tabela["ganho_RMSE_sobre_media"] = tabela.horizonte.map(referencia) - tabela.RMSE
    tabela.to_csv(pasta / "metricas_teste.csv", index=False)
    erros.to_csv(pasta / "erros_por_data.csv", index=False)
    chaves = ["familia", "modelo", "horizonte", "semente"]
    maiores = erros.sort_values("erro_absoluto", ascending=False).groupby(chaves, dropna=False).head(10)
    maiores.to_csv(pasta / "maiores_erros.csv", index=False)
    # O resumo por arquitetura agrega métricas das sementes, sem combinar suas previsões.
    exp = tabela[tabela.familia == "experimentos_cnn"]
    exp.groupby(["horizonte", "modelo", "selecionado_na_validacao_10"]).agg(
        sementes=("RMSE", "size"), MAE_medio=("MAE", "mean"), RMSE_medio=("RMSE", "mean"),
        RMSE_desvio=("RMSE", "std"), MASE_medio=("MASE", "mean")).to_csv(pasta / "experimentos_por_configuracao.csv")
    # Incluímos no gráfico os modelos principais e cada semente da configuração pré-selecionada.
    for h in HORIZONTES:
        dados = erros[(erros.horizonte == h) & ((erros.familia != "experimentos_cnn") | erros.selecionado_na_validacao_10)]
        fig, eixo = plt.subplots(figsize=(10, 5))
        for chave, serie in dados.groupby(["modelo", "familia", "semente"], dropna=False):
            rotulo = chave[0] if chave[1] != "experimentos_cnn" else f"10/{chave[0]} (s={int(chave[2])})"
            eixo.plot(serie.data, serie.erro_absoluto.expanding().mean(), label=rotulo)
        eixo.set(title=f"MAE acumulado no teste — {h} pregão(ões)", xlabel="Origem", ylabel="MAE desde o início do teste")
        eixo.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(pasta / f"mae_acumulado_{h}d.png", dpi=150)
        plt.close(fig)
    registro = dict(inicio=inicio.isoformat(), fim=datetime.now().astimezone().isoformat(),
                    avaliacoes=len(linhas), python=platform.python_version(), torch=str(torch.__version__),
                    sistema=platform.platform(), cpu=platform.processor(), cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(dispositivo) if dispositivo.type == "cuda" else None)
    (pasta / "execucao.json").write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    basico = OUTPUTS / "avaliacao_teste" / "ultimo_run.json"
    temporario = pasta / "marcador.tmp"
    temporario.write_text(json.dumps({"pasta": str(pasta.relative_to(RAIZ))}), encoding="utf-8")
    temporario.replace(basico)
    print(f"Resultados: {pasta}")


if __name__ == "__main__":
    main()
