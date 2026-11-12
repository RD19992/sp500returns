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
    eixo.set(xlabel="Época", ylabel="RMSE do log-retorno", title=f"EXTRA — CNN residual Huber — horizonte de {h} pregões")
    eixo.legend()
    figura.tight_layout()
    figura.savefig(pasta / "curva_aprendizagem.png", dpi=150)
    plt.close(figura)
    sincronizar(dispositivo)
    registro = {
        "status": "concluido", "modelo": "extra_cnn_residual_huber", "horizonte": h, "config": config,
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
    return {"semente": config["semente"], "horizonte": h, "pasta": str(pasta.relative_to(RAIZ)), "pesos": str((pasta / 'pesos_melhor.pt').relative_to(RAIZ))}


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
    if min(CONFIG[k] for k in ["janela", "batch_size", "epocas", "paciencia", "threads_cpu", "clip_gradiente", "taxa_aprendizagem",  "lr_minima"]) <= 0:
        raise ValueError("Dimensões e taxas devem ser positivas.")
    limite_s = CONFIG["tempo_maximo_por_horizonte_s"]
    if limite_s is not None and (not np.isfinite(limite_s) or limite_s <= 0):
        raise ValueError("Use None ou um limite de tempo positivo em segundos.")
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
    horizontes, sementes = CONFIG["horizontes"], CONFIG["sementes"]
    if not horizontes or len(set(horizontes)) != len(horizontes) or not set(horizontes) <= {1, 5}:
        raise ValueError("Escolha horizontes únicos entre 1 e 5.")
    if not sementes or len(set(sementes)) != len(sementes):
        raise ValueError("Informe pelo menos uma semente, sem repetição.")
    if CONFIG["perda"] != "huber" or not np.isfinite(CONFIG["delta_huber"]) or CONFIG["delta_huber"] <= 0:
        raise ValueError("Use perda huber com delta finito e positivo.")

    parametros = sum(p.numel() for p in CNN(CONFIG).parameters())
    if parametros > 1_000_000:
        raise ValueError(f"A rede tem {parametros:,} parâmetros e excede o teto de 1 milhão por rede.")
    print(f"EXTRA — {parametros:,} parâmetros por rede, {len(horizontes) * len(sementes)} ajustes.")
    pasta = PASTA_RESULTADOS / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True, exist_ok=False)
    salvar_json(pasta / "config.json", CONFIG)
    runs = []
    for h in horizontes:
        for semente in sementes:
            config = dict(CONFIG, semente=semente)
            destino = pasta / f"h{h}" / f"semente_{semente}"
            destino.mkdir(parents=True)
            try:
                runs.append(treinar(h, destino, dispositivo, config))
            except Exception as erro:
                salvar_json(destino / "falha.json", {"status": "falhou", "erro": str(erro), "config": config})
                raise

    # O marcador só avança depois que todos os ajustes terminam.
    marcador = {"pasta": str(pasta.relative_to(RAIZ)), "runs": runs}
    salvar_json(pasta / "execucao.json", marcador)
    temporario = pasta / "marcador.json"
    salvar_json(temporario, marcador)
    temporario.replace(PASTA_RESULTADOS / "ultimo_run.json")
    print(f"EXTRA — último run atualizado em {PASTA_RESULTADOS / 'ultimo_run.json'}")


if __name__ == "__main__":
    main()
