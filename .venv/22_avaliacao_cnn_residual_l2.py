"""Avaliação da CNN residual com L2 selecionada pelo script 21, com benchmarks e backtest."""
import importlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

avaliacao = importlib.import_module("11_avaliar_modelos_teste")
trading = importlib.import_module("12_estrategias_trading")
diagramas = importlib.import_module("13_diagramas_arquiteturas")
tabelas = importlib.import_module("14_tabelas_modelos")

# None usa o último treino completo. Para reproduzir outro, informe sua pasta temporal.
RUN_21 = None
DISPOSITIVO = "auto"
LOTE = 64
RAIZ = Path(__file__).resolve().parent
MODELO = "cnn_residual_l2"
TITULO = "CNN residual — seleção de L2 por validação"


def ler(arquivo):
    return json.loads(arquivo.read_text(encoding="utf-8"))


def salvar(arquivo, conteudo):
    arquivo.write_text(json.dumps(conteudo, indent=2, ensure_ascii=False), encoding="utf-8")


def avaliar_horizonte(h, sementes, origem, selecao, historico, dispositivo, pasta):
    # Conferimos a purga também pela realização do último alvo de validação.
    base = RAIZ / "data/conjuntos"
    teste = pd.read_csv(base / f"teste_{h}d.csv", parse_dates=["data", "data_alvo"])
    val = pd.read_csv(base / f"validacao_{h}d.csv", parse_dates=["data_alvo"])
    if (teste.empty or val.empty or val.data_alvo.isna().any() or teste.data.isna().any()
            or teste.data.duplicated().any() or not teste.data.is_monotonic_increasing
            or not (teste.data_alvo > teste.data).all() or not np.isfinite(teste.alvo).all()
            or val.data_alvo.max() >= teste.data.min()):
        raise ValueError(f"Datas ou alvos incompatíveis com a avaliação em h={h}.")

    linhas, erros, comparacoes = [], [], []
    for s in sementes:
        arquivo = origem / f"h{h}" / selecao["candidato"] / f"semente_{s}" / "pesos_melhor.pt"
        resumo = ler(arquivo.with_name("resumo.json"))
        if resumo["config"]["semente"] != s or resumo["status"] != "concluido":
            raise ValueError(f"Ajuste incompleto ou semente incompatível em {arquivo}")
        if resumo["config"]["lambda_l2"] != selecao["lambda_l2"] or resumo["config"]["weight_decay"] != 0:
            raise ValueError(f"Penalização divergente da seleção em {arquivo}")
        # O avaliador 11 recupera a semente do checkpoint para CNNs independentes.
        item = (MODELO, MODELO, h, None, False, arquivo)
        linha, detalhe = avaliacao.avaliar(item, teste, historico.log_retorno_1d, dispositivo)
        if not np.isfinite(detalhe.previsao).all():
            raise ValueError(f"Previsões inválidas em {arquivo}")
        media = resumo["normalizacao"]["media_y"]
        rmse_media = np.sqrt(np.mean((teste.alvo - media) ** 2))
        linha.update(rmse_media_historica=rmse_media, ganho_RMSE_sobre_media=rmse_media-linha["RMSE"],
                     melhor_epoca=resumo["melhor_epoca"], previsor_selecionado=resumo["previsor_selecionado"])
        linha["lambda_l2"] = selecao["lambda_l2"]
        for candidato in ("selecionado", "controle_09", "sem_l2"):
            if candidato == "selecionado":
                metrica = linha
            else:
                nome = "controle_09" if candidato == "controle_09" else selecao["candidato_zero"]
                controle = origem / f"h{h}" / nome / f"semente_{s}" / "pesos_melhor.pt"
                if controle == arquivo:
                    metrica = linha
                else:
                    item_controle = (MODELO, MODELO, h, None, False, controle)
                    metrica, _ = avaliacao.avaliar(item_controle, teste, historico.log_retorno_1d, dispositivo)
                if metrica["semente"] != s:
                    raise ValueError(f"Semente incorreta em {controle}")
            comparacoes.append(dict(horizonte=h, semente=s, candidato=candidato,
                                    **{k: metrica[k] for k in ("MAE", "RMSE", "MASE", "arquivo_modelo")}))
        linhas.append(linha)
        erros.append(detalhe)
        print(f"h={h}, semente={s}, RMSE={linha['RMSE']:.6f}", flush=True)

    # A média das métricas entre sementes e a previsão de um ensemble são objetos distintos.
    # Aqui cada checkpoint tem sua própria previsão, métrica e carteira.
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for s, detalhe in zip(sementes, erros):
        ax.plot(detalhe.data, detalhe.erro_absoluto.expanding().mean(), label=f"Semente {s}")
    ax.set(title=f"CNN residual L2 — erro absoluto médio acumulado, h={h}", ylabel="MAE", xlabel="Origem")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pasta / f"mae_acumulado_{h}d.png", dpi=200)
    plt.close(fig)
    return linhas, pd.concat(erros, ignore_index=True), comparacoes


def simular_carteiras(h, erros, precos, pasta):
    # O calendário do 12 alinha previsões e preços por data, aplica um pregão de
    # defasagem na execução e usa blocos de h pregões sem sobreposição.
    sinais = erros.copy()
    sinais["modelo"] = sinais.semente.map(lambda s: f"semente_{s}")
    trading.MODELOS = dict.fromkeys(sinais.modelo.unique(), MODELO)
    decisoes, serie = trading.calendario(h, precos, sinais)
    treino = pd.read_csv(RAIZ / f"data/conjuntos/treino_{h}d.csv", parse_dates=["data_alvo"])
    if (len(treino) < 2 or not np.isfinite(treino.alvo).all() or treino.data_alvo.isna().any()
            or treino.data_alvo.max() >= decisoes.data.min()):
        raise ValueError(f"Treino inválido para definir a faixa em h={h}.")
    faixa = trading.FATOR_FAIXA * treino.alvo.std()
    n = len(serie) - 1
    carteiras = [("benchmark", "caixa", np.zeros(n)), ("benchmark", "comprar_manter", np.ones(n))]
    for regra in trading.REGRAS:
        for nome in trading.MODELOS:
            pos = trading.posicoes(regra, decisoes[nome].to_numpy(), faixa, h)
            carteiras.append((regra, nome, pos))

    custos = sorted({0., trading.CUSTO_MERCADO_PB, *trading.CUSTOS_PB})
    if not all(0 <= c < 10000 for c in custos) or not np.isfinite(faixa) or faixa < 0:
        raise ValueError("Custos ou faixa de inação inválidos.")
    diarios, linhas = [], []
    for regra, nome, pos in carteiras:
        equilibrio = trading.limites(serie, pos)
        for pb in custos:
            inicio = datetime.now().astimezone()
            t, cpu = perf_counter(), process_time()
            df = trading.simular(serie, pos, pb / 10000).rename_axis("data").reset_index()
            df["data_sinal"], df["previsao_na_decisao"] = pd.NaT, np.nan
            if regra != "benchmark":
                indices = df.index[:-1:h]
                df.loc[indices, "data_sinal"] = decisoes.data.to_numpy()
                df.loc[indices, "previsao_na_decisao"] = decisoes[nome].to_numpy()
            chave = dict(horizonte=h, modelo=nome, regra=regra, custo_pb=pb)
            diarios.append(df.assign(**chave))
            r = df.retorno_diario.dropna()
            vol = r.std()
            pico = df.patrimonio.cummax().clip(lower=trading.CAPITAL)
            linhas.append(dict(**chave, faixa_log=faixa, **equilibrio,
                               retorno_pct=100*(df.patrimonio.iloc[-1]/trading.CAPITAL-1),
                               sharpe=np.sqrt(252)*r.mean()/vol if vol > 0 else np.nan,
                               drawdown_max_pct=100*(1-df.patrimonio/pico).max(),
                               dias_comprado=int((pos==1).sum()), dias_vendido=int((pos==-1).sum()),
                               dias_caixa=int((pos==0).sum()), giro_total=df.giro.sum(),
                               custo_total_pago=df.custo_pago.sum(), inicio=inicio.isoformat(),
                               fim=datetime.now().astimezone().isoformat(),
                               tempo_s=perf_counter()-t, tempo_cpu_s=process_time()-cpu))
    diario = pd.concat(diarios, ignore_index=True)
    diario.to_csv(pasta / f"carteiras_diarias_{h}d.csv", index=False)
    trading.graficos(diario, pasta, h)
    return linhas


def comparar_previsoes(erros, historico, janela, pasta, dispositivo):
    """Mesmas datas de teste. A regressão usa apenas exemplos completos do treino."""
    linhas = []
    for h, grupo in erros.groupby("horizonte"):
        primeira = grupo[grupo.semente == grupo.semente.iloc[0]].sort_values("data")
        for s, dados in grupo.groupby("semente"):
            dados = dados.sort_values("data")
            if not dados[["data", "alvo"]].reset_index(drop=True).equals(
                    primeira[["data", "alvo"]].reset_index(drop=True)):
                raise ValueError("Datas ou alvos divergentes entre sementes.")
            linhas.append(dict(horizonte=h, modelo=MODELO, semente=s,
                               RMSE=np.sqrt(np.mean((dados.alvo - dados.previsao)**2))))
        treino = pd.read_csv(RAIZ / f"data/conjuntos/treino_{h}d.csv", parse_dates=["data"])
        media = treino.alvo.mean()
        sinal = historico.log_retorno_1d.rolling(janela).sum()
        x = sinal.reindex(treino.data).to_numpy()
        ok = np.isfinite(x)
        design = np.column_stack([np.ones(ok.sum()), x[ok]])
        coef, _, posto, _ = np.linalg.lstsq(design, treino.alvo.to_numpy()[ok], rcond=None)
        xt = sinal.reindex(primeira.data).to_numpy()
        if posto != 2 or not np.isfinite(xt).all():
            raise ValueError("Dados insuficientes para a regressão de momentum.")
        for nome, previsao in [("media_historica", media), ("momentum_linear", coef[0] + coef[1]*xt)]:
            linhas.append(dict(horizonte=h, modelo=nome, semente=None,
                               RMSE=np.sqrt(np.mean((primeira.alvo - previsao)**2))))
        marcador = RAIZ / "outputs/cnn_residual/ultimo_run.json"
        if marcador.exists():
            origem = avaliacao.caminho_salvo(ler(marcador)["pasta"])
            arquivo = origem / f"h{h}" / "pesos_melhor.pt"
            if arquivo.is_file():
                item = ("cnn_residual", "cnn_residual", h, None, False, arquivo)
                metrica, _ = avaliacao.avaliar(item, primeira[["data", "data_alvo", "alvo"]],
                                               historico.log_retorno_1d, dispositivo)
                linhas.append(dict(horizonte=h, modelo="cnn_residual", semente=metrica["semente"],
                                   RMSE=metrica["RMSE"], arquivo_modelo=metrica["arquivo_modelo"]))
    pd.DataFrame(linhas).to_csv(pasta / "comparacao_rmse.csv", index=False)


def main():
    inicio = datetime.now().astimezone()
    t, cpu = perf_counter(), process_time()
    raiz21 = RAIZ / "outputs" / MODELO
    origem = raiz21 / RUN_21 if RUN_21 else avaliacao.caminho_salvo(ler(raiz21 / "ultimo_run.json")["pasta"])
    config = ler(origem / "config.json")
    sementes, horizontes = config["sementes"], config["horizontes"]
    execucao = ler(origem / "execucao.json")
    inventario = execucao["runs"]
    selecionados = execucao["selecionados"]
    esperados = {(h, s) for h in horizontes for s in sementes}
    if len(inventario) != len(esperados) or {(r["horizonte"], r["semente"]) for r in inventario} != esperados:
        raise ValueError("Inventário do treino 21 incompleto.")

    # Registramos a classe do 09 somente neste processo. Os scripts originais
    # continuam usando seus próprios modelos e suas pastas de saída.
    avaliacao.scripts[MODELO] = "09_cnn_residual"
    avaliacao.LOTE = LOTE
    dispositivo = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                               if DISPOSITIVO == "auto" else DISPOSITIVO)
    torch.set_num_threads(config["threads_cpu"])
    pasta = RAIZ / "outputs/avaliacao_cnn_residual_l2" / inicio.strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True, exist_ok=False)
    for nome in ["teste", "trading", "arquitetura", "tabelas"]:
        (pasta / nome).mkdir()
    plt.rcParams.update({"font.family":"STIXGeneral", "mathtext.fontset":"stix", "pdf.fonttype":42})
    historico = pd.read_csv(RAIZ / "data/trabalho/sp500_log_retornos.csv",
                            parse_dates=["data"]).set_index("data")
    if (historico.index.hasnans or not historico.index.is_unique or not historico.index.is_monotonic_increasing
            or not np.isfinite(historico.SP500).all() or (historico.SP500 <= 0).any()):
        raise ValueError("Histórico de preços inválido.")

    ajustes, detalhes, carteiras = [], [], []
    comparacoes = []
    for h in horizontes:
        selecao = selecionados[str(h)].copy()
        if any(r["candidato"] != selecao["candidato"] for r in inventario if r["horizonte"] == h):
            raise ValueError(f"Inventário incompatível com a seleção em h={h}.")
        selecao["candidato_zero"] = f"l2_{config['lambdas_l2'].index(0):02d}"
        linhas, erros, controles = avaliar_horizonte(h, sementes, origem, selecao, historico, dispositivo, pasta / "teste")
        comparacoes.extend(controles)
        ajustes.extend(linhas)
        detalhes.append(erros)
        carteiras.extend(simular_carteiras(h, erros, historico.SP500, pasta / "trading"))

    # Guardamos erros assinados, erros absolutos e os dez maiores de cada ajuste.
    teste = pasta / "teste"
    pd.DataFrame(ajustes).to_csv(teste / "metricas_teste.csv", index=False)
    erros = pd.concat(detalhes, ignore_index=True)
    erros.to_csv(teste / "erros_por_data.csv", index=False)
    maiores = erros.sort_values("erro_absoluto", ascending=False).groupby(["horizonte", "semente"]).head(10)
    maiores.to_csv(teste / "maiores_erros.csv", index=False)
    salvar(teste / "execucao.json", {"avaliacoes": len(ajustes)})

    # Calculamos o excesso sobre o investimento passivo sob o mesmo custo.
    estrategias = pd.DataFrame(carteiras)
    chaves = ["horizonte", "custo_pb"]
    indice = estrategias[estrategias.modelo=="comprar_manter"][chaves + ["retorno_pct"]]
    estrategias = estrategias.merge(indice.rename(columns={"retorno_pct":"retorno_indice_pct"}),
                                    on=chaves, validate="many_to_one")
    estrategias["excesso_sobre_indice_pp"] = estrategias.retorno_pct - estrategias.retorno_indice_pct
    estrategias.to_csv(pasta / "trading/sensibilidade_custos.csv", index=False)
    chaves = ["horizonte", "modelo", "regra"]
    bruto = estrategias[estrategias.custo_pb==0][chaves + ["retorno_pct", "sharpe"]]
    principal = estrategias[estrategias.custo_pb==trading.CUSTO_MERCADO_PB].merge(
        bruto.rename(columns={"retorno_pct":"retorno_bruto_pct", "sharpe":"sharpe_bruto"}), on=chaves,
        validate="one_to_one")
    principal.to_csv(pasta / "trading/resumo_estrategias.csv", index=False)

    arquiteturas, catalogo = {}, []
    for linha in ajustes:
        resumo = ler(avaliacao.caminho_salvo(linha["arquivo_modelo"]).with_name("resumo.json"))
        arquitetura = diagramas.arquitetura("cnn_residual", resumo)
        chave = json.dumps(arquitetura, sort_keys=True)
        if chave not in arquiteturas:
            nome = f"residual_l2_{len(arquiteturas)+1:02d}"
            diagramas.desenhar(arquitetura, TITULO, pasta / "arquitetura" / nome)
            arquiteturas[chave] = nome
        catalogo.append(dict(horizonte=linha["horizonte"], semente=linha["semente"],
                             diagrama=arquiteturas[chave], parametros=resumo["parametros"],
                             melhor_epoca=resumo["melhor_epoca"], previsor=resumo["previsor_selecionado"]))
    pd.DataFrame(catalogo).to_csv(pasta / "arquitetura/catalogo.csv", index=False)

    pd.DataFrame(comparacoes).to_csv(teste / "comparacao_pareada_l2.csv", index=False)
    salvar(pasta / "selecao_validacao.json", selecionados)
    comparar_previsoes(erros, historico, config["janela"], pasta / "teste", dispositivo)

    # O 14 recupera treino e validação do mesmo checkpoint avaliado no teste.
    dados, custos = tabelas.reunir(teste)
    resumo = tabelas.resumir(dados)
    saida = pasta / "tabelas"
    tabelas.salvar_csv(dados, saida / "metricas_por_ajuste.csv")
    tabelas.salvar_csv(custos, saida / "custos_por_ajuste.csv")
    tabelas.salvar_csv(resumo, saida / "resumo.csv")
    tabelas.desenhar(resumo, TITULO, saida / "resumo")

    # Publicamos o marcador somente após a geração de todas as saídas.
    salvar(pasta / "execucao.json", dict(run_21=str(origem.relative_to(RAIZ)), avaliacoes=len(ajustes),
           inicio=inicio.isoformat(), fim=datetime.now().astimezone().isoformat(),
           dispositivo=str(dispositivo), torch=str(torch.__version__), cuda=torch.version.cuda,
           gpu=torch.cuda.get_device_name(dispositivo) if dispositivo.type=="cuda" else None,
           tempo_total_s=perf_counter()-t, tempo_cpu_s=process_time()-cpu,
           custo_mercado_pb=trading.CUSTO_MERCADO_PB, capital_inicial=trading.CAPITAL,
           fator_faixa=trading.FATOR_FAIXA, premissas="Caixa e taxa livre de risco zero. Sem aluguel ou dividendos."))
    temporario = pasta / "marcador.tmp"
    salvar(temporario, {"pasta":str(pasta.relative_to(RAIZ))})
    temporario.replace(pasta.parent / "ultimo_run.json")
    print(resumo[resumo.conjunto=="teste"][["horizonte", "ajustes", "MAE", "RMSE", "MASE"]].to_string(index=False))
    print(f"Resultados CNN residual L2: {pasta}")


if __name__ == "__main__":
    main()
