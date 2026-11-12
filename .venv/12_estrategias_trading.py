"""Backtest complementar do MAE 5920. Execute depois do script 11, na mesma pasta."""
import json
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# EDITE AQUI. Os custos são por lado, em pontos-base (1 pb = 0,01%).
RUN_11 = None  # Nome da pasta temporal. None usa o último run concluído.
CUSTO_MERCADO_PB = 1.0
CUSTOS_PB = [0, 1, 5, 10]
FATOR_FAIXA = 0.25
CAPITAL = 100.0
RAIZ = Path(__file__).resolve().parent
MODELOS = {"retorno_zero": "baselines_simples", "media_historica": "baselines_simples",
           "ar_direto": "baseline_ar", "cnn_minima": "cnn_minima", "cnn_residual": "cnn_residual"}
REGRAS = ["comprado_caixa", "comprado_vendido", "faixa_inacao"]
ROTULOS = ["Comprado ou caixa", "Comprado ou vendido", "Faixa de inação"]


def calendario(h, precos, previsoes):
    # Conferimos as datas de cada modelo e a correspondência entre alvos e preços.
    dados = pd.read_csv(RAIZ / f"data/conjuntos/teste_{h}d.csv", parse_dates=["data", "data_alvo"])
    datas = pd.DatetimeIndex(dados.data)
    indices = precos.index.get_indexer(datas)
    finais = precos.index.get_indexer(dados.data_alvo)
    if (len(datas) <= h or (indices < 0).any() or not (np.diff(indices) == 1).all()
            or not np.array_equal(indices + h, finais)):
        raise ValueError(f"Calendário de teste inconsistente para {h} pregões.")
    alvo = np.log(precos.iloc[finais].to_numpy() / precos.iloc[indices].to_numpy())
    if not np.allclose(alvo, dados.alvo, atol=1e-10):
        raise ValueError("Preços e alvos do teste divergem.")
    for modelo, familia in MODELOS.items():
        serie = previsoes.query("modelo == @modelo and familia == @familia and horizonte == @h").sort_values("data")
        if (not datas.equals(pd.DatetimeIndex(serie.data)) or not np.isfinite(serie.previsao).all()
                or not np.array_equal(serie.data_alvo, dados.data_alvo)
                or not np.allclose(serie.alvo, alvo, atol=1e-10)):
            raise ValueError(f"Previsões ausentes, duplicadas ou desalinhadas: {modelo}, h={h}.")
        dados[modelo] = serie.previsao.to_numpy()
    # Cada bloco começa no fechamento seguinte e termina h intervalos depois.
    # Removemos somente o bloco final que exigiria preço além do teste.
    origens = np.arange(0, len(datas), h)
    origens = origens[indices[origens] + 1 + h <= finais[-1]]
    inicio = indices[origens[0]] + 1
    fim = indices[origens[-1]] + 1 + h
    return dados.iloc[origens], precos.iloc[inicio:fim + 1]


def posicoes(regra, sinais, faixa, h):
    # Cada valor retornado indica a posição para um intervalo fechamento a fechamento.
    if regra == "comprado_caixa":
        direcoes = (sinais > 0).astype(int)
    elif regra == "comprado_vendido":
        direcoes = np.sign(sinais).astype(int)
    else:
        direcoes = []
        pos = 0
        for sinal in sinais:
            if sinal > faixa:
                pos = 1
            elif sinal < -faixa:
                pos = 0
            direcoes.append(pos)
    return np.repeat(direcoes, h)


def simular(precos, pos, custo):
    # A posição decide a direção. Mantemos a quantidade até ela mudar.
    # Na virada, encerramos a posição anterior e abrimos a nova já líquida dos custos.
    caixa, qtd, anterior = CAPITAL, 0.0, 0
    linhas = []
    for i, preco in enumerate(precos):
        alvo = int(pos[i]) if i < len(pos) else 0  # Liquidação no último fechamento.
        antes = caixa + qtd * preco
        negociado = 0.0
        if alvo != anterior:
            negociado = abs(qtd) * preco
            caixa = antes - custo * negociado
            qtd = alvo * caixa / ((1 + custo) * preco)
            entrada = abs(qtd) * preco
            caixa -= qtd * preco + custo * entrada
            negociado += entrada
        patrimonio = caixa + qtd * preco
        if patrimonio <= 0:
            raise ValueError("Patrimônio esgotado; confira a trajetória da posição vendida.")
        linhas.append((anterior, alvo, qtd, negociado, negociado/antes, negociado*custo, patrimonio))
        anterior = alvo
    df = pd.DataFrame(linhas, index=precos.index, columns=[
        "posicao_intervalo_encerrado", "posicao_apos_fechamento", "quantidade",
        "valor_negociado", "giro", "custo_pago", "patrimonio"])
    # O primeiro retorno inclui a entrada inicial. Há exatamente um retorno por intervalo.
    base = df.patrimonio.shift().to_numpy(copy=True)
    base[1] = CAPITAL
    df["retorno_diario"] = df.patrimonio / base - 1
    return df


def limites(precos, pos):
    # Agrupamos dias de mesma posição. Os fatores abaixo reproduzem o caixa da simulação.
    bordas = np.r_[0, np.flatnonzero(np.diff(pos)) + 1, len(pos)]
    direcao = pos[bordas[:-1]]
    razoes = precos.to_numpy()[bordas[1:]] / precos.to_numpy()[bordas[:-1]]
    comprados, vendidos = razoes[direcao == 1], razoes[direcao == -1]
    indice = precos.iloc[-1] / precos.iloc[0]

    def patrimonio_relativo(z):
        # z=(1-c)/(1+c). Uma ida e volta comprada multiplica por razão*z;
        # vendida, por 1+z-razão. Capital esgotado encerra a estratégia.
        return np.prod(comprados*z) * np.prod(np.maximum(0, 1 + z - vendidos))

    resultado = {}
    for nome, passivo in [("absoluto", False), ("sobre_indice", True)]:
        def saldo(z):
            return patrimonio_relativo(z) - (indice*z if passivo else 1)

        limite = np.nan
        if not np.any(pos):
            status = "sem operacoes"
        elif saldo(1) <= 1e-12:
            status = "sem vantagem bruta"
        else:
            # O produto de fatores lineares positivos crescentes é convexo em z.
            # Subtrair o índice (linear) mantém a convexidade. Localizamos o mínimo
            # para detectar a primeira perda de vantagem mesmo em carteiras só vendidas.
            a, b = 0., 1.
            for _ in range(65):
                u, v = (2*a+b)/3, (a+2*b)/3
                if saldo(u) < saldo(v):
                    b = v
                else:
                    a = u
            inferior = min([0., (a+b)/2], key=saldo)
            if saldo(inferior) >= -1e-12:
                status = "vantagem mantida para custos abaixo de 100%"
                limite = 100.
            else:
                superior = 1.
                for _ in range(60):
                    meio = (inferior+superior)/2
                    if saldo(meio) >= 0:
                        superior = meio
                    else:
                        inferior = meio
                limite = 100 * (1-superior)/(1+superior)
                status = "equilibrio"
        resultado[f"custo_limite_{nome}_pct_por_lado"] = limite
        resultado[f"situacao_{nome}"] = status
    return resultado


def graficos(diario, pasta, h):
    # Cada painel usa a mesma escala vertical; os benchmarks aparecem nos três.
    custos = list(dict.fromkeys([0., CUSTO_MERCADO_PB]))
    fig, eixos = plt.subplots(3, len(custos), figsize=(14, 10), sharex=True, sharey=True, squeeze=False)
    for i, regra in enumerate(REGRAS):
        for j, custo in enumerate(custos):
            ax = eixos[i, j]
            dados = diario[(diario.regra.isin([regra, "benchmark"])) & (diario.custo_pb == custo)]
            for modelo, serie in dados.groupby("modelo", sort=False):
                benchmark = modelo in ["caixa", "comprar_manter"]
                ax.plot(serie.data, 100 * (serie.patrimonio / CAPITAL - 1), label=modelo,
                        linestyle="--" if benchmark else "-", linewidth=1.5 if benchmark else 1)
            ax.set_title(f"{ROTULOS[i]} | {custo:g} pb por lado")
            ax.set_ylabel("Rentabilidade acumulada (%)")
            ax.grid(alpha=.2)
    handles, labels = eixos[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(f"S&P 500 — estratégias baseadas na previsão de {h} pregão(ões)")
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0,.07,1,.97])
    fig.savefig(pasta / f"rentabilidades_{h}d.png", dpi=160)
    plt.close(fig)


def main():
    inicio = datetime.now().astimezone()
    relogio, cpu = perf_counter(), process_time()
    raiz11 = RAIZ / "outputs/avaliacao_teste"
    if RUN_11:
        origem = raiz11 / RUN_11
    else:
        marcador = json.loads((raiz11 / "ultimo_run.json").read_text(encoding="utf-8"))
        origem = RAIZ / marcador["pasta"].replace(chr(92), "/")
    previsoes = pd.read_csv(origem / "erros_por_data.csv", parse_dates=["data", "data_alvo"])
    precos = pd.read_csv(RAIZ / "data/trabalho/sp500_log_retornos.csv", parse_dates=["data"]).set_index("data").SP500
    if (not precos.index.is_unique or not precos.index.is_monotonic_increasing or precos.index.hasnans
            or not np.isfinite(precos).all() or (precos <= 0).any()):
        raise ValueError("Histórico de preços inválido.")
    custos = sorted(set([0., CUSTO_MERCADO_PB, *CUSTOS_PB]))
    if not all(np.isfinite(c) and 0 <= c < 10000 for c in custos) or not np.isfinite(FATOR_FAIXA) or FATOR_FAIXA < 0:
        raise ValueError("Confira os custos por lado e o multiplicador da faixa.")
    pasta = RAIZ / "outputs/estrategias_trading" / inicio.strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True)
    linhas, faixas = [], {}
    for h in (1, 5):
        sinais, serie = calendario(h, precos, previsoes)
        treino = pd.read_csv(RAIZ / f"data/conjuntos/treino_{h}d.csv", parse_dates=["data_alvo"])
        if (len(treino) < 2 or not np.isfinite(treino.alvo).all() or treino.data_alvo.isna().any()
                or treino.data_alvo.max() >= sinais.data.min()):
            raise ValueError("Revise os alvos e a separação temporal do treino.")
        faixa = FATOR_FAIXA * treino.alvo.std(ddof=1)
        faixas[h] = faixa
        carteiras = [("benchmark", "caixa", np.zeros(len(serie)-1, int)),
                     ("benchmark", "comprar_manter", np.ones(len(serie)-1, int))]
        for regra in REGRAS:
            for modelo in MODELOS:
                carteiras.append((regra, modelo, posicoes(regra, sinais[modelo].to_numpy(), faixa, h)))
        diarios = []
        for regra, modelo, pos in carteiras:
            inicio_carteira = datetime.now().astimezone()
            t_carteira, cpu_carteira = perf_counter(), process_time()
            equilibrio = limites(serie, pos)
            for pb in custos:
                inicio_cenario = datetime.now().astimezone()
                t_cenario, cpu_cenario = perf_counter(), process_time()
                df = simular(serie, pos, pb/10000)
                df.index.name = "data"
                df = df.reset_index()
                df["data_sinal"] = pd.NaT
                df["previsao_na_decisao"] = np.nan
                if regra != "benchmark":
                    decisoes = df.index[:-1:h]
                    df.loc[decisoes, "data_sinal"] = sinais.data.to_numpy()
                    df.loc[decisoes, "previsao_na_decisao"] = sinais[modelo].to_numpy()
                df["regra"], df["modelo"], df["horizonte"], df["custo_pb"] = regra, modelo, h, pb
                diarios.append(df)
                r = df.retorno_diario.dropna()
                vol = r.std(ddof=1)
                pico = df.patrimonio.cummax().clip(lower=CAPITAL)
                linhas.append(dict(regra=regra, modelo=modelo, horizonte=h, custo_pb=pb,
                                   retorno_pct=100*(df.patrimonio.iloc[-1]/CAPITAL-1),
                                   sharpe=np.sqrt(252)*r.mean()/vol if vol > 0 else np.nan,
                                   drawdown_max_pct=100*(1-df.patrimonio/pico).max(),
                                   dias_comprado=int((pos==1).sum()), dias_vendido=int((pos==-1).sum()),
                                   dias_caixa=int((pos==0).sum()), giro_total=df.giro.sum(),
                                   custo_total_pago=df.custo_pago.sum(), inicio_cenario=inicio_cenario.isoformat(),
                                   fim_cenario=datetime.now().astimezone().isoformat(),
                                   tempo_cenario_s=perf_counter()-t_cenario, tempo_cpu_cenario_s=process_time()-cpu_cenario,
                                   **equilibrio))
            print(f"{inicio_carteira.isoformat()} | h={h} {modelo} {regra} | "
                  f"{perf_counter()-t_carteira:.3f} s, CPU {process_time()-cpu_carteira:.3f} s", flush=True)
        diario = pd.concat(diarios, ignore_index=True)
        diario.to_csv(pasta / "carteiras_diarias.csv", mode="w" if h==1 else "a", header=h==1, index=False)
        graficos(diario, pasta, h)
    tabela = pd.DataFrame(linhas)
    chave = ["horizonte", "custo_pb"]
    referencia = tabela[tabela.modelo=="comprar_manter"][chave + ["retorno_pct"]]
    tabela = tabela.merge(referencia.rename(columns={"retorno_pct":"retorno_indice_pct"}), on=chave, validate="many_to_one")
    tabela["excesso_sobre_indice_pp"] = tabela.retorno_pct - tabela.retorno_indice_pct
    tabela.to_csv(pasta / "sensibilidade_custos.csv", index=False)
    principal = tabela[tabela.custo_pb == CUSTO_MERCADO_PB].copy()
    chaves = ["regra", "modelo", "horizonte"]
    bruto = tabela[tabela.custo_pb == 0][chaves + ["retorno_pct", "sharpe"]]
    principal = principal.merge(bruto.rename(columns={"retorno_pct":"retorno_bruto_pct", "sharpe":"sharpe_bruto"}),
                               on=chaves, validate="one_to_one")
    principal.to_csv(pasta / "resumo_estrategias.csv", index=False)
    registro = dict(inicio=inicio.isoformat(), fim=datetime.now().astimezone().isoformat(),
                    tempo_total_s=perf_counter()-relogio, tempo_cpu_s=process_time()-cpu,
                    dispositivo="CPU", sistema=platform.platform(), python=platform.python_version(),
                    pandas=pd.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__,
                    run_11=str(origem.relative_to(RAIZ)), custos_pb_por_lado=custos, custo_mercado_pb=CUSTO_MERCADO_PB,
                    faixa_log_por_horizonte=faixas, fator_faixa=FATOR_FAIXA, capital_inicial=CAPITAL,
                    premissas="Índice de preços; caixa e taxa livre de risco zero; sem aluguel, dividendos ou margem. "
                              "Sinal no fechamento, execução no fechamento seguinte, blocos completos sem sobreposição.")
    (pasta / "execucao.json").write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp = pasta / "marcador.tmp"
    tmp.write_text(json.dumps({"pasta":str(pasta.relative_to(RAIZ))}), encoding="utf-8")
    tmp.replace(pasta.parent / "ultimo_run.json")
    print(principal[["horizonte", "modelo", "regra", "retorno_bruto_pct", "retorno_pct", "sharpe"]].to_string(index=False))
    print(f"Resultados: {pasta}")


if __name__ == "__main__":
    main()
