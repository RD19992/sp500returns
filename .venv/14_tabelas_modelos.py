"""Gera uma tabela editável e uma figura por modelo avaliado no script 11."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RAIZ = Path(__file__).resolve().parent
METRICAS = ["MAE", "RMSE", "MASE"]
CONJUNTOS = ["treino", "validacao", "teste"]
ROTULOS = {"treino": "Treino", "validacao": "Validação", "teste": "Teste"}
NOMES = {"retorno_zero": "Retorno zero", "media_historica": "Média histórica",
         "ar_direto": "AR direto", "cnn_minima": "CNN mínima", "cnn_residual": "CNN residual"}
DPI = 400


def ler_json(arquivo):
    return json.loads(arquivo.read_text(encoding="utf-8"))


def caminho_salvo(texto):
    # Os caminhos registrados no Windows também podem ser lidos no Linux.
    return RAIZ / texto.replace(chr(92), "/")


def reunir(pasta):
    teste = pd.read_csv(pasta / "metricas_teste.csv")
    chaves = ["familia", "modelo", "horizonte", "semente"]
    if (teste.empty or teste.duplicated(chaves).any()
            or len(teste) != ler_json(pasta / "execucao.json")["avaliacoes"]):
        raise ValueError(f"Inventário de avaliações incompleto ou duplicado em {pasta}")

    linhas, custos = [], []
    for t in teste.to_dict("records"):
        # Usamos exatamente o ajuste avaliado, mesmo que exista treino mais recente.
        modelo = caminho_salvo(t["arquivo_modelo"])
        arquivo = modelo.with_name("resumo.json") if modelo.suffix == ".pt" else modelo
        r = ler_json(arquivo)
        if (r["modelo"], r["horizonte"]) != (t["modelo"], t["horizonte"]):
            raise ValueError(f"Modelo ou horizonte incompatível em {arquivo}")
        if modelo.suffix == ".pt" and r["config"]["semente"] != t["semente"]:
            raise ValueError(f"Semente incompatível em {arquivo}")
        if sorted(x["conjunto"] for x in r["resultados"]) != sorted(CONJUNTOS[:2]):
            raise ValueError(f"Esperava treino e validação em {arquivo}")

        chave = {k: t[k] for k in chaves + ["selecionado_na_validacao_10", "arquivo_modelo"]}
        for x in [*r["resultados"], dict(t, conjunto="teste")]:
            linha = {k: x[k] for k in ["conjunto", "observacoes", *METRICAS]}
            linhas.append({**chave, **linha})

        # Para o AR, o custo do ajuste inclui a busca entre ordens candidatas.
        custos.append(dict(**chave, parametros=r.get("parametros", r.get("parametros_estimados",
                           r.get("parametros_estimados_media"))), melhor_epoca=r.get("melhor_epoca"),
                           inicio_treino=r["inicio"], fim_treino=r["fim"],
                           ajuste_s=r.get("tempo_ajuste_s", r.get("tempo_ajuste_todos_candidatos_s")),
                           total_treino_validacao_s=r.get("tempo_total_s", r.get("tempo_total_modelo_s")),
                           inicio_teste=t["inicio"], fim_teste=t["fim"], dispositivo_teste=t["dispositivo"],
                           previsao_teste_s=t["tempo_previsao_s"], total_teste_s=t["tempo_total_avaliacao_s"]))
    detalhes = pd.DataFrame(linhas)
    if detalhes[METRICAS + ["observacoes"]].isna().any().any():
        raise ValueError("Há métricas ou contagens ausentes nos registros de origem.")
    return detalhes, pd.DataFrame(custos)


def resumir(dados):
    # Cada métrica permanece na escala original. O DP amostral mede a variação entre sementes.
    linhas = []
    for h, bloco in dados.groupby("horizonte"):
        for conjunto in CONJUNTOS:
            grupo = bloco[bloco.conjunto == conjunto]
            if grupo.observacoes.nunique() != 1:
                raise ValueError(f"Número de observações divergente entre sementes em {conjunto}, h={h}.")
            linha = dict(horizonte=h, conjunto=conjunto, observacoes=grupo.observacoes.iloc[0],
                         ajustes=len(grupo), selecionado_na_validacao_10=grupo.selecionado_na_validacao_10.iloc[0])
            for m in METRICAS:
                linha[m] = grupo[m].mean()
                linha[m + "_dp"] = grupo[m].std()
            linhas.append(linha)
    return pd.DataFrame(linhas)


def desenhar(tabela, titulo, destino):
    # Tipografia incluída no Matplotlib, alinhamento numérico e traços horizontais.
    plt.rcParams.update({"font.family": "STIXGeneral", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.1, 1.42 + 0.29 * len(tabela)))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.2, len(tabela) + 2)
    ax.text(0, len(tabela) + 1.45, titulo, fontsize=15, va="center")
    posicoes = [0, .105, .32, .54, .77, 1]
    cabecalho = ["Pregões", "Conjunto", "N", *METRICAS]
    for i, valor in enumerate(cabecalho):
        ax.text(posicoes[i], len(tabela) + .45, valor, ha="left" if i < 2 else "right")
    ax.axhline(len(tabela) + .21, lw=.8, color="black")
    for j, linha in tabela.iterrows():
        y = len(tabela) - j - .48
        if j and linha.conjunto == "treino":
            ax.axhline(y + .68, lw=.45, color=".65")
        valores = [str(linha.horizonte), ROTULOS[linha.conjunto], str(linha.observacoes)]
        for m in METRICAS:
            casas = 4 if m == "MASE" else 6
            valor = f"{linha[m]:.{casas}f}"
            if linha.ajustes > 1:
                valor += f" ± {linha[m + '_dp']:.{casas}f}"
            valores.append(valor)
        for i, valor in enumerate(valores):
            ax.text(posicoes[i], y, valor, ha="left" if i < 2 else "right")
    ax.axhline(.18, lw=.8, color="black")
    nota = "MAE e RMSE em log-retorno decimal. MASE adimensional."
    if tabela.ajustes.max() > 1:
        nota += " Média ± DP entre sementes. N por ajuste."
    ax.text(0, -.64, nota, fontsize=9)
    fig.subplots_adjust(left=.035, right=.97, top=.98, bottom=.025)
    for extensao in ["png", "pdf"]:
        fig.savefig(destino.with_suffix("." + extensao), dpi=DPI, facecolor="white")
    plt.close(fig)


def salvar_csv(tabela, arquivo):
    # Separador e vírgula decimal facilitam a abertura no Excel em português.
    tabela.to_csv(arquivo, index=False, sep=";", decimal=",", encoding="utf-8-sig")


def main():
    marcador = RAIZ / "outputs/avaliacao_teste/ultimo_run.json"
    if not marcador.exists():
        raise FileNotFoundError(f"Execute primeiro o script 11. Marcador esperado em {marcador}")
    origem = caminho_salvo(ler_json(marcador)["pasta"])
    detalhes, custos = reunir(origem)
    # Uma pasta por execução preserva as versões anteriores do relatório.
    agora = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pasta = RAIZ / "outputs/tabelas_modelos" / agora
    originais, figuras = pasta / "originais", pasta / "figuras"
    originais.mkdir(parents=True)
    figuras.mkdir()
    salvar_csv(detalhes, originais / "metricas_por_ajuste.csv")
    salvar_csv(custos, originais / "custos_por_ajuste.csv")
    grupos = detalhes.groupby(["familia", "modelo"], sort=False)
    for (familia, modelo), dados in grupos:
        resumo = resumir(dados)
        nome = f"{familia}__{modelo}"
        titulo = NOMES.get(modelo, "CNN experimental — " + modelo.replace("_", " "))
        salvar_csv(resumo, originais / (nome + ".csv"))
        desenhar(resumo, titulo, figuras / nome)
    (pasta / "origem.txt").write_text(str(origem.relative_to(RAIZ)), encoding="utf-8")
    print(f"{len(grupos)} tabelas, {len(custos)} ajustes. Avaliação de origem: {origem.name}")
    print(f"Resultados: {pasta}")


if __name__ == "__main__":
    main()
