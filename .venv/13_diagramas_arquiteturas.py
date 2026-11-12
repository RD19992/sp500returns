"""Desenha as CNNs dos runs 08, 09 e 10 a partir dos resumos gravados no treino."""
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# EDITE AQUI. None usa o último run concluído da família.
RUNS = {"cnn_minima": None, "cnn_residual": None, "experimentos_cnn": None}
FORMATOS = ["pdf", "png"]
DPI = 450
RAIZ = Path(__file__).resolve().parent
TITULOS = {"cnn_minima": "CNN mínima", "cnn_residual": "CNN residual dilatada",
           "experimentos_cnn": "CNN experimental"}
AZUL, TINTA, CINZA = "#165772", "#20323d", "#536875"


def ler(p):
    return json.loads(p.read_text(encoding="utf-8"))


def resumos(familia, nome):
    raiz = RAIZ / "outputs" / familia
    pasta = raiz / nome if nome else RAIZ / ler(raiz / "ultimo_run.json")["pasta"].replace(chr(92), "/")
    if familia == "experimentos_cnn":
        plano = ler(pasta / "plano.json")
        for c in plano["configuracoes"]:
            for h in c["horizontes"]:
                for s in plano["sementes"]:
                    yield pasta / f"h{h}" / c["experimento"] / f"semente_{s}" / "resumo.json"
    else:
        for h in ler(pasta / "config.json")["horizontes"]:
            yield pasta / f"h{h}" / "resumo.json"


def arquitetura(familia, resumo):
    # Espelhamos a ordem de operações de CNN.forward nos três scripts.
    c = resumo["config"]
    residual = familia == "cnn_residual"
    n = c["blocos"] if residual else c["profundidade"]
    canais = c["filtros"] if residual else [c["filtros"]] * n
    kernels = c["kernels"] if residual else [c["kernel"]] * n
    dilatacoes = c["dilatacoes"]
    if not len(canais) == len(kernels) == len(dilatacoes) == n:
        raise ValueError("Número de camadas incompatível com a configuração salva.")
    leitura = c.get("leitura", "ultima")
    t, entrada, campo, parametros = c["janela"], 1, 1, 0
    blocos = []
    for saida, k, d in zip(canais, kernels, dilatacoes):
        incremento = (k - 1) * d * (2 if residual else 1)
        campo += incremento
        if c["padding"] == "valid":
            t -= incremento
        parametros += saida * (k * entrada + 1)
        if residual:
            parametros += saida * (k * saida + 1)
            if c["residuos"] and entrada != saida:
                parametros += saida * (entrada + 1)
        blocos.append((entrada, saida, k, d, t))
        entrada = saida
    parametros += entrada + 1  # Cabeça linear com viés.
    if (t < 1 or parametros != resumo["parametros"] or campo != resumo["campo_receptivo"]
            or resumo["status"] != "concluido"):
        raise ValueError("O diagrama não confere com os valores registrados no treino.")
    return dict(blocos=blocos, janela=c["janela"], padding=c["padding"], leitura=leitura,
                residual=residual, atalho=c.get("residuos", False), dropout=c.get("dropout", 0),
                relu2=c.get("relu_antes_da_soma", True), campo=campo, parametros=parametros)


def desenhar(a, titulo, destino):
    # A coluna esquerda mostra o fluxo. À direita, cada ponto cheio é um peso
    # temporal do kernel; a distância horizontal representa a dilatação.
    residual = a["residual"]
    passo = 1.90 if residual else 1.04
    centros = [2.30 + i * passo for i in range(len(a["blocos"]))]
    leitura_y = centros[-1] + (1.74 if residual else 1.08)
    altura = leitura_y + 1.62
    fig, ax = plt.subplots(figsize=(8.1, altura * .64))
    fig.subplots_adjust(left=.025, right=.975, top=.985, bottom=.015)
    ax.set(xlim=(0, 10), ylim=(altura, 0))
    ax.axis("off")
    def texto(x, y, s, **kw):
        ax.text(x, y, s, va="center", color=kw.pop("color", TINTA), **kw)
    def seta(x1, y1, x2, y2):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(
            arrowstyle="-|>", color=CINZA, lw=.9, shrinkA=0, shrinkB=0, mutation_scale=9))
    def caixa(y, rotulo, sub="", tom="white"):
        ax.add_patch(FancyBboxPatch((1.15,y-.31), 4.08,.62, boxstyle="round,pad=.01,rounding_size=.055",
                                    facecolor=tom, edgecolor=AZUL, linewidth=.9))
        texto(3.19,y-(.115 if sub else 0),rotulo,ha="center",fontsize=10,weight="semibold")
        if sub:
            texto(3.19,y+.14,sub,ha="center",fontsize=8.4)
    texto(.10,.23,titulo,fontsize=16,weight="semibold")
    padding = "causal à esquerda" if a["padding"] == "causal" else "valid (sem preenchimento)"
    texto(.10,.61,f'{a["parametros"]:,} parâmetros   ·   Padding {padding}   ·   Campo local final: {a["campo"]} pregões'.replace(","," "),fontsize=9)
    caixa(1.23,"Entrada temporal",f'1 canal × {a["janela"]} log-retornos diários')
    texto(5.66,1.13,"Amostras de uma convolução",fontsize=10,weight="semibold")
    texto(5.66,1.43,"Passado à esquerda; posição atual à direita",fontsize=8.4)
    anterior = 1.55
    atraso_max = max(1,max((k-1)*d for _,_,k,d,_ in a["blocos"]))
    for i, (y, (cin, cout, k, d, t)) in enumerate(zip(centros, a["blocos"]),1):
        seta(3.19,anterior,3.19,y-.32)
        titulo_bloco = f"Bloco {i}   ·   2 × Conv1d" if residual else f"Conv1d {i}   →   ReLU"
        caixa(y,titulo_bloco,f"{cin} → {cout} canais   ·   k = {k}   ·   d = {d}   ·   T = {t}", "#eaf3f5")
        x0, x1 = 5.72, 9.62
        ax.plot([x0,x1],[y,y],color="#bccbd0",lw=.8)
        for j in range(k):
            x = x1 - (x1-x0)*j*d/atraso_max
            ax.plot(x,y,"o",ms=5,color=AZUL)
        texto(x1,y+.25, f"atrasos: {', '.join(str(j*d) for j in reversed(range(k)))}",ha="right",fontsize=8.3)
        anterior=y+.32
        if residual:
            atalho=a["atalho"]
            if atalho:
                inicio=y-.49
                ax.plot([3.19,.65,.65],[inicio,inicio,y+.89],color=CINZA,lw=.9)
                seta(.65,y+.89,1.14,y+.89)
                texto(.58,y+.20,"1×1" if cin!=cout else "Id",rotation=90,ha="right",fontsize=8)
            seta(3.19,anterior,3.19,y+.57)
            caixa(y+.89,"Soma  →  ReLU" if atalho else "ReLU")
            anterior=y+1.21
    seta(3.19,anterior,3.19,leitura_y-.32)
    caixa(leitura_y,"Última posição" if a["leitura"]=="ultima" else "Média temporal global",
          f'{a["blocos"][-1][1]} características')
    seta(3.19,leitura_y+.32,3.19,leitura_y+.57)
    caixa(leitura_y+.89,"Linear  →  previsão escalar","Alvo padronizado de um horizonte", "#eaf3f5")
    if residual:
        p = a["dropout"]
        texto(5.66,leitura_y-.24,"Ordem no ramo convolucional",fontsize=9.4,weight="semibold")
        texto(5.66,leitura_y+.08,f"Conv 1 → ReLU → Dropout1d ({p:g})",fontsize=8.5)
        relu = "ReLU → " if a["relu2"] else ""
        texto(5.66,leitura_y+.39,f"Conv 2 → {relu}Dropout1d ({p:g})",fontsize=8.5)
        if a["atalho"]:
            texto(5.66,leitura_y+.79,"Id: identidade. 1×1: projeção de canais.",fontsize=8)
            if a["padding"]=="valid":
                texto(5.66,leitura_y+1.10,"Atalho mantém as últimas T posições.",fontsize=8)
    for formato in FORMATOS:
        fig.savefig(destino.with_suffix("."+formato), dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=.09)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none"})
    pasta=RAIZ/"outputs/diagramas_arquiteturas"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pasta.mkdir(parents=True)
    grupos, linhas = {}, []
    for familia, nome in RUNS.items():
        for arquivo in resumos(familia,nome):
            r=ler(arquivo)
            a=arquitetura(familia,r)
            chave=(familia,json.dumps(a,sort_keys=True))
            if chave not in grupos:
                nome_fig=f'{familia}_{len(grupos)+1:02d}'
                grupos[chave]=nome_fig
                titulo=TITULOS[familia]
                if familia=="experimentos_cnn":
                    titulo += " · " + r["modelo"].replace("_"," ")
                desenhar(a,titulo,pasta/nome_fig)
            c=r["config"]
            linhas.append(dict(figura=grupos[chave],familia=familia,modelo=r["modelo"],horizonte=r["horizonte"],
                               semente=c["semente"],parametros=a["parametros"],campo_local=a["campo"],
                               inicializacao=c["inicializacao"],otimizador=c["otimizador"],
                               taxa_aprendizagem=c["taxa_aprendizagem"],weight_decay=c["weight_decay"],
                               origem=str(arquivo.relative_to(RAIZ))))
    with (pasta/"indice_arquiteturas.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=linhas[0].keys())
        w.writeheader()
        w.writerows(linhas)
    print(f"{len(grupos)} arquiteturas desenhadas para {len(linhas)} ajustes. Resultados: {pasta}")


if __name__=="__main__":
    main()
