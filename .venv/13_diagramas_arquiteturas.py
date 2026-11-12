"""Desenha as CNNs dos runs 08, 09 e 10 a partir dos resumos gravados no treino."""
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# EDITE AQUI. None usa o último run concluído da família.
RUNS = {"cnn_minima": None, "cnn_residual": None, "experimentos_cnn": None}
FORMATOS = ["pdf", "png"]
DPI = 450
RAIZ = Path(__file__).resolve().parent
TITULOS = {"cnn_minima": "CNN mínima", "cnn_residual": "CNN residual dilatada",
           "experimentos_cnn": "CNN experimental"}



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
    # Um bloco genérico mostra o fluxo. A tabela descreve cada ocorrência do bloco.
    blocos = a["blocos"]
    residual = a["residual"]
    n = len(blocos)
    altura = 3.15 + .28*n + (.43 if residual else 0)
    fig, ax = plt.subplots(figsize=(7.05, altura))
    ax.set(xlim=(0, 7.05), ylim=(altura, 0))
    fig.subplots_adjust(left=.025, right=.98, bottom=.02, top=.98)
    ax.axis("off")

    def txt(x, y, s, tam=10, ha="left"):
        ax.text(x,y,s,fontsize=tam,ha=ha,va="center",color="#171717")

    def linha(x1,y1,x2,y2):
        ax.plot([x1,x2],[y1,y2],lw=.65,color="#454545")

    def seta(x1,y1,x2,y2):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(
            arrowstyle="-|>",lw=.7,color="#393939",mutation_scale=8,shrinkA=0,shrinkB=0))

    txt(.08,.16,titulo,14)
    txt(6.95,.16,f'{a["parametros"]:,} parâmetros'.replace(","," "),10,"right")
    linha(.08,.36,6.95,.36)
    y = 1.00
    txt(.10,.54,"(a)   Transformação de um bloco" if residual else "(a)   Transformação de uma camada")
    txt(.27,y,r"$x_{i-1}$",13)
    seta(.79,y,1.48,y)
    txt(1.98,y,r"$F_i$",14,"center")
    seta(2.29,y,3.06,y)
    if residual and a["atalho"]:
        txt(3.24,y,r"$\oplus$",14,"center")
        linha(1.04,y,1.04,y-.25)
        linha(1.04,y-.25,3.24,y-.25)
        seta(3.24,y-.25,3.24,y-.10)
        txt(2.10,y-.34,r"$S_i$",9,"center")
        seta(3.42,y,4.01,y)
        txt(4.40,y,"ReLU",11,"center")
        seta(4.78,y,5.41,y)
        txt(5.54,y,r"$x_i$",13)
    else:
        txt(3.52,y,"ReLU",11,"center")
        seta(3.97,y,4.69,y)
        txt(4.83,y,r"$x_i$",13)
    if residual:
        segunda = " → ReLU" if a["relu2"] else ""
        txt(.10,1.34,f'$F_i$: Conv → ReLU → D → Conv{segunda} → D.   D = Dropout1d (p = {a["dropout"]:g}).',9)
        if a["atalho"]:
            nota="$S_i$: identidade ou projeção 1 × 1 quando muda o número de canais."
            if a["padding"]=="valid":
                nota += " Alinhamento à direita."
            txt(.10,1.54,nota,9)
    else:
        txt(.10,1.32,"$F_i$: convolução temporal unidimensional.",9)
    yb=1.92 if residual else 1.60
    txt(.10,yb,"(b)   Sequência de blocos e suporte temporal dos kernels" if residual
        else "(b)   Sequência de camadas e suporte temporal dos kernels")
    yb+=.30
    cols=[.23,1.04,1.85,2.26,2.72,3.28]
    for x,s in zip(cols,["i","Canais","k","d","T","Campo¹"]):
        txt(x,yb,s,9,"center")
    txt(5.37,yb,"Posição relativa na entrada da camada",9,"center")
    linha(.08,yb+.14,6.95,yb+.14)
    campo=1
    atraso=max((k-1)*d for _,_,k,d,_ in blocos)
    x0,x1=3.95,6.80
    for i,(entrada,saida,k,d,t) in enumerate(blocos,1):
        yy=yb+.16+.28*i
        campo+=(k-1)*d*(2 if residual else 1)
        for x,s in zip(cols,[str(i),f"{entrada} → {saida}",str(k),str(d),str(t),str(campo)]):
            txt(x,yy,s,10,"center")
        # Os pontos mostram a mesma escala de defasagens em todas as linhas.
        xs=[x1-(x1-x0)*j*d/max(atraso,1) for j in range(k)]
        linha(min(xs),yy,x1,yy)
        ax.plot(xs,[yy]*k,"o",ms=3.5,color="#171717")
    eixo=yy+.21
    if atraso:
        linha(x0,eixo,x1,eixo)
    for lag in sorted({0,atraso//2,atraso}):
        x=x1-(x1-x0)*lag/max(atraso,1)
        linha(x,eixo,x,eixo+.04)
        txt(x,eixo+.15,"t" if lag==0 else f"t − {lag}",8,"center")
    yfim=eixo+.39
    linha(.08,yfim,6.95,yfim)
    leitura="última posição" if a["leitura"]=="ultima" else "média temporal global"
    pad="causal à esquerda" if a["padding"]=="causal" else "valid"
    txt(.10,yfim+.18,f'Entrada: {a["janela"]} retornos.   Padding: {pad}.   Leitura: {leitura} → linear → escalar.',9)
    txt(.10,yfim+.38,"¹ Campo receptivo local, em pregões.   T: comprimento temporal após o bloco.",8.5)
    for formato in FORMATOS:
        fig.savefig(destino.with_suffix("."+formato), dpi=DPI, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family":"STIXGeneral","font.weight":"normal","mathtext.fontset":"stix","pdf.fonttype":42,"svg.fonttype":"none"})
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