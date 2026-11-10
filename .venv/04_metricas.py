"""Métricas comuns aos baselines e às redes do projeto.

Todos os valores devem estar na escala original dos log-retornos (decimal),
com valores reais e previsões alinhados nas mesmas datas e na mesma ordem.
Este módulo é importado pelos modelos; não executa treinamento nem salva runs.
"""

import numpy as np


def _vetor(valores):
    """Converte uma série em vetor e rejeita entradas inválidas."""
    vetor = np.asarray(valores, dtype=float)
    if vetor.ndim != 1 or vetor.size == 0:
        raise ValueError("Informe um vetor unidimensional não vazio.")
    if not np.isfinite(vetor).all():
        raise ValueError("Há valores ausentes ou infinitos; confira o alinhamento.")
    return vetor


def _erros(reais, previstos):
    """Evita que tamanhos diferentes produzam resultados incorretos."""
    reais = _vetor(reais)
    previstos = _vetor(previstos)
    if reais.shape != previstos.shape:
        raise ValueError("Valores reais e previsões devem ter o mesmo tamanho.")
    return reais - previstos


def mae(reais, previstos):
    """Erro absoluto médio, na mesma unidade dos retornos."""
    return float(np.mean(np.abs(_erros(reais, previstos))))


def rmse(reais, previstos):
    """Raiz do erro quadrático médio; penaliza mais os erros grandes."""
    return float(np.sqrt(np.mean(_erros(reais, previstos) ** 2)))


def escala_mase(alvos_treino, persistencia_treino):
    """Calcula uma vez o MAE da persistência, exclusivamente no treino.

    Para cada origem t e horizonte h:
      alvo futuro = ln(P_{t+h} / P_t);
      persistência = ln(P_t / P_{t-h}), conhecida em t.

    Nos CSVs do script 03, use as colunas alvo e log_retorno_1d ou
    log_retorno_5d, conforme o horizonte. Antes desta chamada, selecione
    as mesmas linhas com ambas as colunas válidas, em ordem cronológica.
    Não use previsão zero como persistência dos retornos.
    """
    escala = mae(alvos_treino, persistencia_treino)
    if escala <= 0:
        raise ValueError("MASE indefinida: o erro da persistência no treino é zero.")
    return escala


def mase(reais, previstos, escala):
    """Divide o MAE pela escala da persistência previamente obtida no treino.

    Reutilize a mesma escala em treino, validação e teste, para cada horizonte.
    MASE é adimensional. Não recalculamos a escala no conjunto avaliado.
    """
    if not np.isscalar(escala) or not np.isfinite(escala) or escala <= 0:
        raise ValueError("A escala da MASE deve ser um número positivo e finito.")
    return mae(reais, previstos) / float(escala)


def calcular_metricas(reais, previstos, escala):
    """Retorna números simples, prontos para o registro de cada execução."""
    return {
        "MAE": mae(reais, previstos),
        "RMSE": rmse(reais, previstos),
        "MASE": mase(reais, previstos, escala),
    }
