"""
Módulo 1: O Baseline e o Desafio da Dimensionalidade
--------------------------------------------------
Objetivo:
1. Criar um conjunto de dados sintético que emule dados de saúde de alta dimensionalidade
   (com atributos informativos, redundantes e ruídos).
2. Treinar um classificador de referência (Baseline) usando todas as variáveis.
3. Medir o tempo de treinamento e extrair métricas de desempenho (Acurácia, F1-Score, ROC-AUC).
4. Gerar visualizações gráficas da Matriz de Confusão e da Curva ROC.
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def gerar_dataset_sintetico_saude(n_samples=2000, n_features=40, n_informative=10, n_redundant=10, random_state=42):
    """
    Gera um dataset sintético representando um cenário clínico/biomédico:
    - 40 atributos no total
    - 10 atributos genuinamente informativos (sinais biológicos/clínicos reais)
    - 10 atributos redundantes (correlacionados com os informativos)
    - 20 atributos de ruído puro (sem valor preditivo)
    """
    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        weights=[0.6, 0.4], # Leve desbalanceamento típico de diagnósticos
        flip_y=0.03,         # 3% de ruído nos rótulos
        random_state=random_state
    )
    
    # Nomes didáticos para os atributos
    feature_names = (
        [f"biomarcador_{i+1}" for i in range(n_informative)] +
        [f"exame_redundante_{i+1}" for i in range(n_redundant)] +
        [f"ruido_metabolico_{i+1}" for i in range(n_features - n_informative - n_redundant)]
    )
    
    df_X = pd.DataFrame(X_raw, columns=feature_names)
    return df_X, y

def treinar_e_avaliar_baseline(X_train, X_test, y_train, y_test, random_state=42):
    """
    Treina o modelo Baseline (Random Forest) com todas as variáveis e mede o tempo de execução.
    """
    modelo = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
    
    # Medindo tempo de treinamento
    inicio_treino = time.perf_counter()
    modelo.fit(X_train, y_train)
    tempo_treino = time.perf_counter() - inicio_treino
    
    # Medindo tempo de inferência
    inicio_inf = time.perf_counter()
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    tempo_inferencia = time.perf_counter() - inicio_inf
    
    # Métricas
    metricas = {
        "acuracia": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tempo_treino_s": tempo_treino,
        "tempo_inferencia_s": tempo_inferencia,
        "num_atributos": X_train.shape[1]
    }
    
    return modelo, metricas, y_pred, y_proba

def gerar_graficos_baseline(y_test, y_pred, y_proba, output_path="modulo1_baseline_metrics.png"):
    """
    Gera a Matriz de Confusão e a Curva ROC do modelo Baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Matriz de Confusão (Baseline - 40 Atributos)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predição")
    axes[0].set_ylabel("Valor Real")
    axes[0].set_xticklabels(["Saudável (0)", "Patologia (1)"])
    axes[0].set_yticklabels(["Saudável (0)", "Patologia (1)"])
    
    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    
    axes[1].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Random Forest Baseline (AUC = {auc_val:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Aleatório (AUC = 0.50)")
    axes[1].set_title("Curva ROC (Baseline)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Taxa de Falsos Positivos (FPR)")
    axes[1].set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Gráfico salvo com sucesso em: {output_path}")

def teste_de_sanidade(metricas, X_train):
    """
    Validação automatizada de integridade do Módulo 1.
    """
    assert metricas["num_atributos"] == 40, f"Erro: Esperado 40 atributos, obtido {metricas['num_atributos']}"
    assert metricas["acuracia"] > 0.70, f"Erro: Acurácia muito baixa ({metricas['acuracia']:.2f})"
    assert metricas["tempo_treino_s"] > 0, "Erro: Tempo de treino inválido"
    print("\n[OK] SANITY CHECK PASSOU: Modelo treinado, tempo contabilizado e metricas validadas com sucesso!")

def executar_modulo_1():
    print("=" * 60)
    print("MÓDULO 1: BASELINE E O DESAFIO DA DIMENSIONALIDADE")
    print("=" * 60)
    
    # 1. Gerar dados
    X, y = gerar_dataset_sintetico_saude()
    print(f"[*] Dataset gerado: {X.shape[0]} amostras, {X.shape[1]} atributos.")
    
    # Split Treino / Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 2. Treinar Baseline
    modelo_baseline, metricas, y_pred, y_proba = treinar_e_avaliar_baseline(X_train, X_test, y_train, y_test)
    
    # Exibir Métricas
    print("\n--- MÉTRICAS DO MODELO BASELINE (TODOS OS 40 ATRIBUTOS) ---")
    print(f"• Atributos Utilizados : {metricas['num_atributos']}")
    print(f"• Acurácia             : {metricas['acuracia']:.4f}")
    print(f"• F1-Score             : {metricas['f1_score']:.4f}")
    print(f"• Precisão             : {metricas['precision']:.4f}")
    print(f"• Recall               : {metricas['recall']:.4f}")
    print(f"• ROC-AUC              : {metricas['roc_auc']:.4f}")
    print(f"• Tempo de Treino      : {metricas['tempo_treino_s']*1000:.2f} ms")
    print(f"• Tempo de Inferência   : {metricas['tempo_inferencia_s']*1000:.2f} ms")
    
    # 3. Teste de Sanidade
    teste_de_sanidade(metricas, X_train)
    
    # 4. Gerar Gráficos
    gerar_graficos_baseline(y_test, y_pred, y_proba)
    
    return X_train, X_test, y_train, y_test, modelo_baseline, metricas

if __name__ == "__main__":
    executar_modulo_1()
