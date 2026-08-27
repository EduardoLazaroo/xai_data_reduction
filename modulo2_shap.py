"""
Módulo 2: SHAP e a Teoria dos Jogos Cooperativos
--------------------------------------------------
Objetivo:
1. Explicar a matemática dos Valores Shapley (atribuição aditiva proporcional baseada em Teoria dos Jogos).
2. Computar explicações globais e locais usando o algoritmo TreeSHAP no modelo Baseline.
3. Gerar gráficos de Summary Plot (beeswarm) e Importância Global Média (|SHAP|).
4. Validar os resultados via Sanity Check.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from modulo1_baseline import gerar_dataset_sintetico_saude, treinar_e_avaliar_baseline

def calcular_explicacoes_shap(modelo, X_train):
    """
    Calcula os valores SHAP usando o algoritmo otimizado TreeExplainer.
    """
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer(X_train)
    
    # Se for classificação binária com formato 3D (n_samples, n_features, n_classes)
    if len(shap_values.shape) == 3:
        shap_vals_class1 = shap_values.values[:, :, 1]
    else:
        shap_vals_class1 = shap_values.values
        
    # Importância global média |SHAP|
    mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
    df_importancia = pd.DataFrame({
        "atributo": X_train.columns,
        "importancia_shap": mean_abs_shap
    }).sort_values(by="importancia_shap", ascending=False).reset_index(drop=True)
    
    return explainer, shap_values, shap_vals_class1, df_importancia

def gerar_graficos_shap(shap_values, X_train, df_importancia, output_path="modulo2_shap_summary.png"):
    """
    Plota o Bar Plot de relevância global e o Beeswarm Summary Plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Bar Plot de Importância Global Média (|SHAP|)
    top15 = df_importancia.head(15)
    colors = ['#1f77b4' if 'biomarcador' in name else ('#ff7f0e' if 'redundante' in name else '#d62728') for name in top15['atributo']]
    
    axes[0].barh(top15['atributo'][::-1], top15['importancia_shap'][::-1], color=colors[::-1])
    axes[0].set_title("Top 15 Atributos por Importância Média |SHAP|", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Impacto Médio Absoluto no Output do Modelo (|SHAP value|)")
    axes[0].grid(True, alpha=0.3)
    
    # Legenda de cores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Biomarcadores (Informativos)'),
        Patch(facecolor='#ff7f0e', label='Exames Redundantes'),
        Patch(facecolor='#d62728', label='Ruído Metabólico')
    ]
    axes[0].legend(handles=legend_elements, loc="lower right")
    
    # 2. Beeswarm Plot customizado nos 15 principais atributos
    plt.sca(axes[1])
    # Extrair shap_values classe 1 para o beeswarm
    if len(shap_values.shape) == 3:
        sv_plot = shap_values[:, :, 1]
    else:
        sv_plot = shap_values
        
    shap.plots.beeswarm(sv_plot, max_display=15, show=False)
    axes[1].set_title("SHAP Beeswarm Plot (Distribuição de Impacto Local)", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfico SHAP salvo com sucesso em: {output_path}")

def teste_de_sanidade_shap(df_importancia, shap_vals_class1, X_train):
    """
    Sanity Check do Módulo 2:
    - Verifica dimensões da matriz SHAP
    - Valida se os atributos informativos (biomarcadores) dominam o topo do ranking em comparação com ruídos
    """
    assert shap_vals_class1.shape == X_train.shape, f"Dimensão SHAP inconsistente: {shap_vals_class1.shape}"
    
    top5_atributos = df_importancia.head(5)["atributo"].tolist()
    informaticos_no_top5 = sum(1 for feat in top5_atributos if "biomarcador" in feat or "redundante" in feat)
    
    assert informaticos_no_top5 >= 3, f"Erro: Esperado que biomarcadores/redundantes dominassem o Top 5, mas obtido {top5_atributos}"
    print("\n[OK] SANITY CHECK PASSOU: Matriz SHAP computada corretamente e atributos informativos validados no topo do ranking!")

def executar_modulo_2():
    print("=" * 60)
    print("MÓDULO 2: SHAP E A TEORIA DOS JOGOS COOPERATIVOS")
    print("=" * 60)
    
    # 1. Carregar dados e baseline
    from sklearn.model_selection import train_test_split
    X, y = gerar_dataset_sintetico_saude()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    modelo_baseline, metricas, _, _ = treinar_e_avaliar_baseline(X_train, X_test, y_train, y_test)
    
    # 2. Computar SHAP
    print("[*] Computando valores TreeSHAP no conjunto de treinamento...")
    explainer, shap_values, shap_vals_class1, df_importancia = calcular_explicacoes_shap(modelo_baseline, X_train)
    
    print("\n--- TOP 10 ATRIBUTOS MAIS RELEVANTES SEGUNTO O SHAP ---")
    for i, row in df_importancia.head(10).iterrows():
        print(f" {i+1:2d}. {row['atributo']:<25} | Importância Média |SHAP|: {row['importancia_shap']:.4f}")
        
    # 3. Teste de Sanidade
    teste_de_sanidade_shap(df_importancia, shap_vals_class1, X_train)
    
    # 4. Gerar Gráficos
    gerar_graficos_shap(shap_values, X_train, df_importancia)
    
    return explainer, shap_values, df_importancia

if __name__ == "__main__":
    executar_modulo_2()
