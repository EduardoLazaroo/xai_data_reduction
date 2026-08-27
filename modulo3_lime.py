"""
Módulo 3: LIME e a Sensibilidade de Vizinhança
--------------------------------------------------
Objetivo:
1. Explicar a matemática de amostragem por perturbação local do LIME.
2. Treinar um explicador LIME Tabular no dataset de treino.
3. Gerar explicações para instâncias limítrofes (casos de alta incerteza preditiva).
4. Gerar e salvar a visualização gráfica dos coeficientes locais (|LIME weights|).
5. Validar via Sanity Check.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime import lime_tabular

from modulo1_baseline import gerar_dataset_sintetico_saude, treinar_e_avaliar_baseline
from sklearn.model_selection import train_test_split

def criar_explicador_lime(X_train):
    """
    Instancia o LimeTabularExplainer.
    """
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=["Saudável (0)", "Patologia (1)"],
        mode="classification",
        random_state=42
    )
    return explainer

def explicar_instancia_lime(explainer, modelo, X_test, index_instancia=0, num_features=10):
    """
    Gera a explicação local LIME para uma instância específica do conjunto de teste.
    """
    instancia = X_test.iloc[index_instancia]
    exp = explainer.explain_instance(
        data_row=np.array(instancia),
        predict_fn=modelo.predict_proba,
        num_features=num_features
    )
    
    list_weights = exp.as_list()
    df_lime = pd.DataFrame(list_weights, columns=["regra_atributo", "peso_local"])
    return exp, df_lime

def gerar_grafico_lime(df_lime, proba_instancia, index_instancia=0, output_path="modulo3_lime_local.png"):
    """
    Plota os pesos locais calculados pelo LIME para a instância analisada.
    """
    plt.figure(figsize=(10, 6))
    
    cores = ['#2ca02c' if peso > 0 else '#d62728' for peso in df_lime['peso_local']]
    
    plt.barh(df_lime['regra_atributo'][::-1], df_lime['peso_local'][::-1], color=cores[::-1])
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f"LIME Local Explanation (Instância #{index_instancia} - Prob. Patologia: {proba_instancia:.2%})", 
              fontsize=12, fontweight="bold")
    plt.xlabel("Contribuição Local do Atributo (Peso LIME)")
    plt.grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Aumenta probabilidade de Patologia (1)'),
        Patch(facecolor='#d62728', label='Diminui probabilidade de Patologia (0)')
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfico LIME salvo com sucesso em: {output_path}")

def teste_de_sanidade_lime(df_lime):
    """
    Sanity Check do Módulo 3:
    - Verifica se a explicação LIME gerou atributos válidos e pesos não-nulos.
    """
    assert len(df_lime) > 0, "Erro: Explicação LIME veio vazia"
    assert df_lime["peso_local"].abs().sum() > 0, "Erro: Pesos locais do LIME somam zero"
    print("\n[OK] SANITY CHECK PASSOU: Explicação local LIME gerada e validada com sucesso!")

def executar_modulo_3():
    print("=" * 60)
    print("MÓDULO 3: LIME E A SENSIBILIDADE DE VIZINHANÇA")
    print("=" * 60)
    
    # 1. Carregar dados e baseline
    X, y = gerar_dataset_sintetico_saude()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    modelo_baseline, metricas, _, y_proba = treinar_e_avaliar_baseline(X_train, X_test, y_train, y_test)
    
    # 2. Criar explicador LIME
    explainer = criar_explicador_lime(X_train)
    
    # Selecionar uma instância no limiar de decisão (próxima de 0.5 de probabilidade) para máxima sensibilidade
    idx_limiar = np.argmin(np.abs(y_proba - 0.5))
    print(f"[*] Selecionada instância #{idx_limiar} para explicação (Probabilidade prevista = {y_proba[idx_limiar]:.4f})")
    
    # 3. Gerar explicação local LIME
    exp, df_lime = explicar_instancia_lime(explainer, modelo_baseline, X_test, index_instancia=idx_limiar)
    
    print("\n--- REGRAS DE ATRIBUTOS E PESOS LOCAIS LIME ---")
    for i, row in df_lime.iterrows():
        print(f" {i+1:2d}. {row['regra_atributo']:<40} | Peso: {row['peso_local']:+.4f}")
        
    # 4. Teste de Sanidade
    teste_de_sanidade_lime(df_lime)
    
    # 5. Gerar Gráfico
    gerar_grafico_lime(df_lime, y_proba[idx_limiar], index_instancia=idx_limiar)
    
    return explainer, df_lime

if __name__ == "__main__":
    executar_modulo_3()
