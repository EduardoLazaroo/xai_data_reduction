"""
Módulo 4: Poda Guiada por XAI vs. Seleção Tradicional (Ablação de Atributos)
-------------------------------------------------------------------------
Objetivo:
1. Implementar a lógica de ablação (poda progressiva) de atributos.
2. Comparar o ranking SHAP com o método clássico RFE (Recursive Feature Elimination).
3. Avaliar métricas (Acurácia, F1-Score, Tempo de Treinamento) ao longo da curva de redução de 40 até 2 atributos.
4. Plotar as curvas comparativas de decaimento de desempenho e eficiência computacional.
5. Validar via Sanity Check.
"""

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from modulo1_baseline import gerar_dataset_sintetico_saude

def obter_ranking_shap(X_train, y_train, random_state=42):
    """
    Retorna a lista de atributos ordenada do mais importante ao menos importante via SHAP.
    """
    modelo = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
    modelo.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer(X_train)
    
    if len(shap_values.shape) == 3:
        vals = shap_values.values[:, :, 1]
    else:
        vals = shap_values.values
        
    mean_abs_shap = np.abs(vals).mean(axis=0)
    df_rank = pd.DataFrame({"atributo": X_train.columns, "importancia": mean_abs_shap})
    df_rank = df_rank.sort_values(by="importancia", ascending=False).reset_index(drop=True)
    return df_rank["atributo"].tolist()

def obter_ranking_rfe(X_train, y_train, random_state=42):
    """
    Retorna a lista de atributos ordenada via RFE (Recursive Feature Elimination).
    """
    modelo_base = RandomForestClassifier(n_estimators=30, random_state=random_state, n_jobs=1)
    rfe = RFE(estimator=modelo_base, n_features_to_select=1, step=2)
    rfe.fit(X_train, y_train)
    
    df_rank = pd.DataFrame({"atributo": X_train.columns, "ranking": rfe.ranking_})
    df_rank = df_rank.sort_values(by="ranking", ascending=True).reset_index(drop=True)
    return df_rank["atributo"].tolist()

def executar_curva_ablacao(X_train, X_test, y_train, y_test, ordem_atributos, nome_metodo, passos=None):
    """
    Treina o modelo progressivamente mantendo do Top N até os 2 atributos principais.
    """
    if passos is None:
        passos = list(range(X_train.shape[1], 1, -2)) # Ex: 40, 38, 36, ..., 2
        
    resultados = []
    for k in passos:
        cols_subconjunto = ordem_atributos[:k]
        
        modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        
        t0 = time.perf_counter()
        modelo.fit(X_train[cols_subconjunto], y_train)
        t_treino = time.perf_counter() - t0
        
        y_pred = modelo.predict(X_test[cols_subconjunto])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        resultados.append({
            "metodo": nome_metodo,
            "n_atributos": k,
            "acuracia": acc,
            "f1_score": f1,
            "tempo_treino_ms": t_treino * 1000
        })
        
    return pd.DataFrame(resultados)

def gerar_graficos_ablacao(df_res_shap, df_res_rfe, output_path="modulo4_ablation_curves.png"):
    """
    Plota as curvas comparativas de F1-Score e Tempo de Treinamento em função do número de atributos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Curva de F1-Score vs Número de Atributos
    axes[0].plot(df_res_shap["n_atributos"], df_res_shap["f1_score"], marker='o', color='#1f77b4', lw=2.5, label="Seleção SHAP (XAI)")
    axes[0].plot(df_res_rfe["n_atributos"], df_res_rfe["f1_score"], marker='s', linestyle='--', color='#ff7f0e', lw=2, label="Seleção RFE (Tradicional)")
    axes[0].axvline(x=10, color='red', linestyle=':', label='Limite Teórico (10 Informativos)')
    axes[0].invert_xaxis()
    axes[0].set_title("Desempenho (F1-Score) vs Nº de Atributos Mantidos", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Número de Atributos no Modelo (Poda <-)")
    axes[0].set_ylabel("F1-Score no Teste")
    axes[0].legend(loc="lower left")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Curva de Tempo de Treinamento vs Número de Atributos
    axes[1].plot(df_res_shap["n_atributos"], df_res_shap["tempo_treino_ms"], marker='o', color='#2ca02c', lw=2.5, label="Tempo Treino (SHAP Subset)")
    axes[1].invert_xaxis()
    axes[1].set_title("Economia de Tempo de Treino vs Redução de Atributos", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Número de Atributos no Modelo (Poda <-)")
    axes[1].set_ylabel("Tempo de Treinamento (ms)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfico de Ablação salvo com sucesso em: {output_path}")

def teste_de_sanidade_ablacao(df_res_shap):
    """
    Sanity Check do Módulo 4:
    - Garante que a seleção SHAP mantém alto F1-score (>0.80) mesmo ao reduzir o dataset para apenas 10 atributos (75% de redução).
    """
    res_10_attrs = df_res_shap[df_res_shap["n_atributos"] == 10]
    assert len(res_10_attrs) > 0, "Erro: Medição de 10 atributos não encontrada"
    f1_10 = res_10_attrs["f1_score"].values[0]
    
    assert f1_10 >= 0.80, f"Erro: F1-Score desabou com 10 atributos SHAP: {f1_10:.4f}"
    print(f"\n[OK] SANITY CHECK PASSOU: Com apenas 10 atributos selecionados pelo SHAP (75% de redução), o F1-Score se manteve alto ({f1_10:.4f})!")

def executar_modulo_4():
    print("=" * 60)
    print("MÓDULO 4: PODA GUIADA POR XAI VS. SELEÇÃO TRADICIONAL")
    print("=" * 60)
    
    # 1. Carregar dados
    X, y = gerar_dataset_sintetico_saude()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 2. Obter Rankings
    print("[*] Calculando ranking de importância SHAP...")
    ranking_shap = obter_ranking_shap(X_train, y_train)
    
    print("[*] Calculando ranking de importância RFE...")
    ranking_rfe = obter_ranking_rfe(X_train, y_train)
    
    # 3. Executar experimentos de ablação
    print("[*] Executando curvas de ablação (poda de 40 a 2 atributos)...")
    df_res_shap = executar_curva_ablacao(X_train, X_test, y_train, y_test, ranking_shap, "SHAP")
    df_res_rfe = executar_curva_ablacao(X_train, X_test, y_train, y_test, ranking_rfe, "RFE")
    
    print("\n--- COMPARATIVO DE DESEMPENHO EM PONTOS CHAVE DE REDUÇÃO ---")
    for n in [40, 20, 10, 4]:
        row_s = df_res_shap[df_res_shap["n_atributos"] == n].iloc[0]
        row_r = df_res_rfe[df_res_rfe["n_atributos"] == n].iloc[0]
        print(f" Atributos: {n:2d} | SHAP F1: {row_s['f1_score']:.4f} (Tempo: {row_s['tempo_treino_ms']:.1f}ms) | RFE F1: {row_r['f1_score']:.4f}")
        
    # 4. Teste de Sanidade
    teste_de_sanidade_ablacao(df_res_shap)
    
    # 5. Gerar Gráficos
    gerar_graficos_ablacao(df_res_shap, df_res_rfe)
    
    return df_res_shap, df_res_rfe

if __name__ == "__main__":
    executar_modulo_4()
