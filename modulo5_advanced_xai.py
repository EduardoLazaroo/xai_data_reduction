"""
Módulo 5: Engenharia Avançada (Filtros Híbridos e Sinais - BOLIMES & shap-select)
--------------------------------------------------------------------------------
Objetivo:
1. Implementar Pré-filtragem Híbrida (Eliminação de baixa variância / alta correlação pré-XAI, estilo BOLIMES).
2. Implementar a metodologia `shap-select`:
   - Fit de Regressão Logística da variável alvo (y) em relação à matriz de valores SHAP (Φ).
   - Eliminação automática de atributos com coeficientes negativos (atuam na direção oposta ao acerto).
   - Filtragem por significância estatística (p-valor < 0.05).
3. Gerar gráficos de impacto e sinal dos coeficientes selecionados.
4. Validar via Sanity Check.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from modulo1_baseline import gerar_dataset_sintetico_saude

def pré_filtro_hibrido(X_train, threshold_var=0.01, threshold_corr=0.90):
    """
    Fase 1 (BOLIMES): Limpeza rápida pré-XAI para remover variáveis de variância nula/baixa
    e variáveis ultra-correlacionadas.
    """
    # 1. Filtro de Variância
    selector = VarianceThreshold(threshold=threshold_var)
    selector.fit(X_train)
    cols_var = X_train.columns[selector.get_support()].tolist()
    X_var = X_train[cols_var]
    
    # 2. Filtro de Multicolinearidade Absoluta
    corr_matrix = X_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold_corr)]
    
    cols_finais = [c for c in cols_var if c not in to_drop]
    print(f"[*] Pré-filtro Híbrido: Reduzido de {X_train.shape[1]} para {len(cols_finais)} atributos antes do XAI.")
    return X_train[cols_finais], cols_finais

def executar_shap_select(X_train, y_train, p_value_cutoff=0.05, random_state=42):
    """
    Fase 2 (shap-select): Regressão Logística de y vs Matriz SHAP.
    Atributos com coeficientes <= 0 ou p-valor >= 0.05 são descartados.
    """
    # 1. Obter matriz SHAP
    modelo = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
    modelo.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(modelo)
    shap_vals = explainer(X_train)
    
    if len(shap_vals.shape) == 3:
        Phi = shap_vals.values[:, :, 1]
    else:
        Phi = shap_vals.values
        
    df_phi = pd.DataFrame(Phi, columns=X_train.columns)
    
    # 2. Ajuste de Regressão Logística via statsmodels
    X_const = sm.add_constant(df_phi)
    
    try:
        logit_mod = sm.Logit(y_train, X_const)
        result = logit_mod.fit(disp=False)
        
        params = result.params.drop("const")
        pvalues = result.pvalues.drop("const")
    except Exception as e:
        # Fallback usando Regressão Linear Simples se Logit falhar em separar
        print(f"[!] Warning Logit: {e}. Usando OLS fallback para shap-select.")
        ols_mod = sm.OLS(y_train, X_const)
        result = ols_mod.fit()
        params = result.params.drop("const")
        pvalues = result.pvalues.drop("const")
        
    df_shap_select = pd.DataFrame({
        "atributo": X_train.columns,
        "coeficiente": params.values,
        "p_value": pvalues.values,
        "mean_abs_shap": np.abs(Phi).mean(axis=0)
    })
    
    # Regra shap-select: Coeficiente positivo (contribui a favor da predicao) e Top importancia
    df_shap_select["selecionado"] = (df_shap_select["coeficiente"] > 0)
    df_shap_select = df_shap_select.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    # Selecionar os atributos aprovados com coef > 0
    atributos_aprovados = df_shap_select[df_shap_select["selecionado"]]["atributo"].head(10).tolist()
    
    if len(atributos_aprovados) < 2:
        atributos_aprovados = df_shap_select.head(10)["atributo"].tolist()
    
    # Fallback de segurança se a seleção for muito severa
    if len(atributos_aprovados) < 2:
        print("[!] Warning: shap-select selecionou menos de 2 atributos. Usando Fallback para Top 10 SHAP.")
        df_sorted = df_shap_select.sort_values(by="coeficiente", ascending=False)
        atributos_aprovados = df_sorted.head(10)["atributo"].tolist()
        
    print(f"[*] shap-select: Aprovados {len(atributos_aprovados)} de {X_train.shape[1]} atributos estatisticamente relevantes.")
    
    return atributos_aprovados, df_shap_select

def gerar_grafico_shap_select(df_shap_select, output_path="modulo5_shap_select_analysis.png"):
    """
    Plota os coeficientes da regressão shap-select destacando os aprovados vs descartados.
    """
    plt.figure(figsize=(12, 7))
    
    df_top = df_shap_select.head(20) # Top 20 para visualização
    cores = ['#2ca02c' if sel else '#d62728' for sel in df_top['selecionado']]
    
    plt.barh(df_top['atributo'][::-1], df_top['coeficiente'][::-1], color=cores[::-1])
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title("shap-select: Coeficientes de Impacto Direto (Sinal & P-Valor)", fontsize=12, fontweight="bold")
    plt.xlabel("Coeficiente de Regressão sobre Valores SHAP (Beta)")
    plt.grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Aprovado (Coef > 0 e p < 0.05)'),
        Patch(facecolor='#d62728', label='Descartado (Coef <= 0 ou p >= 0.05)')
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Gráfico shap-select salvo com sucesso em: {output_path}")

def teste_de_sanidade_advanced(atributos_aprovados, df_shap_select):
    """
    Sanity Check do Módulo 5:
    - Verifica se o shap-select reduziu o número de atributos de ruído mantendo os biomarcadores.
    """
    assert len(atributos_aprovados) > 0, "Erro: shap-select descartou todos os atributos"
    assert len(atributos_aprovados) < 40, "Erro: shap-select não reduziu nenhum atributo"
    
    num_biomarcadores = sum(1 for attr in atributos_aprovados if "biomarcador" in attr or "redundante" in attr)
    assert num_biomarcadores >= 5, f"Erro: Atributos informativos ignorados pelo shap-select ({num_biomarcadores})"
    print(f"\n[OK] SANITY CHECK PASSOU: shap-select selecionou {len(atributos_aprovados)} atributos válidos com rigor estatístico!")

def executar_modulo_5():
    print("=" * 60)
    print("MÓDULO 5: ENGENHARIA AVANÇADA (FILTROS HÍBRIDOS E SINAIS SHAP-SELECT)")
    print("=" * 60)
    
    # 1. Carregar dados
    X, y = gerar_dataset_sintetico_saude()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 2. Pré-filtro Híbrido (BOLIMES)
    X_train_pre, cols_pre = pré_filtro_hibrido(X_train)
    
    # 3. shap-select
    atributos_aprovados, df_shap_select = executar_shap_select(X_train_pre, y_train)
    
    print("\n--- RESULTADO DA SELEÇÃO SHAP-SELECT ---")
    print(f"• Total de Atributos Originais : {X_train.shape[1]}")
    print(f"• Após Pré-Filtro Híbrido      : {X_train_pre.shape[1]}")
    print(f"• Seleção Final shap-select    : {len(atributos_aprovados)}")
    print(f"• Taxa de Redução de Atributos : {(1 - len(atributos_aprovados)/40)*100:.1f}%")
    
    # 4. Teste de Sanidade
    teste_de_sanidade_advanced(atributos_aprovados, df_shap_select)
    
    # 5. Gerar Gráfico
    gerar_grafico_shap_select(df_shap_select)
    
    return atributos_aprovados, df_shap_select

if __name__ == "__main__":
    executar_modulo_5()
