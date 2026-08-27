"""
Módulo 6: O Grande Final (Pipeline Integrado, Otimização com Optuna e Dashboard Comparativo)
-----------------------------------------------------------------------------------------
Objetivo:
1. Consolidar todas as etapas da pesquisa em um pipeline industrial limpo e modular.
2. Executar do início ao fim:
   - Treino e avaliação do modelo Baseline (40 atributos).
   - Filtragem XAI Avançada (Pré-filtro Híbrido + shap-select).
   - Re-otimização de hiperparâmetros no subconjunto reduzido via OPTUNA.
   - Treino e avaliação do modelo Reduzido Otimizado.
3. Gerar o Dashboard Comparativo Final (Tempo de Treino, Latência de Inferência, Nº de Atributos e F1-Score).
4. Validar os resultados do pipeline completo através do Sanity Check final.
"""

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

from modulo1_baseline import gerar_dataset_sintetico_saude, treinar_e_avaliar_baseline
from modulo5_advanced_xai import pré_filtro_hibrido, executar_shap_select

def otimizar_hiperparametros_optuna(X_train_reduced, y_train, n_trials=15, random_state=42):
    """
    Otimiza os hiperparâmetros do RandomForest no espaço reduzido de atributos usando Optuna.
    """
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 30, 200)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=1
        )
        
        scores = cross_val_score(clf, X_train_reduced, y_train, cv=3, scoring="f1", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print(f"[*] Optuna finalizado! Melhor F1 de Validação Cruzada: {study.best_value:.4f}")
    print(f"    Melhores Hiperparâmetros: {study.best_params}")
    return study.best_params

def treinar_e_avaliar_modelo(modelo, X_train, X_test, y_train, y_test):
    """
    Função utilitária para treinar e extrair métricas completas de um modelo.
    """
    t0_treino = time.perf_counter()
    modelo.fit(X_train, y_train)
    tempo_treino = time.perf_counter() - t0_treino
    
    t0_inf = time.perf_counter()
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    tempo_inf = time.perf_counter() - t0_inf
    
    return {
        "acuracia": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tempo_treino_ms": tempo_treino * 1000,
        "tempo_inferencia_ms": tempo_inf * 1000,
        "num_atributos": X_train.shape[1]
    }

def gerar_dashboard_comparativo_final(m_baseline, m_reduzido, output_path="dashboard_final_comparativo.png"):
    """
    Plota o Dashboard Comparativo Final de 4 Painéis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DASHBOARD FINAL: PIPELINE DE REDUÇÃO DE DADOS BASEADO EM XAI", fontsize=14, fontweight="bold")
    
    categorias = ["Baseline (Completo)", "XAI Reduzido + Optuna"]
    cores = ["#d62728", "#2ca02c"]
    
    # 1. Número de Atributos
    attrs = [m_baseline["num_atributos"], m_reduzido["num_atributos"]]
    axes[0, 0].bar(categorias, attrs, color=cores, width=0.5)
    axes[0, 0].set_title("1. Dimensão do Dataset (Nº de Atributos)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Quantidade de Atributos")
    for i, v in enumerate(attrs):
        axes[0, 0].text(i, v / 2, str(v), ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Tempo de Treinamento
    tempos_t = [m_baseline["tempo_treino_ms"], m_reduzido["tempo_treino_ms"]]
    axes[0, 1].bar(categorias, tempos_t, color=cores, width=0.5)
    axes[0, 1].set_title("2. Tempo de Treinamento (ms)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Milissegundos (ms)")
    for i, v in enumerate(tempos_t):
        axes[0, 1].text(i, v / 2, f"{v:.1f} ms", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Latência de Inferência
    tempos_i = [m_baseline["tempo_inferencia_ms"], m_reduzido["tempo_inferencia_ms"]]
    axes[1, 0].bar(categorias, tempos_i, color=cores, width=0.5)
    axes[1, 0].set_title("3. Latência de Inferência (ms)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Milissegundos (ms)")
    for i, v in enumerate(tempos_i):
        axes[1, 0].text(i, v / 2, f"{v:.1f} ms", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Desempenho Preditivo (Acurácia & F1-Score)
    x = np.arange(len(categorias))
    w = 0.35
    axes[1, 1].bar(x - w/2, [m_baseline["acuracia"], m_reduzido["acuracia"]], w, label="Acurácia", color="#1f77b4")
    axes[1, 1].bar(x + w/2, [m_baseline["f1_score"], m_reduzido["f1_score"]], w, label="F1-Score", color="#ff7f0e")
    axes[1, 1].set_title("4. Desempenho Preditivo (Acurácia vs F1)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categorias)
    axes[1, 1].set_ylim(0.7, 1.0)
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Dashboard Comparativo Final salvo com sucesso em: {output_path}")

def teste_de_sanidade_final(m_baseline, m_reduzido):
    """
    Sanity Check Final do Módulo 6:
    - Valida se o modelo reduzido eliminou ao menos 50% dos atributos.
    - Valida se o F1-Score do modelo reduzido permaneceu dentro de uma margem aceitável em relação ao baseline.
    - Valida se houve redução no tempo de treinamento.
    """
    reducao_atributos = (1 - m_reduzido["num_atributos"] / m_baseline["num_atributos"]) * 100
    diferenca_f1 = m_reduzido["f1_score"] - m_baseline["f1_score"]
    
    assert reducao_atributos >= 40.0, f"Erro: Redução de atributos insuficiente ({reducao_atributos:.1f}%)"
    assert m_reduzido["f1_score"] >= (m_baseline["f1_score"] - 0.05), f"Erro: Perda excessiva de F1-Score ({diferenca_f1:.4f})"
    print(f"\n[OK] SANITY CHECK FINAL PASSOU:")
    print(f"     + Reducao de Dimensao    : {reducao_atributos:.1f}% dos atributos eliminados!")
    print(f"     + Manutencao do F1-Score : Baseline={m_baseline['f1_score']:.4f} vs Reduzido={m_reduzido['f1_score']:.4f}")
    print(f"     + Ganho em Vel. Treino   : {(1 - m_reduzido['tempo_treino_ms']/m_baseline['tempo_treino_ms'])*100:.1f}% mais rapido!")

def executar_pipeline_completo():
    print("=" * 70)
    print("MÓDULO 6: PIPELINE COMPLETO DE REDUÇÃO DE DADOS VIA XAI E OPTUNA")
    print("=" * 70)
    
    # 1. Carregar dados
    X, y = gerar_dataset_sintetico_saude()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 2. Treinar e Avaliar Baseline
    print("\n[ETAPA 1/4] Treinando Modelo Baseline (40 Atributos)...")
    modelo_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    m_baseline = treinar_e_avaliar_modelo(modelo_base, X_train, X_test, y_train, y_test)
    
    # 3. Filtragem XAI Avançada
    print("\n[ETAPA 2/4] Executando Filtragem Híbrida + Seleção shap-select...")
    X_train_pre, cols_pre = pré_filtro_hibrido(X_train)
    atributos_selecionados, df_shap_select = executar_shap_select(X_train_pre, y_train)
    
    X_train_red = X_train[atributos_selecionados]
    X_test_red = X_test[atributos_selecionados]
    
    # 4. Re-otimização com Optuna
    print(f"\n[ETAPA 3/4] Re-otimizando Hiperparâmetros via Optuna no espaço de {len(atributos_selecionados)} atributos...")
    best_params = otimizar_hiperparametros_optuna(X_train_red, y_train, n_trials=15)
    
    # 5. Treinar Modelo Reduzido Otimizado
    print("\n[ETAPA 4/4] Treinando Modelo Reduzido Otimizado...")
    modelo_red_otimizado = RandomForestClassifier(**best_params, random_state=42, n_jobs=1)
    m_reduzido = treinar_e_avaliar_modelo(modelo_red_otimizado, X_train_red, X_test_red, y_train, y_test)
    
    # Exibir Comparativo no Terminal
    print("\n" + "=" * 70)
    print("RELATÓRIO COMPARATIVO FINAL: BASELINE vs PIPELINE XAI")
    print("=" * 70)
    print(f" Atributo                 | Baseline (Bruto) | XAI Reduzido + Optuna")
    print(f"--------------------------+------------------+----------------------")
    print(f" Nº de Atributos          | {m_baseline['num_atributos']:<16} | {m_reduzido['num_atributos']:<20}")
    print(f" Acurácia                 | {m_baseline['acuracia']:<16.4f} | {m_reduzido['acuracia']:<20.4f}")
    print(f" F1-Score                 | {m_baseline['f1_score']:<16.4f} | {m_reduzido['f1_score']:<20.4f}")
    print(f" ROC-AUC                  | {m_baseline['roc_auc']:<16.4f} | {m_reduzido['roc_auc']:<20.4f}")
    print(f" Tempo de Treino (ms)     | {m_baseline['tempo_treino_ms']:<16.2f} | {m_reduzido['tempo_treino_ms']:<20.2f}")
    print(f" Tempo de Inferência (ms) | {m_baseline['tempo_inferencia_ms']:<16.2f} | {m_reduzido['tempo_inferencia_ms']:<20.2f}")
    print("=" * 70)
    
    # 6. Sanity Check Final
    teste_de_sanidade_final(m_baseline, m_reduzido)
    
    # 7. Gerar Dashboard
    gerar_dashboard_comparativo_final(m_baseline, m_reduzido)
    
    return m_baseline, m_reduzido

if __name__ == "__main__":
    executar_pipeline_completo()
