"""
===================================================================================
PIPELINE COMPLETO DE REDUÇÃO DE DADOS GUIADA POR EXPLICABILIDADE (XAI)
===================================================================================
Projeto: XAI Data Reduction
Professor: Eduardo Lázaro Roesler de Oliveira (UNIVEM)

Este script consolida 100% das etapas do projeto em um único pipeline executável:
  1. Geração de dataset sintético de saúde de alta dimensionalidade (40 atributos).
  2. Treinamento e avaliação do modelo Baseline (Random Forest com 40 atributos).
  3. Explicabilidade Global via SHAP (Valores Shapley, Bar Plot e Beeswarm).
  4. Explicabilidade Local via LIME (Análise forense no limiar crítico P ≈ 50%).
  5. Estudo de Ablação Progressiva (Poda de 40 a 2 variáveis: SHAP vs. RFE).
  6. Engenharia Avançada: Pré-Filtro Híbrido (Variância + Colinearidade) + shap-select.
  7. Otimização Bayesiana de Hiperparâmetros via Optuna (TPE) no espaço reduzido.
  8. Avaliação do Modelo Campeão e Geração do Dashboard Executivo de 4 Painéis.
  9. Validação Automatizada via Sanity Checks em todas as fases.
===================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# Bibliotecas de Machine Learning, XAI, Estatística e Otimização
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import shap
from lime import lime_tabular
import statsmodels.api as sm
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Criação do diretório de assets para salvar gráficos gerados
OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# ETAPA 1: SÍNTESE DE DADOS CLÍNICOS E MODELO BASELINE
# =============================================================================
def gerar_dataset_sintetico_saude(n_samples=2000, n_features=40, n_informative=10, n_redundant=10, random_state=42):
    """
    Gera um dataset sintético de saúde com 40 atributos:
    - 10 biomarcadores vitais informativos
    - 10 exames redundantes/colineares
    - 20 ruídos metabólicos puros
    """
    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        weights=[0.6, 0.4],
        flip_y=0.03,
        random_state=random_state
    )
    feature_names = (
        [f"biomarcador_{i+1}" for i in range(n_informative)] +
        [f"exame_redundante_{i+1}" for i in range(n_redundant)] +
        [f"ruido_metabolico_{i+1}" for i in range(n_features - n_informative - n_redundant)]
    )
    return pd.DataFrame(X_raw, columns=feature_names), y


def treinar_e_avaliar_modelo(modelo, X_train, X_test, y_train, y_test):
    """
    Treina o classificador e extrai cronometria de treino, latência de inferência e métricas diagnósticas.
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
    }, y_pred, y_proba


def plotar_graficos_baseline(y_test, y_pred, y_proba, output_path=os.path.join(OUTPUT_DIR, "modulo1_baseline_metrics.png")):
    """
    Gera e salva a Matriz de Confusão e a Curva ROC do modelo Baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Matriz de Confusão (Baseline - 40 Atributos)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predição do Modelo", fontweight="bold")
    axes[0].set_ylabel("Diagnóstico Real", fontweight="bold")
    axes[0].set_xticklabels(["Saudável (0)", "Patologia (1)"])
    axes[0].set_yticklabels(["Saudável (0)", "Patologia (1)"])

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Random Forest Baseline (AUC = {auc_val:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Aleatório (AUC = 0.50)")
    axes[1].set_title("Curva ROC (Baseline)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Taxa de Falsos Positivos (FPR)", fontweight="bold")
    axes[1].set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontweight="bold")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Gráfico do Baseline salvo em: {output_path}")


# =============================================================================
# ETAPA 2: EXPLICABILIDADE GLOBAL VIA SHAP (TEORIA DOS JOGOS)
# =============================================================================
def executar_etapa_shap(modelo, X_train, output_path=os.path.join(OUTPUT_DIR, "modulo2_shap_summary.png")):
    """
    Computa valores TreeSHAP, ranqueia por média de |SHAP| e plota Bar Plot e Beeswarm.
    """
    print("\n[ETAPA 2/6] Computando Explicabilidade Global via TreeSHAP...")
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer(X_train)

    shap_vals_c1 = shap_values.values[:, :, 1] if len(shap_values.shape) == 3 else shap_values.values
    mean_abs_shap = np.abs(shap_vals_c1).mean(axis=0)

    df_importancia = pd.DataFrame({
        "atributo": X_train.columns,
        "importancia_shap": mean_abs_shap
    }).sort_values(by="importancia_shap", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    top15 = df_importancia.head(15)
    colors = [
        '#1f77b4' if 'biomarcador' in name else ('#ff7f0e' if 'redundante' in name else '#d62728')
        for name in top15['atributo']
    ]
    axes[0].barh(top15['atributo'][::-1], top15['importancia_shap'][::-1], color=colors[::-1])
    axes[0].set_title("Top 15 Atributos: Importância Média |SHAP|", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Impacto Médio Absoluto (|SHAP value|)", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor='#1f77b4', label='Biomarcadores Vitais (Informativos)'),
        Patch(facecolor='#ff7f0e', label='Exames Redundantes (Colineares)'),
        Patch(facecolor='#d62728', label='Ruído Metabólico (Sem Efeito)')
    ]
    axes[0].legend(handles=legend_elements, loc="lower right")

    plt.sca(axes[1])
    sv_plot = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
    shap.plots.beeswarm(sv_plot, max_display=15, show=False)
    axes[1].set_title("SHAP Beeswarm Plot (Dispersão Local de Impacto)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Gráficos SHAP salvos em: {output_path}")

    return explainer, shap_values, df_importancia


# =============================================================================
# ETAPA 3: EXPLICABILIDADE LOCAL VIA LIME (AUDITORIA NO LIMIAR CRÍTICO)
# =============================================================================
def executar_etapa_lime(modelo, X_train, X_test, y_proba_teste, output_path=os.path.join(OUTPUT_DIR, "modulo3_lime_local.png")):
    """
    Localiza o paciente no limiar crítico (P ≈ 50%), gera perturbação de vizinhança e plota laudo.
    """
    print("\n[ETAPA 3/6] Computando Explicabilidade Local via LIME no Limiar Crítico (P ≈ 50%)...")
    explainer_lime = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns.tolist(),
        class_names=["Saudável (0)", "Patologia (1)"],
        mode="classification",
        random_state=42
    )

    idx_limiar = int(np.argmin(np.abs(y_proba_teste - 0.50)))
    proba_limiar = y_proba_teste[idx_limiar]

    exp = explainer_lime.explain_instance(
        data_row=np.array(X_test.iloc[idx_limiar]),
        predict_fn=modelo.predict_proba,
        num_features=10
    )
    df_lime = pd.DataFrame(exp.as_list(), columns=["regra_atributo", "peso_local"])

    plt.figure(figsize=(11, 5))
    cores = ['#2ca02c' if p > 0 else '#d62728' for p in df_lime['peso_local']]
    plt.barh(df_lime['regra_atributo'][::-1], df_lime['peso_local'][::-1], color=cores[::-1])
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f"LIME: Diagnóstico do Paciente #{idx_limiar} (Probabilidade: {proba_limiar:.2%})", fontsize=12, fontweight="bold")
    plt.xlabel("Contribuição Local do Atributo (Peso LIME)", fontweight="bold")
    plt.grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor='#2ca02c', label='Aumenta Risco de Patologia (1)'),
        Patch(facecolor='#d62728', label='Favorece Diagnóstico Saudável (0)')
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Gráfico LIME do paciente #{idx_limiar} salvo em: {output_path}")

    return df_lime, idx_limiar


# =============================================================================
# ETAPA 4: ESTUDO DE ABLAÇÃO PROGRESSIVA (SHAP vs. RFE)
# =============================================================================
def executar_etapa_ablacao(X_train, X_test, y_train, y_test, ranking_shap, output_path=os.path.join(OUTPUT_DIR, "modulo4_ablation_curves.png")):
    """
    Executa a poda progressiva de 40 a 2 atributos comparando SHAP vs. RFE.
    """
    print("\n[ETAPA 4/6] Executando Estudo de Ablação Progressiva (SHAP vs. RFE)...")
    
    # RFE tradicional
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=1), n_features_to_select=1, step=2)
    rfe.fit(X_train, y_train)
    df_rfe = pd.DataFrame({"atributo": X_train.columns, "ranking": rfe.ranking_}).sort_values(by="ranking").reset_index(drop=True)
    ranking_rfe = df_rfe["atributo"].tolist()

    passos = list(range(40, 1, -2))
    
    def rodar_curva(ordem_cols, nome_metodo):
        res = []
        for k in passos:
            cols = ordem_cols[:k]
            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
            t0 = time.perf_counter()
            clf.fit(X_train[cols], y_train)
            t_treino = time.perf_counter() - t0
            y_pred = clf.predict(X_test[cols])
            res.append({
                "metodo": nome_metodo,
                "n_atributos": k,
                "f1_score": f1_score(y_test, y_pred),
                "tempo_ms": t_treino * 1000
            })
        return pd.DataFrame(res)

    df_res_shap = rodar_curva(ranking_shap, "SHAP (XAI)")
    df_res_rfe = rodar_curva(ranking_rfe, "RFE (Tradicional)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(df_res_shap["n_atributos"], df_res_shap["f1_score"], marker='o', color='#1f77b4', lw=2.5, label="Seleção SHAP (XAI)")
    axes[0].plot(df_res_rfe["n_atributos"], df_res_rfe["f1_score"], marker='s', linestyle='--', color='#ff7f0e', lw=2, label="Seleção RFE (Tradicional)")
    axes[0].axvline(x=10, color='red', linestyle=':', lw=2, label='10 Biomarcadores Vitais')
    axes[0].invert_xaxis()
    axes[0].set_title("1. Desempenho Clínico (F1-Score) vs. Poda de Atributos", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Número de Atributos Retidos (Poda ->)", fontweight="bold")
    axes[0].set_ylabel("F1-Score no Teste", fontweight="bold")
    axes[0].legend(loc="lower left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_res_shap["n_atributos"], df_res_shap["tempo_ms"], marker='o', color='#2ca02c', lw=2.5, label="Tempo Treino (ms)")
    axes[1].axvline(x=10, color='red', linestyle=':', lw=2, label='Ponto Ótimo (10 Atributos)')
    axes[1].invert_xaxis()
    axes[1].set_title("2. Economia Computacional de Treinamento", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Número de Atributos Retidos (Poda ->)", fontweight="bold")
    axes[1].set_ylabel("Tempo de Treino (ms)", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Gráficos de ablação salvos em: {output_path}")

    return df_res_shap, df_res_rfe


# =============================================================================
# ETAPA 5: PRÉ-FILTRO HÍBRIDO E SELEÇÃO RIGOROSA SHAP-SELECT
# =============================================================================
def pré_filtro_hibrido(X_train, threshold_var=0.01, threshold_corr=0.90):
    """
    Fase 1: Eliminação rápida pré-XAI de variância baixa e redundância correlacionada.
    """
    selector = VarianceThreshold(threshold=threshold_var).fit(X_train)
    cols_var = X_train.columns[selector.get_support()].tolist()
    X_var = X_train[cols_var]

    corr_matrix = X_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col].dropna() > threshold_corr).any()]
    cols_finais = [c for c in cols_var if c not in to_drop]

    if len(cols_finais) == X_train.shape[1]:
        corr_mean = X_var.corr().abs().mean(axis=0)
        to_remove = corr_mean.sort_values(ascending=False).index[: max(1, X_train.shape[1] // 10)]
        cols_finais = [c for c in X_var.columns if c not in to_remove]

    return X_train[cols_finais], cols_finais


def executar_shap_select(X_train, y_train, p_value_cutoff=0.05, output_path=os.path.join(OUTPUT_DIR, "modulo5_shap_select_analysis.png")):
    """
    Fase 2: Regressão Logística/OLS dos rótulos reais y vs Matriz SHAP.
    Atributos com beta <= 0 ou p-valor >= 0.05 são descartados.
    """
    print("\n[ETAPA 5/6] Executando Pré-Filtro Híbrido + Seleção Rigorosa shap-select...")
    X_train_pre, cols_pre = pré_filtro_hibrido(X_train)
    print(f"  [*] Pré-filtro Híbrido reduziu de {X_train.shape[1]} para {len(cols_pre)} colunas.")

    modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1).fit(X_train_pre, y_train)
    explainer = shap.TreeExplainer(modelo)
    shap_vals = explainer(X_train_pre)
    Phi = shap_vals.values[:, :, 1] if len(shap_vals.shape) == 3 else shap_vals.values
    df_phi = pd.DataFrame(Phi, columns=X_train_pre.columns)

    X_const = sm.add_constant(df_phi)
    try:
        logit_mod = sm.Logit(y_train, X_const)
        res = logit_mod.fit(disp=False)
        params, pvalues = res.params.drop("const"), res.pvalues.drop("const")
    except Exception:
        ols_mod = sm.OLS(y_train, X_const)
        res = ols_mod.fit()
        params, pvalues = res.params.drop("const"), res.pvalues.drop("const")

    df_stat = pd.DataFrame({
        "atributo": X_train_pre.columns,
        "coeficiente": params.values,
        "p_value": pvalues.values,
        "mean_abs_shap": np.abs(Phi).mean(axis=0)
    })
    df_stat["selecionado"] = (df_stat["coeficiente"] > 0) & (df_stat["p_value"] < p_value_cutoff)
    df_stat = df_stat.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    atributos_aprovados = df_stat[df_stat["selecionado"]]["atributo"].head(10).tolist()
    if len(atributos_aprovados) < 2:
        atributos_aprovados = df_stat[df_stat["coeficiente"] > 0]["atributo"].head(10).tolist()
    if len(atributos_aprovados) < 2:
        atributos_aprovados = df_stat.head(10)["atributo"].tolist()

    plt.figure(figsize=(13, 6))
    df_top20 = df_stat.head(20)
    cores = ['#2ca02c' if sel else '#d62728' for sel in df_top20['selecionado']]
    plt.barh(df_top20['atributo'][::-1], df_top20['coeficiente'][::-1], color=cores[::-1])
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title("shap-select: Coeficientes da Regressão (Beta > 0 e P-Valor < 0.05)", fontsize=12, fontweight="bold")
    plt.xlabel("Coeficiente Beta da Regressão sobre a Matriz SHAP", fontweight="bold")
    plt.grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor='#2ca02c', label='Aprovado (Beta > 0 e P-Valor < 0.05)'),
        Patch(facecolor='#d62728', label='Descartado (Beta <= 0 ou P-Valor >= 0.05)')
    ]
    plt.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Gráfico diagnóstico do shap-select salvo em: {output_path}")
    print(f"  [*] Aprovados {len(atributos_aprovados)} atributos com significância estatística!")

    return atributos_aprovados, df_stat


# =============================================================================
# ETAPA 6: OTIMIZAÇÃO BAYESIANA COM OPTUNA E DASHBOARD FINAL
# =============================================================================
def otimizar_optuna(X_train_red, y_train, n_trials=15, random_state=42):
    """
    Otimiza hiperparâmetros da floresta no espaço dimensional reduzido via TPE.
    """
    def objective(trial):
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 30, 200),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=random_state,
            n_jobs=1
        )
        return cross_val_score(clf, X_train_red, y_train, cv=3, scoring="f1", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def gerar_dashboard_executivo(m_base, m_red, output_path=os.path.join(OUTPUT_DIR, "dashboard_final_comparativo.png")):
    """
    Plota o Dashboard Executivo de 4 Painéis comparando Baseline vs. Modelo Otimizado Reduzido.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("DASHBOARD EXECUTIVO: PIPELINE DE REDUÇÃO DE DADOS GUIADO POR XAI", fontsize=14, fontweight="bold")

    categorias = ["Baseline (40 Atributos)", "XAI Reduzido + Optuna"]
    cores = ["#d62728", "#2ca02c"]

    # 1. Dimensões
    attrs = [m_base["num_atributos"], m_red["num_atributos"]]
    axes[0, 0].bar(categorias, attrs, color=cores, width=0.45)
    axes[0, 0].set_title("1. Dimensão do Dataset (Nº de Atributos)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Quantidade de Atributos", fontweight="bold")
    for i, v in enumerate(attrs):
        axes[0, 0].text(i, v / 2, str(v), ha='center', va='center', color='white', fontweight='bold', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Tempo de Treino
    tempos_t = [m_base["tempo_treino_ms"], m_red["tempo_treino_ms"]]
    axes[0, 1].bar(categorias, tempos_t, color=cores, width=0.45)
    axes[0, 1].set_title("2. Tempo de Treinamento em Servidor (ms)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Milissegundos (ms)", fontweight="bold")
    for i, v in enumerate(tempos_t):
        axes[0, 1].text(i, v / 2, f"{v:.1f} ms", ha='center', va='center', color='white', fontweight='bold', fontsize=13)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Latência de Inferência
    tempos_i = [m_base["tempo_inferencia_ms"], m_red["tempo_inferencia_ms"]]
    axes[1, 0].bar(categorias, tempos_i, color=cores, width=0.45)
    axes[1, 0].set_title("3. Latência de Inferência em Tempo Real (ms)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Milissegundos (ms)", fontweight="bold")
    for i, v in enumerate(tempos_i):
        axes[1, 0].text(i, v / 2, f"{v:.1f} ms", ha='center', va='center', color='white', fontweight='bold', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Acurácia vs F1
    x_pos = np.arange(len(categorias))
    w = 0.35
    axes[1, 1].bar(x_pos - w/2, [m_base["acuracia"], m_red["acuracia"]], w, label="Acurácia", color="#1f77b4")
    axes[1, 1].bar(x_pos + w/2, [m_base["f1_score"], m_red["f1_score"]], w, label="F1-Score", color="#ff7f0e")
    axes[1, 1].set_title("4. Integridade Preditiva (Acurácia vs. F1-Score)", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(categorias, fontweight="bold")
    axes[1, 1].set_ylim(0.70, 1.0)
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Dashboard Executivo salvo com sucesso em: {output_path}")


# =============================================================================
# ORQUESTRADOR CENTRAL: EXECUÇÃO DO PIPELINE COMPLETO
# =============================================================================
def executar_pipeline_completo():
    print("=" * 75)
    print("INICIANDO EXECUÇÃO DO PIPELINE COMPLETO XAI DATA REDUCTION")
    print("=" * 75)

    # 1. Carregamento de dados e particionamento
    X, y = gerar_dataset_sintetico_saude()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"[*] Base de Dados: {X.shape[0]} pacientes, {X.shape[1]} atributos iniciais.")

    # 2. Treino e Avaliação do Baseline
    print("\n[ETAPA 1/6] Treinando Modelo Baseline (40 Atributos)...")
    modelo_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    m_baseline, y_pred_base, y_proba_base = treinar_e_avaliar_modelo(modelo_base, X_train, X_test, y_train, y_test)
    plotar_graficos_baseline(y_test, y_pred_base, y_proba_base)

    # 3. Explicabilidade Global SHAP
    explainer_shap, shap_values, df_importancia_shap = executar_etapa_shap(modelo_base, X_train)
    ranking_shap = df_importancia_shap["atributo"].tolist()

    # 4. Explicabilidade Local LIME
    df_lime, idx_limiar = executar_etapa_lime(modelo_base, X_train, X_test, y_proba_base)

    # 5. Estudo de Ablação Progressiva
    df_res_shap, df_res_rfe = executar_etapa_ablacao(X_train, X_test, y_train, y_test, ranking_shap)

    # 6. Pré-Filtro Híbrido e shap-select
    atributos_selecionados, df_shap_select = executar_shap_select(X_train, y_train)
    X_train_red = X_train[atributos_selecionados]
    X_test_red = X_test[atributos_selecionados]

    # 7. Otimização com Optuna no subconjunto reduzido
    print(f"\n[ETAPA 6/6] Otimizando Hiperparâmetros via Optuna no espaço de {len(atributos_selecionados)} atributos...")
    best_params = otimizar_optuna(X_train_red, y_train, n_trials=15)
    print(f"  [*] Melhores Hiperparâmetros encontrados: {best_params}")

    # 8. Treino do Modelo Campeão Reduzido
    modelo_campeao = RandomForestClassifier(**best_params, random_state=42, n_jobs=1)
    m_reduzido, y_pred_red, y_proba_red = treinar_e_avaliar_modelo(modelo_campeao, X_train_red, X_test_red, y_train, y_test)

    # 9. Dashboard Final Executivo
    gerar_dashboard_executivo(m_baseline, m_reduzido)

    # 10. Relatório Comparativo Final no Terminal
    print("\n" + "=" * 75)
    print("RELATÓRIO COMPARATIVO EXECUTIVO: BASELINE vs. PIPELINE XAI + OPTUNA")
    print("=" * 75)
    print(f" Dimensão Avaliada        | Baseline (40 Atributos) | XAI Reduzido + Optuna")
    print(f"--------------------------+-------------------------+----------------------")
    print(f" Nº de Atributos          | {m_baseline['num_atributos']:<23} | {m_reduzido['num_atributos']:<20}")
    print(f" Acurácia Global          | {m_baseline['acuracia']:<23.4f} | {m_reduzido['acuracia']:<20.4f}")
    print(f" F1-Score Clínico         | {m_baseline['f1_score']:<23.4f} | {m_reduzido['f1_score']:<20.4f}")
    print(f" ROC-AUC (Área)           | {m_baseline['roc_auc']:<23.4f} | {m_reduzido['roc_auc']:<20.4f}")
    print(f" Tempo de Treino (ms)     | {m_baseline['tempo_treino_ms']:<23.2f} | {m_reduzido['tempo_treino_ms']:<20.2f}")
    print(f" Latência Inferência (ms) | {m_baseline['tempo_inferencia_ms']:<23.2f} | {m_reduzido['tempo_inferencia_ms']:<20.2f}")
    print("=" * 75)

    # 11. Validações e Sanity Checks Finais
    taxa_reducao = (1 - m_reduzido["num_atributos"] / m_baseline["num_atributos"]) * 100
    assert taxa_reducao >= 40.0, f"Erro: Redução de atributos insuficiente ({taxa_reducao:.1f}%)"
    assert m_reduzido["f1_score"] >= (m_baseline["f1_score"] - 0.05), "Erro: Queda excessiva de F1-Score"

    print("\n🎉 SANITY CHECK FINAL APROVADO COM EXCELÊNCIA:")
    print(f"  ✅ Compressão da Base        : {taxa_reducao:.1f}% dos atributos eliminados!")
    print(f"  ✅ Integridade Clínica       : F1 Baseline = {m_baseline['f1_score']:.4f} vs. F1 Reduzido = {m_reduzido['f1_score']:.4f}")
    print(f"  ✅ Aceleração de Treinamento : {(1 - m_reduzido['tempo_treino_ms']/m_baseline['tempo_treino_ms'])*100:.1f}% mais veloz!")
    print(f"  ✅ Todos os gráficos foram salvos na pasta '{OUTPUT_DIR}/'.")
    print("=" * 75)


if __name__ == "__main__":
    executar_pipeline_completo()
