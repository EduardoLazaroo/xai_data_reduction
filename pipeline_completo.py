"""
===================================================================================
PIPELINE COMPLETO DE REDUÇÃO DE DADOS GUIADA POR EXPLICABILIDADE (XAI)
===================================================================================
Projeto: XAI Data Reduction
Instituição: UNIVEM
Autor / Pesquisa: Eduardo Lázaro Roesler de Oliveira

Este script consolida 100% das etapas experimentais atualizadas:
  1. Geração de dataset sintético de saúde de alta dimensionalidade (40 atributos:
     10 biomarcadores vitais, 10 exames redundantes/colineares, 20 ruídos metabólicos).
  2. Treinamento e avaliação do modelo Baseline (Random Forest com 40 atributos).
  3. Explicabilidade Global via TreeSHAP (Valores Shapley, Bar Plot e Beeswarm).
  4. Comparativo Sistemático de Seleção de Características em 4 Famílias:
     - Filter: Mutual Information e ANOVA F-score
     - Wrapper: Recursive Feature Elimination (RFE)
     - Embedded: Regressão Logística Lasso L1 e Random Forest MDI (Gini)
     - XAI: TreeSHAP Ranking e Pré-Filtro Híbrido + shap-select
  5. Estudo de Ablação Progressiva Multivariada (40 a 2 atributos) em todas as famílias.
  6. Engenharia Avançada: Pré-Filtro Híbrido (Variância + Colinearidade) + shap-select (OLS/Logit com p-valor).
  7. Otimização Bayesiana de Hiperparâmetros via Optuna (TPE) no espaço dimensional reduzido.
  8. Avaliação Forense e Auditoria Local via LIME no Limiar Crítico (P ≈ 50%):
     - Comparação direta da Fidelidade Local (Surrogate R²) entre Baseline e Modelo Reduzido.
  9. Avaliação Multidimensional do Ganho de Interpretabilidade:
     - Concentração de Massa de Shapley no Top-10
     - Ganho de Fidelidade Local LIME (R²)
     - Carga Cognitiva Clínica (Lei de Miller: 7 ± 2 atributos)
     - Taxa de Rejeição de Ruído Puro (Noise Elimination Rate)
  10. Dashboard Executivo e Exportação da Tabela Comparativa Mestre (CSV).
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
from sklearn.feature_selection import VarianceThreshold, RFE, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Diretório de saída para gráficos e tabelas
OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# ETAPA 1: SÍNTESE DE DADOS CLÍNICOS E MODELO BASELINE
# =============================================================================
def gerar_dataset_sintetico_saude(n_samples=2000, n_features=40, n_informative=10, n_redundant=10, random_state=42):
    """
    Gera um dataset sintético controlado com atributos informativos, redundantes e ruído.
    Os nomes representam papéis preditivos simulados, não variáveis clínicas causais.
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
        shuffle=False,
        random_state=random_state
    )
    feature_names = (
        [f"biomarcador_{i+1}" for i in range(n_informative)] +
        [f"exame_redundante_{i+1}" for i in range(n_redundant)] +
        [f"ruido_metabolico_{i+1}" for i in range(n_features - n_informative - n_redundant)]
    )
    df = pd.DataFrame(X_raw, columns=feature_names)
    return df, y


def treinar_e_avaliar_modelo(modelo, X_train, X_test, y_train, y_test):
    """
    Treina o estimador e extrai cronometria de treino (ms), latência de inferência (ms)
    e todas as métricas diagnósticas padrão ouro.
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
    Gera e salva Matriz de Confusão e Curva ROC do Modelo Baseline.
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
# ETAPA 2: EXPLICABILIDADE GLOBAL VIA TREESHAP (TEORIA DOS JOGOS)
# =============================================================================
def executar_etapa_shap(modelo, X_train, output_path=os.path.join(OUTPUT_DIR, "modulo2_shap_summary.png")):
    """
    Computa valores TreeSHAP, ranqueia por média de |SHAP| e gera Bar Plot e Beeswarm.
    """
    print("\n[ETAPA 2/7] Computando Explicabilidade Global via TreeSHAP...")
    t0 = time.perf_counter()
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer(X_train)
    tempo_shap = time.perf_counter() - t0

    shap_vals_c1 = shap_values.values[:, :, 1] if len(shap_values.shape) == 3 else shap_values.values
    mean_abs_shap = np.abs(shap_vals_c1).mean(axis=0)

    df_importancia = pd.DataFrame({
        "atributo": X_train.columns,
        "importancia_shap": mean_abs_shap
    }).sort_values(by="importancia_shap", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(25, 8.5), gridspec_kw={'width_ratios': [1, 1.25], 'wspace': 0.38})
    top15 = df_importancia.head(15)
    colors = [
        '#1f77b4' if 'biomarcador' in name else ('#ff7f0e' if 'redundante' in name else '#d62728')
        for name in top15['atributo']
    ]
    axes[0].barh(top15['atributo'][::-1], top15['importancia_shap'][::-1], color=colors[::-1], height=0.7)
    axes[0].set_title("Top 15 Atributos: Importância Média |SHAP|", fontsize=14, fontweight="bold", pad=15)
    axes[0].set_xlabel("Impacto Médio Absoluto (|SHAP value|)", fontweight="bold", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=11)

    legend_elements = [
        Patch(facecolor='#1f77b4', label='Biomarcadores Vitais (Informativos)'),
        Patch(facecolor='#ff7f0e', label='Exames Redundantes (Colineares)'),
        Patch(facecolor='#d62728', label='Ruído Metabólico (Sem Efeito)')
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.9)

    plt.sca(axes[1])
    sv_plot = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
    shap.plots.beeswarm(sv_plot, max_display=15, show=False, plot_size=None)
    axes[1].set_title("SHAP Beeswarm Plot (Dispersão Local de Impacto)", fontsize=14, fontweight="bold", pad=15)
    axes[1].tick_params(axis='both', which='major', labelsize=11)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()
    print(f"  [+] Gráficos SHAP salvos em: {output_path}")

    return explainer, shap_values, df_importancia, tempo_shap


# =============================================================================
# ETAPA 3: BENCHMARK AMPLIADO DE BASELINES DE SELEÇÃO (FILTER, WRAPPER, EMBEDDED, XAI)
# =============================================================================
def gerar_rankings_todas_familias(X_train, y_train, modelo_base, df_importancia_shap, tempo_shap):
    """
    Gera os rankings completos aprendidos estritamente em X_train para todas as categorias:
    1. Filter: Mutual Information e ANOVA F-score
    2. Wrapper: RFE com Random Forest
    3. Embedded: Regressão Logística Lasso L1 e Random Forest MDI (Gini)
    4. XAI: TreeSHAP
    """
    print("\n[ETAPA 3/7] Gerando rankings de atributos em todas as famílias de seleção...")
    rankings = {}
    tempos_selecao = {}

    # 1. Filter: Mutual Information
    t0 = time.perf_counter()
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    tempos_selecao["Filter (Mutual Info)"] = time.perf_counter() - t0
    rankings["Filter (Mutual Info)"] = X_train.columns[np.argsort(mi_scores)[::-1]].tolist()

    # 2. Filter: ANOVA F-test
    t0 = time.perf_counter()
    f_scores, _ = f_classif(X_train, y_train)
    tempos_selecao["Filter (ANOVA F)"] = time.perf_counter() - t0
    rankings["Filter (ANOVA F)"] = X_train.columns[np.argsort(f_scores)[::-1]].tolist()

    # 3. Wrapper: RFE com Random Forest
    t0 = time.perf_counter()
    rfe = RFE(estimator=RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=1), n_features_to_select=1, step=2)
    rfe.fit(X_train, y_train)
    tempos_selecao["Wrapper (RFE)"] = time.perf_counter() - t0
    df_rfe = pd.DataFrame({"atributo": X_train.columns, "ranking": rfe.ranking_}).sort_values(by="ranking").reset_index(drop=True)
    rankings["Wrapper (RFE)"] = df_rfe["atributo"].tolist()

    # 4. Embedded: Logística L1 (Lasso)
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    l1_model = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=0.1, random_state=42, max_iter=1000)
    l1_model.fit(X_train_scaled, y_train)
    tempos_selecao["Embedded (Lasso L1)"] = time.perf_counter() - t0
    l1_coefs = np.abs(l1_model.coef_[0])
    rankings["Embedded (Lasso L1)"] = X_train.columns[np.argsort(l1_coefs)[::-1]].tolist()

    # 5. Embedded: Random Forest MDI (Gini Importance)
    t0 = time.perf_counter()
    mdi_scores = modelo_base.feature_importances_
    tempos_selecao["Embedded (Tree MDI)"] = time.perf_counter() - t0
    rankings["Embedded (Tree MDI)"] = X_train.columns[np.argsort(mdi_scores)[::-1]].tolist()

    # 6. XAI: TreeSHAP
    rankings["XAI (TreeSHAP)"] = df_importancia_shap["atributo"].tolist()
    tempos_selecao["XAI (TreeSHAP)"] = tempo_shap

    return rankings, tempos_selecao


def avaliar_comparativo_familias(rankings, tempos_selecao, X_train, X_test, y_train, y_test, top_k=10):
    """
    Treina modelos no mesmo número de atributos (Top-K = 10) para cada método
    e constrói a tabela comparativa mestre com métricas preditivas, eficiência e pureza de seleção.
    """
    linhas = []

    for metodo, cols_ordenadas in rankings.items():
        cols_k = cols_ordenadas[:top_k]
        
        # Contagem de pureza de seleção
        n_bio = sum(1 for c in cols_k if 'biomarcador' in c)
        n_red = sum(1 for c in cols_k if 'redundante' in c)
        n_ruido = sum(1 for c in cols_k if 'ruido' in c)

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        metricas, _, _ = treinar_e_avaliar_modelo(clf, X_train[cols_k], X_test[cols_k], y_train, y_test)

        linhas.append({
            "Metodo": metodo,
            "Familia": metodo.split()[0].replace("(", "").replace(")", ""),
            "N_Atributos": top_k,
            "Biomarcadores_Retidos (max 10)": n_bio,
            "Ruidos_Selecionados (max 20)": n_ruido,
            "Taxa_Rejeicao_Ruido": f"{(1 - n_ruido / 20.0) * 100:.1f}%",
            "F1_Score": metricas["f1_score"],
            "Acuracia": metricas["acuracia"],
            "Recall": metricas["recall"],
            "ROC_AUC": metricas["roc_auc"],
            "Tempo_Selecao_s": tempos_selecao.get(metodo, 0.0),
            "Tempo_Treino_ms": metricas["tempo_treino_ms"],
            "Latencia_Inf_ms": metricas["tempo_inferencia_ms"]
        })

    df_comp = pd.DataFrame(linhas).sort_values(by="F1_Score", ascending=False).reset_index(drop=True)
    return df_comp


# =============================================================================
# ETAPA 4: ESTUDO DE ABLAÇÃO PROGRESSIVA (CURVAS DE PODA)
# =============================================================================
def executar_etapa_ablacao(X_train, X_test, y_train, y_test, rankings, output_path=os.path.join(OUTPUT_DIR, "modulo4_ablation_curves.png")):
    """
    Gera as curvas completas de ablação progressiva (de 40 a 2 atributos) para
    representantes de Filter, Wrapper, Embedded e XAI.
    """
    print("\n[ETAPA 4/7] Executando estudo de ablação progressiva em todas as famílias...")
    passos = list(range(X_train.shape[1], 1, -2))

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

    curvas = {
        "XAI (TreeSHAP)": rodar_curva(rankings["XAI (TreeSHAP)"], "XAI (TreeSHAP)"),
        "Filter (Mutual Info)": rodar_curva(rankings["Filter (Mutual Info)"], "Filter (Mutual Info)"),
        "Filter (ANOVA F)": rodar_curva(rankings["Filter (ANOVA F)"], "Filter (ANOVA F)"),
        "Wrapper (RFE)": rodar_curva(rankings["Wrapper (RFE)"], "Wrapper (RFE)"),
        "Embedded (Lasso L1)": rodar_curva(rankings["Embedded (Lasso L1)"], "Embedded (Lasso L1)"),
        "Embedded (Tree MDI)": rodar_curva(rankings["Embedded (Tree MDI)"], "Embedded (Tree MDI)")
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    estilos = {
        "XAI (TreeSHAP)": ('#1f77b4', 'o', '-', 2.5),
        "Filter (Mutual Info)": ('#9467bd', '^', '-.', 1.8),
        "Filter (ANOVA F)": ('#8c564b', 'v', ':', 1.8),
        "Wrapper (RFE)": ('#ff7f0e', 's', '--', 2.0),
        "Embedded (Lasso L1)": ('#2ca02c', 'D', ':', 1.8),
        "Embedded (Tree MDI)": ('#e377c2', 'p', '--', 1.8)
    }

    for nome, df_c in curvas.items():
        cor, mk, ls, lw = estilos[nome]
        axes[0].plot(df_c["n_atributos"], df_c["f1_score"], marker=mk, linestyle=ls, color=cor, lw=lw, label=nome)

    axes[0].axvline(x=10, color='red', linestyle=':', lw=2, label='10 Biomarcadores Vitais')
    axes[0].invert_xaxis()
    axes[0].set_title("1. Desempenho Clínico (F1-Score) vs. Poda de Atributos", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Número de Atributos Retidos (Poda ->)", fontweight="bold")
    axes[0].set_ylabel("F1-Score no Teste", fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(curvas["XAI (TreeSHAP)"]["n_atributos"], curvas["XAI (TreeSHAP)"]["tempo_ms"], marker='o', color='#1f77b4', lw=2.5, label="XAI (SHAP)")
    axes[1].plot(curvas["Wrapper (RFE)"]["n_atributos"], curvas["Wrapper (RFE)"]["tempo_ms"], marker='s', color='#ff7f0e', lw=2.0, label="Wrapper (RFE)")
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

    return curvas


# =============================================================================
# ETAPA 5: PRÉ-FILTRO HÍBRIDO E SELEÇÃO RIGOROSA SHAP-SELECT
# =============================================================================
def pré_filtro_hibrido(X_train, threshold_var=0.01, threshold_corr=0.90):
    """
    Fase 1: Eliminação rápida pré-XAI de variância quase constante e colinearidade extrema.
    """
    selector = VarianceThreshold(threshold=threshold_var).fit(X_train)
    cols_var = X_train.columns[selector.get_support()].tolist()
    X_var = X_train[cols_var]

    corr_matrix = X_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col].dropna() > threshold_corr).any()]
    cols_finais = [c for c in cols_var if c not in to_drop]

    return X_train[cols_finais], cols_finais


def executar_shap_select(X_train, y_train, p_value_cutoff=0.05, output_path=os.path.join(OUTPUT_DIR, "modulo5_shap_select_analysis.png")):
    """
    Fase 2: Regressão OLS/Logística dos rótulos reais y vs Matriz SHAP.
    Atributos com beta <= 0 ou p-valor >= 0.05 são descartados.
    """
    print("\n[ETAPA 5/7] Executando Pré-Filtro Híbrido + Seleção Rigorosa shap-select...")
    t0_selecao = time.perf_counter()
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
        if not res.mle_retvals.get("converged", True):
            raise RuntimeError("o ajuste Logit não convergiu")
        params, pvalues = res.params.drop("const"), res.pvalues.drop("const")
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, RuntimeError) as error:
        print(f"  [!] Logit indisponível ({error}); usando OLS como fallback explícito.")
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
    if not atributos_aprovados:
        raise ValueError(
            "shap-select não encontrou atributos que atendam aos critérios "
            "de coeficiente positivo e p-valor inferior ao cutoff."
        )
    tempo_selecao = time.perf_counter() - t0_selecao

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
    print(f"  [*] Selecionados {len(atributos_aprovados)} atributos aprovados pelos critérios estatísticos.")

    return atributos_aprovados, df_stat, tempo_selecao


# =============================================================================
# ETAPA 6: OTIMIZAÇÃO BAYESIANA COM OPTUNA NO ESPAÇO REDUZIDO
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

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# =============================================================================
# ETAPA 7: AUDITORIA FORENSE LOCAL VIA LIME E GANHO DE FIDELIDADE (R²)
# =============================================================================
def executar_auditoria_lime_comparativa(modelo_base, modelo_red, X_train_base, X_train_red, X_test_base, X_test_red, y_proba_base, output_path=os.path.join(OUTPUT_DIR, "modulo3_lime_local.png")):
    """
    Audita o mesmo paciente no limiar crítico (P ≈ 50%) em ambos os modelos:
    1. Baseline (40 atributos): superfície ruidosa, regras difusas, menor R².
    2. Modelo Reduzido Campeão (10 atributos): superfície limpa, regras focadas, maior R².
    Demonstra empiricamente o ganho de fidelidade local (Surrogate R²) do laudo médico.
    """
    print("\n[ETAPA 7/7] Executando Auditoria Forense Local via LIME (Comparativo de Fidelidade R²)...")

    # Localiza o paciente mais limítrofe no teste (maior dúvida diagnóstica)
    idx_limiar = int(np.argmin(np.abs(y_proba_base - 0.50)))
    proba_base = y_proba_base[idx_limiar]

    # Explainer Baseline (40 atributos)
    exp_lime_base = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_base),
        feature_names=X_train_base.columns.tolist(),
        class_names=["Saudável (0)", "Patologia (1)"],
        mode="classification",
        random_state=42
    )
    exp_inst_base = exp_lime_base.explain_instance(
        data_row=np.array(X_test_base.iloc[idx_limiar]),
        predict_fn=modelo_base.predict_proba,
        num_features=10
    )
    r2_base = getattr(exp_inst_base, 'score', 0.0)
    df_lime_base = pd.DataFrame(exp_inst_base.as_list(), columns=["regra_atributo", "peso_local"])

    # Explainer Reduzido (10 atributos)
    exp_lime_red = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_red),
        feature_names=X_train_red.columns.tolist(),
        class_names=["Saudável (0)", "Patologia (1)"],
        mode="classification",
        random_state=42
    )
    proba_red = modelo_red.predict_proba(X_test_red.iloc[[idx_limiar]])[0, 1]
    exp_inst_red = exp_lime_red.explain_instance(
        data_row=np.array(X_test_red.iloc[idx_limiar]),
        predict_fn=modelo_red.predict_proba,
        num_features=10
    )
    r2_red = getattr(exp_inst_red, 'score', 0.0)
    df_lime_red = pd.DataFrame(exp_inst_red.as_list(), columns=["regra_atributo", "peso_local"])

    # Visualização Comparativa Lado a Lado
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    cores_base = ['#2ca02c' if p > 0 else '#d62728' for p in df_lime_base['peso_local']]
    axes[0].barh(df_lime_base['regra_atributo'][::-1], df_lime_base['peso_local'][::-1], color=cores_base[::-1])
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_title(f"A. LIME Baseline (40 Atributos)\nP(Patologia) = {proba_base:.1%} | Fidelidade R² = {r2_base:.3f}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Impacto Local do Atributo (Peso LIME)", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    cores_red = ['#2ca02c' if p > 0 else '#d62728' for p in df_lime_red['peso_local']]
    axes[1].barh(df_lime_red['regra_atributo'][::-1], df_lime_red['peso_local'][::-1], color=cores_red[::-1])
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_title(f"B. LIME Reduzido Campeão (10 Atributos)\nP(Patologia) = {proba_red:.1%} | Fidelidade R² = {r2_red:.3f}", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Impacto Local do Atributo (Peso LIME)", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor='#2ca02c', label='Aumenta Risco de Patologia (1)'),
        Patch(facecolor='#d62728', label='Favorece Diagnóstico Saudável (0)')
    ]
    axes[1].legend(handles=legend_elements, loc="lower right")

    plt.suptitle(f"AUDITORIA FORENSE LOCAL: PACIENTE #{idx_limiar} NO LIMIAR CRÍTICO DE DECISÃO", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Auditoria LIME comparativa salva em: {output_path}")
    print(f"  [*] Fidelidade do Modelo Substituto (R²): Baseline = {r2_base:.3f} -> Reduzido = {r2_red:.3f}")

    return {
        "paciente_idx": idx_limiar,
        "proba_base": proba_base,
        "proba_red": proba_red,
        "r2_lime_base": r2_base,
        "r2_lime_red": r2_red,
        "df_lime_base": df_lime_base,
        "df_lime_red": df_lime_red
    }


def avaliar_metricas_interpretabilidade_multidimensional(
    df_importancia_shap,
    auditoria_lime,
    n_attrs_base,
    n_attrs_red,
    atributos_selecionados,
    top_k=10
):
    """
    Calcula formalmente o ganho de interpretabilidade através de 4 eixos declarados:
    1. Concentração da Massa de Shapley no Top-K
    2. Ganho de Fidelidade Local LIME (R²)
    3. Carga Cognitiva Humana (Miller's Span: 7 ± 2)
    4. Taxa de Eliminação de Ruído
    """
    massa_total = df_importancia_shap["importancia_shap"].sum()
    massa_top_k = df_importancia_shap.head(top_k)["importancia_shap"].sum()
    concentracao_shap = (massa_top_k / massa_total) if massa_total > 0 else 0.0
    massa_selecionada = df_importancia_shap.loc[
        df_importancia_shap["atributo"].isin(atributos_selecionados),
        "importancia_shap"
    ].sum()
    massa_shap_selecionada = (massa_selecionada / massa_total) if massa_total > 0 else 0.0

    r2_base = auditoria_lime["r2_lime_base"]
    r2_red = auditoria_lime["r2_lime_red"]
    ganho_fidelidade_pct = ((r2_red - r2_base) / abs(r2_base) * 100) if r2_base != 0 else 0.0

    return {
        "reducao_dimensional_pct": (1 - n_attrs_red / n_attrs_base) * 100,
        "concentracao_shap_top10": concentracao_shap,
        "massa_shap_selecionada_pct": massa_shap_selecionada,
        "r2_fidelidade_base": r2_base,
        "r2_fidelidade_red": r2_red,
        "ganho_fidelidade_r2_pct": ganho_fidelidade_pct,
        "carga_cognitiva_miller": (
            f"{n_attrs_red} atributos; acima do limite heurístico de 9"
            if n_attrs_red > 9
            else f"{n_attrs_red} atributos; dentro do limite heurístico de 9"
        ),
        "declaracao_epistemica": (
            "Dados sintéticos; SHAP/LIME descrevem o modelo e não demonstram causalidade "
            "ou validade clínica. Miller é apenas uma heurística, sem teste com usuários."
        )
    }


# =============================================================================
# ETAPA 8: DASHBOARD EXECUTIVO FINAL DE 4 PAINÉIS
# =============================================================================
def gerar_dashboard_executivo(m_base, m_red, inter_metrics, output_path=os.path.join(OUTPUT_DIR, "dashboard_final_comparativo.png")):
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
    axes[0, 0].set_title(f"1. Dimensão dos Dados (-{inter_metrics['reducao_dimensional_pct']:.0f}% de Coleta)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Quantidade de Atributos", fontweight="bold")
    for i, v in enumerate(attrs):
        axes[0, 0].text(i, v / 2, str(v), ha='center', va='center', color='white', fontweight='bold', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Eficiência Computacional (Treino e Inferência)
    x_pos = np.arange(len(categorias))
    w = 0.35
    axes[0, 1].bar(x_pos - w/2, [m_base["tempo_treino_ms"], m_red["tempo_treino_ms"]], w, label="Treino (ms)", color="#1f77b4")
    axes[0, 1].bar(x_pos + w/2, [m_base["tempo_inferencia_ms"], m_red["tempo_inferencia_ms"]], w, label="Inferência (ms)", color="#ff7f0e")
    axes[0, 1].set_title("2. Custo Computacional (Treino vs. Inferência)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(categorias, fontweight="bold")
    axes[0, 1].set_ylabel("Milissegundos (ms)", fontweight="bold")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Integridade Preditiva
    axes[1, 0].bar(x_pos - w/2, [m_base["acuracia"], m_red["acuracia"]], w, label="Acurácia", color="#1f77b4")
    axes[1, 0].bar(x_pos + w/2, [m_base["f1_score"], m_red["f1_score"]], w, label="F1-Score", color="#2ca02c")
    axes[1, 0].set_title("3. Integridade Preditiva (Acurácia vs. F1-Score)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(categorias, fontweight="bold")
    axes[1, 0].set_ylim(0.70, 1.0)
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Métricas Formais de Interpretabilidade
    metricas_inter = ["Fidelidade LIME (R²)", "Massa SHAP do baseline retida"]
    valores_base = [inter_metrics["r2_fidelidade_base"], 1.0]
    valores_red = [
        inter_metrics["r2_fidelidade_red"],
        inter_metrics["massa_shap_selecionada_pct"]
    ]

    x_inter = np.arange(len(metricas_inter))
    axes[1, 1].bar(x_inter - w/2, valores_base, w, label="Baseline", color="#d62728")
    axes[1, 1].bar(x_inter + w/2, valores_red, w, label="Atributos selecionados", color="#2ca02c")
    axes[1, 1].set_title("4. Proxies de Explicabilidade", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(x_inter)
    axes[1, 1].set_xticklabels(metricas_inter, fontweight="bold")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend(loc="upper left")
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  [+] Dashboard Executivo salvo com sucesso em: {output_path}")


# =============================================================================
# ORQUESTRADOR CENTRAL: EXECUÇÃO COMPLETA DO EXPERIMENTO
# =============================================================================
def executar_pipeline_completo():
    print("=" * 80)
    print("INICIANDO PIPELINE CIENTÍFICO XAI DATA REDUCTION (VERSÃO FINAL)")
    print("=" * 80)

    # 1. Carregamento de dados e particionamento
    X, y = gerar_dataset_sintetico_saude()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"[*] Base de Dados: {X.shape[0]} instâncias, {X.shape[1]} atributos iniciais.")
    print(f"    - 10 atributos informativos sintéticos")
    print(f"    - 10 atributos redundantes sintéticos")
    print(f"    - 20 atributos de ruído sintético")

    # 2. Treino e Avaliação do Baseline
    print("\n[ETAPA 1/7] Treinando Modelo Baseline (40 Atributos)...")
    modelo_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    m_baseline, y_pred_base, y_proba_base = treinar_e_avaliar_modelo(modelo_base, X_train, X_test, y_train, y_test)
    plotar_graficos_baseline(y_test, y_pred_base, y_proba_base)

    # 3. Explicabilidade Global SHAP
    explainer_shap, shap_values, df_importancia_shap, tempo_shap = executar_etapa_shap(modelo_base, X_train)

    # 4. Rankings em todas as 4 famílias de Baselines
    rankings, tempos_selecao = gerar_rankings_todas_familias(
        X_train, y_train, modelo_base, df_importancia_shap, tempo_shap
    )
    df_comparativo_familias = avaliar_comparativo_familias(rankings, tempos_selecao, X_train, X_test, y_train, y_test, top_k=10)

    # 5. Estudo de Ablação Progressiva
    curvas_ablacao = executar_etapa_ablacao(X_train, X_test, y_train, y_test, rankings)

    # 6. Engenharia Avançada: Pré-Filtro Híbrido + shap-select
    atributos_selecionados, df_shap_select, tempo_selecao_proposta = executar_shap_select(X_train, y_train)
    X_train_red = X_train[atributos_selecionados]
    X_test_red = X_test[atributos_selecionados]

    # 7. Otimização Bayesiana com Optuna
    print(f"\n[ETAPA 6/7] Otimizando Hiperparâmetros via Optuna no espaço de {len(atributos_selecionados)} atributos...")
    best_params = otimizar_optuna(X_train_red, y_train, n_trials=15)
    print(f"  [*] Melhores Hiperparâmetros encontrados: {best_params}")

    # Treino do Modelo Campeão Reduzido
    modelo_campeao = RandomForestClassifier(**best_params, random_state=42, n_jobs=1)
    m_reduzido, y_pred_red, y_proba_red = treinar_e_avaliar_modelo(modelo_campeao, X_train_red, X_test_red, y_train, y_test)

    # 8. Auditoria Forense Local via LIME (Comparativo de Fidelidade)
    auditoria_lime = executar_auditoria_lime_comparativa(
        modelo_base, modelo_campeao,
        X_train, X_train_red,
        X_test, X_test_red,
        y_proba_base
    )

    # 9. Avaliação Multidimensional da Interpretabilidade
    inter_metrics = avaliar_metricas_interpretabilidade_multidimensional(
        df_importancia_shap, auditoria_lime,
        m_baseline["num_atributos"], m_reduzido["num_atributos"],
        atributos_selecionados, top_k=10
    )

    # 10. Dashboard Executivo Final
    gerar_dashboard_executivo(m_baseline, m_reduzido, inter_metrics)

    # Adiciona o Modelo Campeão à Tabela Comparativa de Famílias
    n_bio_camp = sum(1 for c in atributos_selecionados if 'biomarcador' in c)
    n_ruido_camp = sum(1 for c in atributos_selecionados if 'ruido' in c)
    linha_campeao = pd.DataFrame([{
        "Metodo": "XAI Proposto (shap-select + Optuna)",
        "Familia": "XAI (Proposta)",
        "N_Atributos": len(atributos_selecionados),
        "Biomarcadores_Retidos (max 10)": n_bio_camp,
        "Ruidos_Selecionados (max 20)": n_ruido_camp,
        "Taxa_Rejeicao_Ruido": f"{(1 - n_ruido_camp / 20.0) * 100:.1f}%",
        "F1_Score": m_reduzido["f1_score"],
        "Acuracia": m_reduzido["acuracia"],
        "Recall": m_reduzido["recall"],
        "ROC_AUC": m_reduzido["roc_auc"],
        "Tempo_Selecao_s": tempo_selecao_proposta,
        "Tempo_Treino_ms": m_reduzido["tempo_treino_ms"],
        "Latencia_Inf_ms": m_reduzido["tempo_inferencia_ms"],
        "R2_LIME_Baseline": auditoria_lime["r2_lime_base"],
        "R2_LIME_Reduzido": auditoria_lime["r2_lime_red"],
        "Massa_SHAP_Retida_pct": inter_metrics["massa_shap_selecionada_pct"],
        "Paciente_Limiar": auditoria_lime["paciente_idx"],
        "Probabilidade_Baseline_Limiar": auditoria_lime["proba_base"]
    }])
    linha_baseline = pd.DataFrame([{
        "Metodo": "Baseline Completo (40 Atributos)",
        "Familia": "Referência",
        "N_Atributos": m_baseline["num_atributos"],
        "Biomarcadores_Retidos (max 10)": 10,
        "Ruidos_Selecionados (max 20)": 20,
        "Taxa_Rejeicao_Ruido": "0.0%",
        "F1_Score": m_baseline["f1_score"],
        "Acuracia": m_baseline["acuracia"],
        "Recall": m_baseline["recall"],
        "ROC_AUC": m_baseline["roc_auc"],
        "Tempo_Selecao_s": 0.0,
        "Tempo_Treino_ms": m_baseline["tempo_treino_ms"],
        "Latencia_Inf_ms": m_baseline["tempo_inferencia_ms"],
        "R2_LIME_Baseline": np.nan,
        "R2_LIME_Reduzido": np.nan,
        "Massa_SHAP_Retida_pct": np.nan,
        "Paciente_Limiar": np.nan,
        "Probabilidade_Baseline_Limiar": np.nan
    }])
    df_tabela_mestra = pd.concat(
        [linha_campeao, df_comparativo_familias, linha_baseline],
        ignore_index=True
    )
    caminho_csv = os.path.join(OUTPUT_DIR, "tabela_comparativa_completa.csv")
    df_tabela_mestra.to_csv(caminho_csv, index=False)
    print(f"\n[+] Tabela comparativa mestre exportada com sucesso em: {caminho_csv}")

    # 11. Relatório Comparativo Final no Terminal
    print("\n" + "=" * 90)
    print("TABELA COMPARATIVA MESTRE: TODAS AS FAMÍLIAS DE SELEÇÃO vs. PROPOSTA XAI")
    print("=" * 90)
    print(df_tabela_mestra.to_string(index=False))

    print("\n" + "=" * 90)
    print("RELATÓRIO COMPARATIVO EXECUTIVO: BASELINE (40) vs. XAI REDUZIDO (10)")
    print("=" * 90)
    print(f" Dimensão Avaliada        | Baseline (40 Atributos) | XAI Reduzido + Optuna")
    print(f"--------------------------+-------------------------+----------------------")
    print(f" Nº de Atributos          | {m_baseline['num_atributos']:<23} | {m_reduzido['num_atributos']:<20}")
    print(f" Acurácia Global          | {m_baseline['acuracia']:<23.4f} | {m_reduzido['acuracia']:<20.4f}")
    print(f" F1-Score (sintético)     | {m_baseline['f1_score']:<23.4f} | {m_reduzido['f1_score']:<20.4f}")
    print(f" ROC-AUC (Área)           | {m_baseline['roc_auc']:<23.4f} | {m_reduzido['roc_auc']:<20.4f}")
    print(f" Tempo de Treino (ms)     | {m_baseline['tempo_treino_ms']:<23.2f} | {m_reduzido['tempo_treino_ms']:<20.2f}")
    print(f" Latência Inferência (ms) | {m_baseline['tempo_inferencia_ms']:<23.2f} | {m_reduzido['tempo_inferencia_ms']:<20.2f}")
    print(f" Redução Dimensional     | 0.0%                    | {inter_metrics['reducao_dimensional_pct']:.1f}%")
    print(f" Massa SHAP do baseline   | 100.0%                  | {inter_metrics['massa_shap_selecionada_pct']:.1%}")
    print(f" Fidelidade Local LIME R² | {inter_metrics['r2_fidelidade_base']:<23.4f} | {inter_metrics['r2_fidelidade_red']:<20.4f}")
    print(f" Proxy de cardinalidade   | 40 atributos            | {inter_metrics['carga_cognitiva_miller']}")
    print("=" * 90)

    # 12. Validação via Sanity Checks
    taxa_reducao = inter_metrics["reducao_dimensional_pct"]
    assert taxa_reducao >= 50.0, f"Erro: Redução dimensional insuficiente ({taxa_reducao:.1f}%)"
    assert m_reduzido["f1_score"] >= (m_baseline["f1_score"] - 0.05), "Erro: Queda excessiva no F1-Score"

    print("\nSANITY CHECKS APROVADOS COM SUCESSO:")
    print(f"  [OK] Redução Dimensional       : {taxa_reducao:.1f}% dos atributos eliminados!")
    print(f"  [INFO] Fidelidade local LIME    : R² {inter_metrics['r2_fidelidade_base']:.3f} -> {inter_metrics['r2_fidelidade_red']:.3f} (uma instância)")
    print(f"  [OK] Pureza de Seleção         : {n_bio_camp}/10 biomarcadores retidos e {n_ruido_camp} ruídos aceitos!")
    print(f"  [INFO] F1 em dados sintéticos   : Baseline = {m_baseline['f1_score']:.4f} vs. Reduzido = {m_reduzido['f1_score']:.4f}")
    print("=" * 90)

    return df_tabela_mestra, m_baseline, m_reduzido, inter_metrics


if __name__ == "__main__":
    executar_pipeline_completo()
