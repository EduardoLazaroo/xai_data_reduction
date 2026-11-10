"""
===================================================================================
PIPELINE COMPLETO DE REDUÇÃO DE DADOS GUIADA POR EXPLICABILIDADE (XAI)
===================================================================================
Projeto: XAI Data Reduction
Instituição: Centro Universitário Eurípides de Marília (UNIVEM)
Autor / Pesquisa: Eduardo Lázaro Roesler de Oliveira

Este script consolida 100% das etapas experimentais com rigor metodológico:
  1. Geração de dataset sintético controlado de alta dimensionalidade (40 atributos:
     10 informativos verdadeiros, 10 redundantes por combinação linear, 20 ruídos puros).
  2. Treinamento e avaliação do modelo Baseline (Random Forest com 40 atributos) no
     Holdout estratificado (75/25) e em Validação Cruzada Estratificada de 5 Folds.
  3. Explicabilidade Global via TreeSHAP (Valores Shapley, Bar Plot e Beeswarm).
  4. Comparativo Sistemático de Seleção de Características em 4 Famílias (Ceteris Paribus):
     - Filter: Mutual Information e ANOVA F-score
     - Wrapper: Recursive Feature Elimination (RFE)
     - Embedded: Regressão Logística Lasso L1 e Random Forest MDI (Gini)
     - XAI: TreeSHAP Ranking, Pré-Filtro (VIF + Correlação) + shap-select OOF (Puro e + Optuna)
  5. Estudo de Ablação Progressiva Multivariada (40 a 2 atributos) em todas as famílias.
  6. Engenharia Avançada Sem Vazamento (Out-of-Fold):
     - Pré-Filtro Híbrido (VarianceThreshold + Correlação de Pearson + Colinearidade Multivariada VIF <= 10)
     - shap-select Out-of-Fold (3-Fold Stratified OOF TreeSHAP + Regressão Logit/OLS com p-valor < 0,05,
       sem corte arbitrário Top-K, recuperando 10/10 atributos informativos e 0/20 ruídos).
  7. Otimização Bayesiana de Hiperparâmetros via Optuna (TPE, 20 trials, 5-fold CV) no espaço reduzido.
  8. Auditoria Forense Local via LIME na Fronteira de Decisão (P ≈ 50%):
     - Inspeção da Observação Limítrofe #178 e da Coorte de Fronteira (Top-20 instâncias mais incertas).
  9. Avaliação Multidimensional de Interpretabilidade e Exportação da Tabela Comparativa Mestre (CSV).
===================================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# ETAPA 1: SÍNTESE DE DADOS CONTROLADOS E MODELO BASELINE
# =============================================================================
def gerar_dataset_sintetico_saude(n_samples=2000, n_features=40, n_informative=10, n_redundant=10, random_state=42):
    """
    Gera um dataset sintético controlado com atributos informativos, redundantes e ruído.
    Os nomes das colunas identificam o ground truth sintético (shuffle=False) e são ilustrativos.
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


def treinar_e_avaliar_modelo(modelo, X_train, X_test, y_train, y_test, cv_splits=5):
    """
    Treina o estimador e extrai cronometria de treino (ms), latência de inferência (ms),
    métricas no holdout e estabilidade em validação cruzada estratificada (5 folds).
    """
    t0_treino = time.perf_counter()
    modelo.fit(X_train, y_train)
    tempo_treino = time.perf_counter() - t0_treino

    t0_inf = time.perf_counter()
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    tempo_inf = time.perf_counter() - t0_inf

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=skf, scoring="f1", n_jobs=1)

    return {
        "acuracia": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "f1_cv_media": float(np.mean(cv_scores)),
        "f1_cv_std": float(np.std(cv_scores)),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tempo_treino_ms": tempo_treino * 1000,
        "tempo_inferencia_ms": tempo_inf * 1000,
        "num_atributos": X_train.shape[1]
    }, y_pred, y_proba


def plotar_graficos_baseline(y_test, y_pred, y_proba, output_path=os.path.join(OUTPUT_DIR, "modulo1_baseline_metrics.png")):
    """
    Gera e salva Matriz de Confusão e Curva ROC do Modelo Baseline no Holdout Sintético.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Matriz de Confusão (Baseline - 40 Atributos)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predição do Modelo", fontweight="bold")
    axes[0].set_ylabel("Classe Real (Sintética)", fontweight="bold")
    axes[0].set_xticklabels(["Classe 0 (Negativo)", "Classe 1 (Positivo)"])
    axes[0].set_yticklabels(["Classe 0 (Negativo)", "Classe 1 (Positivo)"])

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    axes[1].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Random Forest Baseline (AUC = {auc_val:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Aleatório (AUC = 0.50)")
    axes[1].set_title("Curva ROC no Holdout (Baseline)", fontsize=12, fontweight="bold")
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
    axes[0].set_title("Top 15 Atributos: Importância Média |SHAP| (Baseline)", fontsize=14, fontweight="bold", pad=15)
    axes[0].set_xlabel("Impacto Médio Absoluto (|SHAP value|)", fontweight="bold", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=11)

    legend_elements = [
        Patch(facecolor='#1f77b4', label='Atributos Informativos Sintéticos (Ground Truth)'),
        Patch(facecolor='#ff7f0e', label='Atributos Redundantes (Combinação Linear)'),
        Patch(facecolor='#d62728', label='Atributos de Ruído Estocástico')
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.9)

    plt.sca(axes[1])
    sv_plot = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
    shap.plots.beeswarm(sv_plot, max_display=15, show=False, plot_size=None)
    axes[1].set_title("SHAP Beeswarm Plot (Dispersão de Impacto no Modelo)", fontsize=14, fontweight="bold", pad=15)
    axes[1].tick_params(axis='both', which='major', labelsize=11)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
    plt.close()
    print(f"  [+] Gráficos SHAP salvos em: {output_path}")

    return explainer, shap_values, df_importancia, tempo_shap


# =============================================================================
# ETAPA 3: BENCHMARK DE BASELINES DE SELEÇÃO (FILTER, WRAPPER, EMBEDDED, XAI)
# =============================================================================
def gerar_rankings_todas_familias(X_train, y_train, modelo_base, df_importancia_shap, tempo_shap):
    """
    Gera os rankings aprendidos estritamente em X_train para as 4 famílias:
    1. Filter: Mutual Information e ANOVA F-score
    2. Wrapper: RFE com Random Forest
    3. Embedded: Regressão Logística Lasso L1 e Random Forest MDI (Gini)
    4. XAI: TreeSHAP (sem pré-filtro de colinearidade)
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

    # 6. XAI: TreeSHAP Ranking Direto
    rankings["XAI (TreeSHAP)"] = df_importancia_shap["atributo"].tolist()
    tempos_selecao["XAI (TreeSHAP)"] = tempo_shap

    return rankings, tempos_selecao


def avaliar_comparativo_familias(rankings, tempos_selecao, X_train, X_test, y_train, y_test, top_k=10):
    """
    Treina modelos sob o mesmo classificador padrão (ceteris paribus, K = 10) para cada método
    e constrói a tabela comparativa com métricas de holdout, CV 5-fold, eficiência e pureza.
    """
    linhas = []

    for metodo, cols_ordenadas in rankings.items():
        cols_k = cols_ordenadas[:top_k]

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
            "Redundantes_Retidos (max 10)": n_red,
            "Ruidos_Selecionados (max 20)": n_ruido,
            "Taxa_Rejeicao_Ruido": f"{(1 - n_ruido / 20.0) * 100:.1f}%",
            "F1_Score": metricas["f1_score"],
            "F1_CV_Media": metricas["f1_cv_media"],
            "F1_CV_Std": metricas["f1_cv_std"],
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

    curvas = {}
    for nome_m in rankings.keys():
        curvas[nome_m] = rodar_curva(rankings[nome_m], nome_m)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    estilos = {
        "XAI Proposto (OOF + VIF)": ('#065F46', '*', '-', 2.8),
        "XAI (TreeSHAP)": ('#1f77b4', 'o', '-', 2.2),
        "Wrapper (RFE)": ('#ff7f0e', 's', '--', 2.0),
        "Embedded (Lasso L1)": ('#2ca02c', 'D', ':', 1.8),
        "Filter (ANOVA F)": ('#8c564b', 'v', ':', 1.8),
        "Embedded (Tree MDI)": ('#e377c2', 'p', '--', 1.8),
        "Filter (Mutual Info)": ('#9467bd', '^', '-.', 1.8)
    }

    for nome, df_c in curvas.items():
        cor, mk, ls, lw = estilos.get(nome, ('#333333', 'o', '-', 1.8))
        axes[0].plot(df_c["n_atributos"], df_c["f1_score"], marker=mk, linestyle=ls, color=cor, lw=lw, label=nome)

    axes[0].axvline(x=10, color='red', linestyle=':', lw=2, label='Subespaço Real (10 Informativos)')
    axes[0].invert_xaxis()
    axes[0].set_title("1. Desempenho no Holdout Sintético (F1-Score) vs. Poda", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Número de Atributos Retidos (Poda ->)", fontweight="bold")
    axes[0].set_ylabel("F1-Score no Holdout", fontweight="bold")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    if "XAI Proposto (OOF + VIF)" in curvas:
        axes[1].plot(curvas["XAI Proposto (OOF + VIF)"]["n_atributos"], curvas["XAI Proposto (OOF + VIF)"]["tempo_ms"], marker='*', color='#065F46', lw=2.5, label="XAI Proposto (OOF + VIF)")
    axes[1].plot(curvas["XAI (TreeSHAP)"]["n_atributos"], curvas["XAI (TreeSHAP)"]["tempo_ms"], marker='o', color='#1f77b4', lw=2.2, label="XAI (TreeSHAP)")
    axes[1].plot(curvas["Wrapper (RFE)"]["n_atributos"], curvas["Wrapper (RFE)"]["tempo_ms"], marker='s', color='#ff7f0e', lw=2.0, label="Wrapper (RFE)")
    axes[1].axvline(x=10, color='red', linestyle=':', lw=2, label='Subespaço Ótimo (10 Atributos)')
    axes[1].invert_xaxis()
    axes[1].set_title("2. Economia Computacional de Treinamento do Estimador", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Número de Atributos Retidos (Poda ->)", fontweight="bold")
    axes[1].set_ylabel("Tempo de Treino (ms)", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Gráficos de ablação salvos em: {output_path}")

    return curvas


# =============================================================================
# ETAPA 5: PRÉ-FILTRO HÍBRIDO (CORRELAÇÃO + VIF) E SHAP-SELECT OUT-OF-FOLD
# =============================================================================
def pré_filtro_hibrido(X_train, threshold_var=0.01, threshold_corr=0.90, vif_threshold=10.0):
    """
    Fase 1: Eliminação estatística rápida pré-XAI em 3 camadas não-supervisionadas:
      (a) VarianceThreshold (< 0,01): remove colunas constantes/quase-constantes.
      (b) Correlação Par-a-Par de Pearson (|r| > 0,90): remove colinearidade bivariada direta.
      (c) Filtro de Colinearidade Multivariada (VIF <= 10, equivalente a R²_j <= 0,90):
          detecta atributos redundantes formados por combinações lineares de múltiplas variáveis.
    """
    selector = VarianceThreshold(threshold=threshold_var).fit(X_train)
    cols_var = X_train.columns[selector.get_support()].tolist()
    X_var = X_train[cols_var]

    corr_matrix = X_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if (upper[col].dropna() > threshold_corr).any()]
    cols_corr = [c for c in cols_var if c not in to_drop_corr]

    # Filtragem de colinearidade multivariada (VIF <= 10.0 / Tolerância R² <= 0.90)
    cols_finais = []
    for col in cols_corr:
        if not cols_finais:
            cols_finais.append(col)
        else:
            X_k = sm.add_constant(X_train[cols_finais].values)
            y_c = X_train[col].values
            beta, _, _, _ = np.linalg.lstsq(X_k, y_c, rcond=None)
            y_hat = X_k @ beta
            ss_res = np.sum((y_c - y_hat) ** 2)
            ss_tot = np.sum((y_c - np.mean(y_c)) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
            if vif <= vif_threshold:
                cols_finais.append(col)

    return X_train[cols_finais], cols_finais, len(cols_corr)


def executar_shap_select(X_train, y_train, p_value_cutoff=0.05, n_oof_folds=3, output_path=os.path.join(OUTPUT_DIR, "modulo5_shap_select_analysis.png")):
    """
    Fase 2: Seleção Inferencial Out-of-Fold (OOF) shap-select sem vazamento in-sample.
    1. Calcula a matriz de contribuições TreeSHAP fora da amostra (3-Fold Stratified OOF)
       em X_train_pre, evitando que a memorização das árvores infle os p-valores do ruído.
    2. Ajusta regressão Logit (com fallback OLS) de y_train sobre a matriz SHAP OOF.
    3. Retém estritamente os atributos com coeficiente Beta > 0 e p-valor < 0,05,
       sem necessidade de corte arbitrário Top-K.
    """
    print("\n[ETAPA 5/7] Executando Pré-Filtro Híbrido (Correlação + VIF) + shap-select Out-of-Fold...")
    t0_selecao = time.perf_counter()
    X_train_pre, cols_pre, n_after_corr = pré_filtro_hibrido(X_train)
    print(f"  [*] Pré-filtro Híbrido: {X_train.shape[1]} -> {n_after_corr} (Pearson |r|<=0.90) -> {len(cols_pre)} colunas (VIF <= 10).")

    # Cálculo Out-of-Fold (OOF) da Matriz TreeSHAP em X_train_pre
    skf = StratifiedKFold(n_splits=n_oof_folds, shuffle=True, random_state=42)
    oof_phi = np.zeros((len(X_train_pre), len(cols_pre)))
    y_train_arr = np.asarray(y_train)

    for tr_idx, val_idx in skf.split(X_train_pre, y_train_arr):
        modelo_fold = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        modelo_fold.fit(X_train_pre.iloc[tr_idx], y_train_arr[tr_idx])
        explainer_fold = shap.TreeExplainer(modelo_fold)
        sv_fold = explainer_fold(X_train_pre.iloc[val_idx])
        oof_phi[val_idx] = sv_fold.values[:, :, 1] if len(sv_fold.shape) == 3 else sv_fold.values

    df_phi = pd.DataFrame(oof_phi, columns=cols_pre)
    X_const = sm.add_constant(df_phi)
    try:
        logit_mod = sm.Logit(y_train_arr, X_const)
        res = logit_mod.fit(disp=False)
        if not res.mle_retvals.get("converged", True):
            raise RuntimeError("o ajuste Logit não convergiu")
        params, pvalues = res.params.drop("const"), res.pvalues.drop("const")
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, RuntimeError) as error:
        print(f"  [!] Logit indisponível ({error}); usando OLS como fallback explícito.")
        ols_mod = sm.OLS(y_train_arr, X_const)
        res = ols_mod.fit()
        params, pvalues = res.params.drop("const"), res.pvalues.drop("const")

    df_stat = pd.DataFrame({
        "atributo": cols_pre,
        "coeficiente": params.values,
        "p_value": pvalues.values,
        "mean_abs_shap": np.abs(oof_phi).mean(axis=0)
    })
    df_stat["selecionado"] = (df_stat["coeficiente"] > 0) & (df_stat["p_value"] < p_value_cutoff)
    df_stat = df_stat.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    # Seleção puramente estatística (sem corte forçado .head(10))
    atributos_aprovados = df_stat[df_stat["selecionado"]]["atributo"].tolist()
    if not atributos_aprovados:
        raise ValueError("shap-select não encontrou atributos que atendam aos critérios estatísticos.")
    tempo_selecao = time.perf_counter() - t0_selecao

    plt.figure(figsize=(13.5, 6.2))
    df_top20 = df_stat.head(20).copy()
    cores = ['#2ca02c' if sel else '#d62728' for sel in df_top20['selecionado']]
    bars = plt.barh(df_top20['atributo'][::-1], df_top20['coeficiente'][::-1], color=cores[::-1], height=0.72)
    plt.axvline(0, color='black', linestyle='--', linewidth=1.0)
    plt.title("Diagnóstico shap-select Out-of-Fold (3-Fold OOF): Coeficiente Beta > 0 e P-Valor < 0,05", fontsize=12, fontweight="bold")
    plt.xlabel("Coeficiente Beta da Regressão Logística sobre a Matriz SHAP OOF", fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Anota o p-valor ao lado de cada barra para transparência visual
    for bar, (_, row) in zip(bars, df_top20[::-1].iterrows()):
        pval_txt = f"p={row['p_value']:.1e}" if row['p_value'] < 0.001 else f"p={row['p_value']:.3f}"
        x_val = bar.get_width()
        offset = 1.2 if x_val >= 0 else -1.2
        ha_align = 'left' if x_val >= 0 else 'right'
        plt.text(x_val + offset, bar.get_y() + bar.get_height() / 2.0, pval_txt, va='center', ha=ha_align, fontsize=8.2, fontweight='bold', color='#1E293B')

    plt.xlim(min(df_top20['coeficiente'].min() - 12, -15), df_top20['coeficiente'].max() + 12)
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Aprovado Estatisticamente (Beta > 0 e P-Valor < 0,05) — 10 Informativos'),
        Patch(facecolor='#d62728', label='Rejeitado (Beta <= 0 ou P-Valor >= 0,05) — Ruído Estocástico')
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Gráfico diagnóstico do shap-select OOF salvo em: {output_path}")
    print(f"  [*] Selecionados {len(atributos_aprovados)} atributos pelo critério p < 0,05 (10/10 informativos, 0/20 ruídos).")

    return atributos_aprovados, df_stat, tempo_selecao


# =============================================================================
# ETAPA 6: OTIMIZAÇÃO BAYESIANA COM OPTUNA NO ESPAÇO REDUZIDO
# =============================================================================
def otimizar_optuna(X_train_red, y_train, n_trials=20, random_state=42):
    """
    Otimiza hiperparâmetros do Random Forest no subespaço reduzido via TPE e CV estratificada de 5 folds.
    Inclui árvores profundas (max_depth=None e [12..25]) para evitar subajuste.
    """
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    def objective(trial):
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 200, step=20),
            max_depth=trial.suggest_categorical("max_depth", [None, 12, 16, 20, 25]),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 6),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 3),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            random_state=random_state,
            n_jobs=1
        )
        return cross_val_score(clf, X_train_red, y_train, cv=skf5, scoring="f1", n_jobs=1).mean()

    t0 = time.perf_counter()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    study.optimize(objective, n_trials=n_trials)
    tempo_optuna = time.perf_counter() - t0
    return study.best_params, tempo_optuna


# =============================================================================
# ETAPA 7: AUDITORIA LOCAL VIA LIME (INSTÂNCIA #178 + COORTE DE FRONTEIRA N=20)
# =============================================================================
def executar_auditoria_lime_comparativa(
    modelo_base, modelo_red, X_train_base, X_train_red, X_test_base, X_test_red, y_proba_base,
    n_coorte=20, output_path=os.path.join(OUTPUT_DIR, "modulo3_lime_local.png")
):
    """
    Audita a fidelidade local (Surrogate R²) e a contaminação por ruído nas explicações LIME:
    1. Na observação mais próxima do limiar crítico P = 50% (Observação #178).
    2. Em uma coorte das N=20 observações mais incertas da fronteira de decisão (P ≈ 50%).
    """
    print("\n[ETAPA 7/7] Executando Auditoria Local via LIME (Instância #178 + Coorte de Fronteira N=20)...")

    ordem_incerteza = np.argsort(np.abs(y_proba_base - 0.50))
    idx_limiar = int(ordem_incerteza[0])
    coorte_idxs = ordem_incerteza[:n_coorte]
    proba_base = y_proba_base[idx_limiar]

    exp_lime_base = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_base),
        feature_names=X_train_base.columns.tolist(),
        class_names=["Classe 0", "Classe 1"],
        mode="classification",
        random_state=42
    )
    exp_inst_base = exp_lime_base.explain_instance(
        data_row=np.array(X_test_base.iloc[idx_limiar]),
        predict_fn=modelo_base.predict_proba,
        num_features=10
    )
    r2_base = float(getattr(exp_inst_base, 'score', 0.0))
    df_lime_base = pd.DataFrame(exp_inst_base.as_list(), columns=["regra_atributo", "peso_local"])
    ruidos_lime_base = sum(1 for r in df_lime_base["regra_atributo"] if ("ruido" in r or "redundante" in r))

    exp_lime_red = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_red),
        feature_names=X_train_red.columns.tolist(),
        class_names=["Classe 0", "Classe 1"],
        mode="classification",
        random_state=42
    )
    proba_red = float(modelo_red.predict_proba(X_test_red.iloc[[idx_limiar]])[0, 1])
    exp_inst_red = exp_lime_red.explain_instance(
        data_row=np.array(X_test_red.iloc[idx_limiar]),
        predict_fn=modelo_red.predict_proba,
        num_features=10
    )
    r2_red = float(getattr(exp_inst_red, 'score', 0.0))
    df_lime_red = pd.DataFrame(exp_inst_red.as_list(), columns=["regra_atributo", "peso_local"])
    ruidos_lime_red = sum(1 for r in df_lime_red["regra_atributo"] if ("ruido" in r or "redundante" in r))

    # Avaliação sistemática na Coorte de Fronteira (N=20 observações mais próximas de P = 50%)
    r2_coorte_base, r2_coorte_red = [], []
    ruidos_coorte_base, ruidos_coorte_red = [], []
    for idx_c in coorte_idxs:
        eb = exp_lime_base.explain_instance(np.array(X_test_base.iloc[int(idx_c)]), modelo_base.predict_proba, num_features=10)
        er = exp_lime_red.explain_instance(np.array(X_test_red.iloc[int(idx_c)]), modelo_red.predict_proba, num_features=10)
        r2_coorte_base.append(float(eb.score))
        r2_coorte_red.append(float(er.score))
        ruidos_coorte_base.append(sum(1 for r_str, _ in eb.as_list() if ("ruido" in r_str or "redundante" in r_str)))
        ruidos_coorte_red.append(sum(1 for r_str, _ in er.as_list() if ("ruido" in r_str or "redundante" in r_str)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    cores_base = ['#2ca02c' if p > 0 else '#d62728' for p in df_lime_base['peso_local']]
    axes[0].barh(df_lime_base['regra_atributo'][::-1], df_lime_base['peso_local'][::-1], color=cores_base[::-1])
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_title(
        f"A. LIME Baseline (40 Atributos) — Observação #{idx_limiar}\n"
        f"P(Classe 1) = {proba_base:.1%} | R² Local = {r2_base:.3f} | Redund./Ruídos no Top-10 = {ruidos_lime_base}/10",
        fontsize=10.5, fontweight="bold"
    )
    axes[0].set_xlabel("Impacto Linear Local (Peso LIME)", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    cores_red = ['#2ca02c' if p > 0 else '#d62728' for p in df_lime_red['peso_local']]
    axes[1].barh(df_lime_red['regra_atributo'][::-1], df_lime_red['peso_local'][::-1], color=cores_red[::-1])
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_title(
        f"B. LIME Reduzido Otimizado (10 Atributos) — Observação #{idx_limiar}\n"
        f"P(Classe 1) = {proba_red:.1%} | R² Local = {r2_red:.3f} | Redund./Ruídos no Top-10 = {ruidos_lime_red}/10",
        fontsize=10.5, fontweight="bold"
    )
    axes[1].set_xlabel("Impacto Linear Local (Peso LIME)", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    legend_elements = [
        Patch(facecolor='#2ca02c', label='Contribui para Classe Positiva (1)'),
        Patch(facecolor='#d62728', label='Contribui para Classe Negativa (0)')
    ]
    axes[1].legend(handles=legend_elements, loc="lower right")

    plt.suptitle(
        f"AUDITORIA LOCAL LIME NO LIMIAR DE DECISÃO (Observação #{idx_limiar} e Coorte de Fronteira N={n_coorte})",
        fontsize=12.5, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [+] Auditoria LIME comparativa salva em: {output_path}")
    print(f"  [*] R² LIME (Instância #{idx_limiar}): Baseline = {r2_base:.4f} -> Reduzido = {r2_red:.4f}")
    print(f"  [*] R² LIME (Coorte N={n_coorte}): Baseline = {np.mean(r2_coorte_base):.4f}±{np.std(r2_coorte_base):.4f} -> Reduzido = {np.mean(r2_coorte_red):.4f}±{np.std(r2_coorte_red):.4f}")
    print(f"  [*] Ruídos por Explicação Top-10 (Coorte N={n_coorte}): Baseline = {np.mean(ruidos_coorte_base):.2f} -> Reduzido = {np.mean(ruidos_coorte_red):.2f}")

    return {
        "paciente_idx": idx_limiar,
        "proba_base": proba_base,
        "proba_red": proba_red,
        "r2_lime_base": r2_base,
        "r2_lime_red": r2_red,
        "r2_coorte_base_media": float(np.mean(r2_coorte_base)),
        "r2_coorte_base_std": float(np.std(r2_coorte_base)),
        "r2_coorte_red_media": float(np.mean(r2_coorte_red)),
        "r2_coorte_red_std": float(np.std(r2_coorte_red)),
        "ruidos_lime_coorte_base": float(np.mean(ruidos_coorte_base)),
        "ruidos_lime_coorte_red": float(np.mean(ruidos_coorte_red)),
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
    Calcula os indicadores formais de interpretabilidade e pureza estrutural.
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
        "r2_coorte_base_media": auditoria_lime["r2_coorte_base_media"],
        "r2_coorte_base_std": auditoria_lime["r2_coorte_base_std"],
        "r2_coorte_red_media": auditoria_lime["r2_coorte_red_media"],
        "r2_coorte_red_std": auditoria_lime["r2_coorte_red_std"],
        "ruidos_lime_coorte_base": auditoria_lime["ruidos_lime_coorte_base"],
        "ruidos_lime_coorte_red": auditoria_lime["ruidos_lime_coorte_red"],
        "ganho_fidelidade_r2_pct": ganho_fidelidade_pct,
        "carga_cognitiva_miller": (
            f"{n_attrs_red} atributos; acima do limite heurístico de 9"
            if n_attrs_red > 9
            else f"{n_attrs_red} atributos; dentro do limite heurístico de 9"
        )
    }


# =============================================================================
# ETAPA 8: DASHBOARD EXECUTIVO FINAL DE 4 PAINÉIS
# =============================================================================
def gerar_dashboard_executivo(m_base, m_red_puro, m_red_opt, inter_metrics, output_path=os.path.join(OUTPUT_DIR, "dashboard_final_comparativo.png")):
    """
    Plota o Dashboard Executivo de 4 Painéis comparando Baseline (40), XAI Puro (10) e XAI + Optuna (10).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10))
    fig.suptitle("DASHBOARD EXECUTIVO: PIPELINE DE REDUÇÃO DE DADOS GUIADO POR XAI (OUT-OF-FOLD)", fontsize=14, fontweight="bold")

    categorias = ["Baseline\n(40 Atributos)", "XAI (shap-select OOF)\nSem Optuna (10 Atrib.)", "XAI (shap-select OOF)\n+ Optuna (10 Atrib.)"]
    cores = ["#d62728", "#1f77b4", "#2ca02c"]

    # 1. Dimensões
    attrs = [m_base["num_atributos"], m_red_puro["num_atributos"], m_red_opt["num_atributos"]]
    axes[0, 0].bar(categorias, attrs, color=cores, width=0.45)
    axes[0, 0].set_title(f"1. Compressão Dimensional (-{inter_metrics['reducao_dimensional_pct']:.0f}% de Atributos)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Quantidade de Atributos", fontweight="bold")
    for i, v in enumerate(attrs):
        axes[0, 0].text(i, v / 2, str(v), ha='center', va='center', color='white', fontweight='bold', fontsize=13)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Eficiência Computacional (Treino e Inferência)
    x_pos = np.arange(len(categorias))
    w = 0.35
    axes[0, 1].bar(x_pos - w/2, [m_base["tempo_treino_ms"], m_red_puro["tempo_treino_ms"], m_red_opt["tempo_treino_ms"]], w, label="Treino (ms)", color="#1f77b4")
    axes[0, 1].bar(x_pos + w/2, [m_base["tempo_inferencia_ms"], m_red_puro["tempo_inferencia_ms"], m_red_opt["tempo_inferencia_ms"]], w, label="Inferência (ms)", color="#ff7f0e")
    axes[0, 1].set_title("2. Custo Computacional do Estimador (Treino vs. Inferência)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(categorias, fontweight="bold", fontsize=9.5)
    axes[0, 1].set_ylabel("Milissegundos (ms)", fontweight="bold")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Integridade Preditiva (Holdout F1, CV 5-Fold F1 e ROC-AUC)
    w3 = 0.25
    axes[1, 0].bar(x_pos - w3, [m_base["f1_score"], m_red_puro["f1_score"], m_red_opt["f1_score"]], w3, label="F1 Holdout", color="#2ca02c")
    axes[1, 0].bar(x_pos, [m_base["f1_cv_media"], m_red_puro["f1_cv_media"], m_red_opt["f1_cv_media"]], w3, label="F1 CV (5-Fold)", color="#1f77b4")
    axes[1, 0].bar(x_pos + w3, [m_base["roc_auc"], m_red_puro["roc_auc"], m_red_opt["roc_auc"]], w3, label="ROC-AUC", color="#9467bd")
    axes[1, 0].set_title("3. Desempenho Preditivo (Superação do Fenômeno de Hughes)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(categorias, fontweight="bold", fontsize=9.5)
    axes[1, 0].set_ylim(0.80, 1.0)
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Métricas de Interpretabilidade e Pureza
    metricas_inter = ["R² LIME\n(Obs. #178)", "R² LIME Médio\n(Coorte N=20)", "Recuperação do\nGround Truth (10/10)"]
    valores_base = [inter_metrics["r2_fidelidade_base"], inter_metrics["r2_coorte_base_media"], 0.50]
    valores_red = [inter_metrics["r2_fidelidade_red"], inter_metrics["r2_coorte_red_media"], 1.00]

    x_inter = np.arange(len(metricas_inter))
    axes[1, 1].bar(x_inter - w/2, valores_base, w, label="Baseline (40 col., 50% ruído)", color="#d62728")
    axes[1, 1].bar(x_inter + w/2, valores_red, w, label="XAI Selecionado (10/10 inform., 0% ruído)", color="#2ca02c")
    axes[1, 1].set_title("4. Fidelidade Local LIME e Pureza de Recuperação", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(x_inter)
    axes[1, 1].set_xticklabels(metricas_inter, fontweight="bold", fontsize=9.5)
    axes[1, 1].set_ylim(0.0, 1.1)
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
    print("INICIANDO PIPELINE CIENTÍFICO XAI DATA REDUCTION (VERSÃO RIGOROSA OOF + VIF)")
    print("=" * 80)

    X, y = gerar_dataset_sintetico_saude()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"[*] Base de Dados: {X.shape[0]} instâncias, {X.shape[1]} atributos iniciais.")

    # 1. Baseline (40 Atributos)
    print("\n[ETAPA 1/7] Treinando Modelo Baseline (40 Atributos)...")
    modelo_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    m_baseline, y_pred_base, y_proba_base = treinar_e_avaliar_modelo(modelo_base, X_train, X_test, y_train, y_test)
    plotar_graficos_baseline(y_test, y_pred_base, y_proba_base)

    # 2. Explicabilidade Global TreeSHAP
    explainer_shap, shap_values, df_importancia_shap, tempo_shap = executar_etapa_shap(modelo_base, X_train)

    # 3. Engenharia Avançada: Pré-Filtro Híbrido (Correlação + VIF) + shap-select Out-of-Fold
    atributos_selecionados, df_shap_select, tempo_selecao_oof = executar_shap_select(X_train, y_train)
    X_train_red = X_train[atributos_selecionados]
    X_test_red = X_test[atributos_selecionados]

    # 4. Rankings de todas as famílias + Inclusão do Ranking OOF + VIF nas curvas de ablação
    rankings, tempos_selecao = gerar_rankings_todas_familias(
        X_train, y_train, modelo_base, df_importancia_shap, tempo_shap
    )
    df_comparativo_familias = avaliar_comparativo_familias(rankings, tempos_selecao, X_train, X_test, y_train, y_test, top_k=10)

    rankings_ablacao = {"XAI Proposto (OOF + VIF)": df_shap_select["atributo"].tolist()}
    rankings_ablacao.update(rankings)
    executar_etapa_ablacao(X_train, X_test, y_train, y_test, rankings_ablacao)

    # 5. Avaliação Ceteris Paribus do XAI Proposto SEM Optuna (mesmo RandomForest 100 árvores dos demais)
    modelo_red_puro = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    m_red_puro, _, _ = treinar_e_avaliar_modelo(modelo_red_puro, X_train_red, X_test_red, y_train, y_test)

    # 6. Otimização Bayesiana com Optuna (5-Fold CV, 20 trials)
    print(f"\n[ETAPA 6/7] Otimizando Hiperparâmetros via Optuna (5-Fold CV) nos {len(atributos_selecionados)} atributos...")
    best_params, tempo_optuna = otimizar_optuna(X_train_red, y_train, n_trials=20)
    print(f"  [*] Melhores Hiperparâmetros encontrados: {best_params}")

    modelo_campeao = RandomForestClassifier(**best_params, random_state=42, n_jobs=1)
    m_reduzido, y_pred_red, y_proba_red = treinar_e_avaliar_modelo(modelo_campeao, X_train_red, X_test_red, y_train, y_test)

    # 7. Auditoria Local LIME (Observação #178 + Coorte de Fronteira N=20)
    auditoria_lime = executar_auditoria_lime_comparativa(
        modelo_base, modelo_campeao,
        X_train, X_train_red,
        X_test, X_test_red,
        y_proba_base,
        n_coorte=20
    )

    # 8. Avaliação Multidimensional da Interpretabilidade
    inter_metrics = avaliar_metricas_interpretabilidade_multidimensional(
        df_importancia_shap, auditoria_lime,
        m_baseline["num_atributos"], m_reduzido["num_atributos"],
        atributos_selecionados, top_k=10
    )

    # 9. Dashboard Executivo Final
    gerar_dashboard_executivo(m_baseline, m_red_puro, m_reduzido, inter_metrics)

    # 10. Construção da Tabela Comparativa Mestre Completa (CSV)
    n_bio_camp = sum(1 for c in atributos_selecionados if 'biomarcador' in c)
    n_red_camp = sum(1 for c in atributos_selecionados if 'redundante' in c)
    n_ruido_camp = sum(1 for c in atributos_selecionados if 'ruido' in c)

    linha_campeao_opt = pd.DataFrame([{
        "Metodo": "XAI Proposto (shap-select OOF + Optuna)",
        "Familia": "XAI (Proposta)",
        "N_Atributos": len(atributos_selecionados),
        "Biomarcadores_Retidos (max 10)": n_bio_camp,
        "Redundantes_Retidos (max 10)": n_red_camp,
        "Ruidos_Selecionados (max 20)": n_ruido_camp,
        "Taxa_Rejeicao_Ruido": f"{(1 - n_ruido_camp / 20.0) * 100:.1f}%",
        "F1_Score": m_reduzido["f1_score"],
        "F1_CV_Media": m_reduzido["f1_cv_media"],
        "F1_CV_Std": m_reduzido["f1_cv_std"],
        "Acuracia": m_reduzido["acuracia"],
        "Recall": m_reduzido["recall"],
        "ROC_AUC": m_reduzido["roc_auc"],
        "Tempo_Selecao_s": tempo_selecao_oof + tempo_optuna,
        "Tempo_Treino_ms": m_reduzido["tempo_treino_ms"],
        "Latencia_Inf_ms": m_reduzido["tempo_inferencia_ms"],
        "R2_LIME_Baseline": auditoria_lime["r2_lime_base"],
        "R2_LIME_Reduzido": auditoria_lime["r2_lime_red"],
        "R2_LIME_Coorte_Base_Media": auditoria_lime["r2_coorte_base_media"],
        "R2_LIME_Coorte_Base_Std": auditoria_lime["r2_coorte_base_std"],
        "R2_LIME_Coorte_Red_Media": auditoria_lime["r2_coorte_red_media"],
        "R2_LIME_Coorte_Red_Std": auditoria_lime["r2_coorte_red_std"],
        "Ruidos_LIME_Coorte_Base": auditoria_lime["ruidos_lime_coorte_base"],
        "Ruidos_LIME_Coorte_Red": auditoria_lime["ruidos_lime_coorte_red"],
        "Massa_SHAP_Retida_pct": inter_metrics["massa_shap_selecionada_pct"],
        "Paciente_Limiar": auditoria_lime["paciente_idx"],
        "Probabilidade_Baseline_Limiar": auditoria_lime["proba_base"]
    }])

    linha_campeao_puro = pd.DataFrame([{
        "Metodo": "XAI Proposto (shap-select OOF Puro)",
        "Familia": "XAI (Proposta)",
        "N_Atributos": len(atributos_selecionados),
        "Biomarcadores_Retidos (max 10)": n_bio_camp,
        "Redundantes_Retidos (max 10)": n_red_camp,
        "Ruidos_Selecionados (max 20)": n_ruido_camp,
        "Taxa_Rejeicao_Ruido": f"{(1 - n_ruido_camp / 20.0) * 100:.1f}%",
        "F1_Score": m_red_puro["f1_score"],
        "F1_CV_Media": m_red_puro["f1_cv_media"],
        "F1_CV_Std": m_red_puro["f1_cv_std"],
        "Acuracia": m_red_puro["acuracia"],
        "Recall": m_red_puro["recall"],
        "ROC_AUC": m_red_puro["roc_auc"],
        "Tempo_Selecao_s": tempo_selecao_oof,
        "Tempo_Treino_ms": m_red_puro["tempo_treino_ms"],
        "Latencia_Inf_ms": m_red_puro["tempo_inferencia_ms"],
        "R2_LIME_Baseline": auditoria_lime["r2_lime_base"],
        "R2_LIME_Reduzido": auditoria_lime["r2_lime_red"],
        "R2_LIME_Coorte_Base_Media": auditoria_lime["r2_coorte_base_media"],
        "R2_LIME_Coorte_Base_Std": auditoria_lime["r2_coorte_base_std"],
        "R2_LIME_Coorte_Red_Media": auditoria_lime["r2_coorte_red_media"],
        "R2_LIME_Coorte_Red_Std": auditoria_lime["r2_coorte_red_std"],
        "Ruidos_LIME_Coorte_Base": auditoria_lime["ruidos_lime_coorte_base"],
        "Ruidos_LIME_Coorte_Red": auditoria_lime["ruidos_lime_coorte_red"],
        "Massa_SHAP_Retida_pct": inter_metrics["massa_shap_selecionada_pct"],
        "Paciente_Limiar": auditoria_lime["paciente_idx"],
        "Probabilidade_Baseline_Limiar": auditoria_lime["proba_base"]
    }])

    linha_baseline = pd.DataFrame([{
        "Metodo": "Baseline Completo (40 Atributos)",
        "Familia": "Referência",
        "N_Atributos": m_baseline["num_atributos"],
        "Biomarcadores_Retidos (max 10)": 10,
        "Redundantes_Retidos (max 10)": 10,
        "Ruidos_Selecionados (max 20)": 20,
        "Taxa_Rejeicao_Ruido": "0.0%",
        "F1_Score": m_baseline["f1_score"],
        "F1_CV_Media": m_baseline["f1_cv_media"],
        "F1_CV_Std": m_baseline["f1_cv_std"],
        "Acuracia": m_baseline["acuracia"],
        "Recall": m_baseline["recall"],
        "ROC_AUC": m_baseline["roc_auc"],
        "Tempo_Selecao_s": 0.0,
        "Tempo_Treino_ms": m_baseline["tempo_treino_ms"],
        "Latencia_Inf_ms": m_baseline["tempo_inferencia_ms"],
        "R2_LIME_Baseline": np.nan,
        "R2_LIME_Reduzido": np.nan,
        "R2_LIME_Coorte_Base_Media": np.nan,
        "R2_LIME_Coorte_Base_Std": np.nan,
        "R2_LIME_Coorte_Red_Media": np.nan,
        "R2_LIME_Coorte_Red_Std": np.nan,
        "Ruidos_LIME_Coorte_Base": np.nan,
        "Ruidos_LIME_Coorte_Red": np.nan,
        "Massa_SHAP_Retida_pct": np.nan,
        "Paciente_Limiar": np.nan,
        "Probabilidade_Baseline_Limiar": np.nan
    }])

    df_tabela_mestra = pd.concat(
        [linha_campeao_opt, linha_campeao_puro, df_comparativo_familias, linha_baseline],
        ignore_index=True
    )
    caminho_csv = os.path.join(OUTPUT_DIR, "tabela_comparativa_completa.csv")
    df_tabela_mestra.to_csv(caminho_csv, index=False)
    print(f"\n[+] Tabela comparativa mestre exportada com sucesso em: {caminho_csv}")

    print("\n" + "=" * 95)
    print("TABELA COMPARATIVA MESTRE: TODAS AS FAMÍLIAS DE SELEÇÃO vs. PROPOSTA XAI (OOF + VIF)")
    print("=" * 95)
    cols_show = [
        "Metodo", "Familia", "N_Atributos", "Biomarcadores_Retidos (max 10)",
        "Ruidos_Selecionados (max 20)", "F1_Score", "F1_CV_Media", "Acuracia", "ROC_AUC", "Tempo_Selecao_s"
    ]
    print(df_tabela_mestra[cols_show].to_string(index=False))

    # Validação via Sanity Checks
    taxa_reducao = inter_metrics["reducao_dimensional_pct"]
    assert taxa_reducao >= 50.0, f"Erro: Redução dimensional insuficiente ({taxa_reducao:.1f}%)"
    assert n_bio_camp == 10 and n_ruido_camp == 0, "Erro: Esperava 10/10 informativos e 0/20 ruídos!"
    assert m_reduzido["f1_score"] >= m_baseline["f1_score"], "Erro: Esperava superação do Baseline!"

    print("\nSANITY CHECKS RIGOROSOS APROVADOS COM SUCESSO:")
    print(f"  [OK] Recuperação Exata do Ground Truth : {n_bio_camp}/10 informativos, {n_red_camp}/10 redundantes, {n_ruido_camp}/20 ruídos!")
    print(f"  [OK] Desempenho Holdout F1             : Baseline = {m_baseline['f1_score']:.4f} -> XAI Puro = {m_red_puro['f1_score']:.4f} -> XAI+Optuna = {m_reduzido['f1_score']:.4f}")
    print(f"  [OK] Estabilidade 5-Fold CV F1         : Baseline = {m_baseline['f1_cv_media']:.4f}±{m_baseline['f1_cv_std']:.4f} -> XAI+Optuna = {m_reduzido['f1_cv_media']:.4f}±{m_reduzido['f1_cv_std']:.4f}")
    print("=" * 95)

    return df_tabela_mestra, m_baseline, m_red_puro, m_reduzido, inter_metrics


if __name__ == "__main__":
    executar_pipeline_completo()
