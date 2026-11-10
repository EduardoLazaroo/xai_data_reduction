"""
===================================================================================
GERADOR DA APRESENTAÇÃO DO PROJETO (PDF WIDESCREEN 16:9 — 15 MINUTOS)
===================================================================================
Projeto: Redução de Dimensionalidade Guiada por Explicabilidade (XAI)
Instituição: Centro Universitário Eurípides de Marília (UNIVEM)
Autor: Eduardo Lázaro Roesler de Oliveira
Entrega: 10/11 — Tempo: 15 min (12 min apresentação + 3 min perguntas)

Estrutura estritamente alinhada às exigências da disciplina e às melhorias metodológicas:
  Slide 01: Capa Institucional e Identificação do Projeto
  Slide 02: Introdução — Contextualização, Motivação e Problema Investigado
  Slide 03: Introdução — Questão de Pesquisa, Hipóteses e Objetivos
  Slide 04: Trabalhos Relacionados — Seleção Clássica vs. XAI (e Fragilidades In-Sample)
  Slide 05: Metodologia (Parte 1) — Dados Sintéticos Controlados e Baseline (Holdout + 5-Fold CV)
  Slide 06: Metodologia (Parte 2) — Pipeline Proposto (Pré-Filtro VIF + shap-select OOF + Optuna)
  Slide 07: Experimentos e Resultados — Benchmark Multi-Famílias (Holdout e 5-Fold CV)
  Slide 08: Experimentos e Resultados — Estudo de Ablação Progressiva e Custo Computacional
  Slide 09: Experimentos e Discussão — Auditoria Local LIME (Obs. #178 + Coorte N=20) e Dashboard
  Slide 10: Conclusão — Resposta à Questão de Pesquisa, Achados e Limitações
  Slide 11: Referências Bibliográficas e Agradecimentos (Abertura para Perguntas)
===================================================================================
"""

import os
import csv
import shutil
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.image as mpimg

BASE_DIR = r"c:\Users\eduar\projetos\xai_data_reduction"
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PDF_OUT = os.path.join(BASE_DIR, "apresentacao_projeto_xai.pdf")
DOCS_PDF_OUT = os.path.join(BASE_DIR, "docs", "apresentacao_projeto_xai.pdf")

NAVY = "#102C57"
BLUE = "#1D4ED8"
SLATE = "#334155"
DARK_TEXT = "#1E293B"
MUTED_TEXT = "#64748B"
LIGHT_BG = "#F8FAFC"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#CBD5E1"
EMERALD_BG = "#ECFDF5"
EMERALD_BORDER = "#10B981"
EMERALD_TEXT = "#065F46"
AMBER_BG = "#FFFBEB"
AMBER_BORDER = "#F59E0B"
AMBER_TEXT = "#92400E"
BLUE_BG = "#EFF6FF"
BLUE_BORDER = "#3B82F6"


def load_benchmark_rows():
    path = os.path.join(ASSETS_DIR, "tabela_comparativa_completa.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {row["Metodo"]: row for row in csv.DictReader(f)}


def fmt_dec(val, places=4):
    return f"{float(val):.{places}f}".replace(".", ",")


def create_slide(title=None, subtitle=None, section_tag=None, slide_num=1, total_slides=11, time_hint=""):
    fig = plt.figure(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor(LIGHT_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    if title is not None:
        header_rect = Rectangle((0, 7.85), 16, 1.15, facecolor=NAVY, edgecolor="none", zorder=1)
        ax.add_patch(header_rect)
        accent_bar = Rectangle((0, 7.78), 16, 0.07, facecolor="#3B82F6", edgecolor="none", zorder=2)
        ax.add_patch(accent_bar)

        if section_tag:
            ax.text(
                0.65, 8.66, section_tag.upper(),
                fontsize=9.5, fontweight="bold", color="#93C5FD",
                family="sans-serif", zorder=3
            )

        ax.text(
            0.65, 8.20, title,
            fontsize=20, fontweight="bold", color="#FFFFFF",
            family="sans-serif", zorder=3
        )

        if subtitle:
            ax.text(
                0.65, 7.93, subtitle,
                fontsize=10.5, style="italic", color="#E2E8F0",
                family="sans-serif", zorder=3
            )

        footer_line = Rectangle((0.65, 0.52), 14.7, 0.015, facecolor="#CBD5E1", edgecolor="none", zorder=1)
        ax.add_patch(footer_line)
        ax.text(
            0.65, 0.25,
            "Eduardo Lázaro Roesler de Oliveira  |  UNIVEM — Ciência da Computação  |  Redução de Dados Guiada por XAI (OOF + VIF)",
            fontsize=8.5, color=MUTED_TEXT, family="sans-serif"
        )
        right_label = f"Tempo sugerido: {time_hint}   •   Slide {slide_num}/{total_slides}" if time_hint else f"Slide {slide_num}/{total_slides}"
        ax.text(
            15.35, 0.25, right_label,
            fontsize=8.5, fontweight="bold", color=NAVY,
            ha="right", family="sans-serif"
        )

    return fig, ax


def draw_card(ax, x, y, w, h, title=None, bg=CARD_BG, border=CARD_BORDER, title_color=NAVY, header_bg=None, lw=1.2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=bg, edgecolor=border, linewidth=lw, zorder=2
    )
    ax.add_patch(box)

    if title:
        if header_bg:
            hdr = FancyBboxPatch(
                (x, y + h - 0.52), w, 0.52,
                boxstyle="round,pad=0.02,rounding_size=0.12",
                facecolor=header_bg, edgecolor=border, linewidth=lw, zorder=3
            )
            ax.add_patch(hdr)
            ax.text(
                x + 0.22, y + h - 0.32, title,
                fontsize=12, fontweight="bold", color="#FFFFFF",
                family="sans-serif", zorder=4
            )
        else:
            ax.text(
                x + 0.22, y + h - 0.36, title,
                fontsize=12.5, fontweight="bold", color=title_color,
                family="sans-serif", zorder=4
            )
            sep = Rectangle((x + 0.20, y + h - 0.50), w - 0.40, 0.02, facecolor=border, edgecolor="none", zorder=3)
            ax.add_patch(sep)


def add_bullet_list(ax, x, y_start, items, width_chars=60, fontsize=9.8, line_gap=0.22, item_gap=0.14, bullet_color=BLUE, text_color=DARK_TEXT):
    curr_y = y_start
    for item in items:
        lines = textwrap.wrap(item, width=width_chars)
        ax.text(x, curr_y, "▸", fontsize=fontsize + 1, fontweight="bold", color=bullet_color, family="sans-serif", zorder=4)
        for line in lines:
            ax.text(x + 0.24, curr_y, line, fontsize=fontsize, color=text_color, family="sans-serif", zorder=4)
            curr_y -= line_gap
        curr_y -= item_gap
    return curr_y


def add_image_panel(fig, img_filename, rect_coords, caption=None):
    img_path = os.path.join(ASSETS_DIR, img_filename)
    if not os.path.exists(img_path):
        return
    img = mpimg.imread(img_path)
    ax_img = fig.add_axes(rect_coords, zorder=5)
    ax_img.imshow(img)
    ax_img.axis("off")
    if caption:
        fig.text(
            rect_coords[0] + rect_coords[2] / 2.0,
            rect_coords[1] - 0.018,
            caption,
            ha="center", va="top", fontsize=8.5, style="italic", color=SLATE, family="sans-serif"
        )


def build_presentation():
    benchmark = load_benchmark_rows()
    proposed = benchmark["XAI Proposto (shap-select OOF + Optuna)"]
    prop_puro = benchmark["XAI Proposto (shap-select OOF Puro)"]
    baseline = benchmark["Baseline Completo (40 Atributos)"]

    with PdfPages(PDF_OUT) as pdf:
        # =====================================================================
        # SLIDE 1: CAPA INSTITUCIONAL
        # =====================================================================
        fig, ax = create_slide(title=None)
        bg_rect = Rectangle((0, 0), 16, 9, facecolor=NAVY, edgecolor="none")
        ax.add_patch(bg_rect)
        side_stripe = Rectangle((0, 0), 0.45, 9, facecolor="#3B82F6", edgecolor="none")
        ax.add_patch(side_stripe)
        top_badge = FancyBboxPatch(
            (1.2, 7.35), 6.8, 0.50,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="#1E3A8A", edgecolor="#60A5FA", linewidth=1.2
        )
        ax.add_patch(top_badge)
        ax.text(
            1.45, 7.53,
            "DEFESA E APRESENTAÇÃO DE PROJETO DE PESQUISA  •  15 MINUTOS",
            fontsize=10, fontweight="bold", color="#93C5FD", family="sans-serif"
        )

        ax.text(
            1.2, 6.25,
            "Redução de Dimensionalidade Guiada por\nExplicabilidade (XAI) em Machine Learning:",
            fontsize=28, fontweight="bold", color="#FFFFFF", family="sans-serif", linespacing=1.25
        )
        ax.text(
            1.2, 5.15,
            "Seleção Out-of-Fold (shap-select + VIF), Benchmark Multi-Famílias e Auditoria Local LIME",
            fontsize=15.0, color="#BFDBFE", style="italic", family="sans-serif"
        )

        q_box = FancyBboxPatch(
            (1.2, 3.20), 13.6, 1.50,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#172554", edgecolor="#3B82F6", linewidth=1.4
        )
        ax.add_patch(q_box)
        ax.text(
            1.5, 4.35, "QUESTÃO DE PESQUISA NORTEADORA:",
            fontsize=10.5, fontweight="bold", color="#60A5FA", family="sans-serif"
        )
        q_text = (
            "\"Em um benchmark sintético controlado, como uma seleção de atributos informada por pré-filtragem VIF e TreeSHAP Out-of-Fold "
            "(shap-select) se compara a métodos filter, wrapper e embedded em recuperação do subespaço informativo, rejeição de ruído, "
            "desempenho preditivo e custo computacional, e o que a auditoria LIME na fronteira de decisão permite concluir sobre fidelidade local?\""
        )
        for idx, line in enumerate(textwrap.wrap(q_text, width=108)):
            ax.text(1.5, 3.98 - idx * 0.27, line, fontsize=10.8, style="italic", color="#F8FAFC", family="sans-serif")

        ax.text(1.2, 2.25, "Autor: Eduardo Lázaro Roesler de Oliveira", fontsize=13.5, fontweight="bold", color="#FFFFFF", family="sans-serif")
        ax.text(1.2, 1.82, "Instituição: Centro Universitário Eurípides de Marília (UNIVEM) — Ciência da Computação", fontsize=11.5, color="#CBD5E1", family="sans-serif")
        ax.text(1.2, 1.42, "Entregáveis: Apresentação (PDF)  •  Implementação Reprodutível (Python)  •  Artigo Científico IEEE (3 Páginas)", fontsize=10.5, color="#93C5FD", family="sans-serif")
        ax.text(1.2, 0.75, "Nota metodológica: Benchmark sintético controlado (make_classification); nomes de variáveis identificam o ground truth experimental.", fontsize=9.2, style="italic", color="#94A3B8", family="sans-serif")
        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SLIDE 2: INTRODUÇÃO — CONTEXTUALIZAÇÃO, MOTIVAÇÃO E PROBLEMA
        # =====================================================================
        fig, ax = create_slide(
            title="Contextualização, Motivação e Problema de Pesquisa",
            subtitle="Alta dimensionalidade, colinearidade multivariada e armadilhas na seleção via XAI",
            section_tag="1. Introdução",
            slide_num=2, time_hint="0:00 – 1:30"
        )

        draw_card(ax, 0.65, 3.70, 7.15, 3.80, title="Contextualização e Motivação", border=BLUE_BORDER)
        add_bullet_list(ax, 0.90, 6.82, [
            "Expansão dimensional: bases tabulares acumulam variáveis redundantes (combinações lineares) e ruído estocástico.",
            "Fenômeno de Hughes (Mal da Dimensionalidade): com amostra fixa, atributos irrelevantes diluem a densidade amostral, degradam o F1-Score e elevam o custo de treino.",
            "Dilema Caixa-Preta vs. Poda Estrutural: ensembles como Random Forest são robustos, mas distribuem importância entre dezenas de colunas colineares.",
            "Motivação: transformar atribuições de Shapley (TreeSHAP) em um seletor estatístico rigoroso, livre de cortes Top-K arbitrários."
        ], width_chars=62, fontsize=9.8, line_gap=0.22, item_gap=0.14)

        draw_card(ax, 8.20, 3.70, 7.15, 3.80, title="Dois Problemas Críticos Tratados neste Projeto", border=AMBER_BORDER, bg=AMBER_BG, title_color=AMBER_TEXT)
        add_bullet_list(ax, 8.45, 6.82, [
            "1. Colinearidade Multivariada Oculta: variáveis formadas por combinação linear de múltiplos atributos têm correlação par-a-par moderada (|r| < 0,85), escapando de filtros simples e roubando massa SHAP dos atributos reais.",
            "2. Vazamento Interno (In-Sample Overfitting) no shap-select: calcular SHAP nos mesmos dados onde a floresta foi treinada infla artificialmente os p-valores do ruído (memorização).",
            "Solução Proposta: Pré-filtragem multivariada via VIF (≤ 10) acoplada ao cálculo TreeSHAP Out-of-Fold (3-Fold OOF) antes da regressão de p-valor."
        ], width_chars=62, fontsize=9.7, line_gap=0.22, item_gap=0.15, bullet_color=AMBER_TEXT, text_color=DARK_TEXT)

        draw_card(ax, 0.65, 0.80, 14.70, 2.65, title="Estrutura Exata do Ground Truth Sintético Investigado (D = 40 Atributos, N = 2.000)", border=CARD_BORDER)
        boxes = [
            (1.00, "10 Informativos (25%)", "Geradores independentes\ndo sinal verdadeiro", "#DBEAFE", "#1E40AF"),
            (4.65, "10 Redundantes (25%)", "Combinações lineares\nmultivariadas (VIF → ∞)", "#FEF3C7", "#92400E"),
            (8.30, "20 Ruídos Puros (50%)", "Variáveis gaussianas\nsem relação com y", "#FEE2E2", "#991B1B"),
            (11.95, "Resultado Alcançado", "10/10 informativos retidos\n0/10 red. | 0/20 ruídos", "#D1FAE5", "#065F46")
        ]
        for bx, btitle, bdesc, bbg, bcol in boxes:
            bpatch = FancyBboxPatch((bx, 1.05), 3.10, 1.65, boxstyle="round,pad=0.02,rounding_size=0.10", facecolor=bbg, edgecolor=bcol, linewidth=1.3, zorder=3)
            ax.add_patch(bpatch)
            ax.text(bx + 1.55, 2.32, btitle, fontsize=11, fontweight="bold", color=bcol, ha="center", family="sans-serif", zorder=4)
            ax.text(bx + 1.55, 1.58, bdesc, fontsize=9.8, color=DARK_TEXT, ha="center", va="center", family="sans-serif", zorder=4)

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SLIDE 3: QUESTÃO DE PESQUISA, HIPÓTESES E OBJETIVOS
        # =====================================================================
        fig, ax = create_slide(
            title="Questão de Pesquisa, Hipóteses e Objetivos",
            subtitle="Desenho experimental orientado por hipóteses quantitativas auditáveis",
            section_tag="1. Introdução",
            slide_num=3, time_hint="1:30 – 3:00"
        )

        draw_card(ax, 0.65, 5.75, 14.70, 1.75, title="Questão Central de Pesquisa", bg=BLUE_BG, border=BLUE_BORDER, title_color=NAVY)
        q_lines = textwrap.wrap(
            "\"Em um benchmark sintético controlado, como uma seleção de atributos informada por pré-filtragem VIF e TreeSHAP Out-of-Fold "
            "(shap-select) se compara a métodos filter, wrapper e embedded em recuperação do subespaço informativo, rejeição de ruído, "
            "desempenho preditivo e custo computacional, e o que a auditoria LIME na fronteira de decisão permite concluir sobre fidelidade local?\"",
            width=114
        )
        for i, line in enumerate(q_lines):
            ax.text(0.95, 6.74 - i * 0.28, line, fontsize=11.0, fontweight="bold", style="italic", color=NAVY, family="sans-serif", zorder=4)

        hyps = [
            (
                0.65, "Hipótese H1 (Desempenho Preditivo)",
                "Eliminar ao menos 50% dos atributos mantendo ou superando o F1-Score do Baseline (queda máxima tolerada de 0,05) tanto no Holdout quanto na validação cruzada de 5 folds.",
                EMERALD_BG, EMERALD_BORDER, EMERALD_TEXT
            ),
            (
                5.65, "Hipótese H2 (Pureza e Custo de Seleção)",
                "Recuperar mais atributos informativos verdadeiros e rejeitar 100% do ruído sintético frente às famílias Filter, Wrapper e Embedded, avaliando o trade-off de tempo contra o RFE.",
                AMBER_BG, AMBER_BORDER, AMBER_TEXT
            ),
            (
                10.65, "Hipótese H3 (Fidelidade Local LIME)",
                "Na observação mais crítica (P = 50%) e na coorte de fronteira (N = 20), eliminar 100% da contaminação por ruído nas regras locais sem degradar o ajuste R² do LIME.",
                CARD_BG, BLUE_BORDER, BLUE
            )
        ]
        for hx, htitle, hdesc, hbg, hborder, hcol in hyps:
            draw_card(ax, hx, 3.05, 4.70, 2.45, title=htitle, bg=hbg, border=hborder, title_color=hcol)
            for idx, line in enumerate(textwrap.wrap(hdesc, width=42)):
                ax.text(hx + 0.22, 4.72 - idx * 0.26, line, fontsize=9.8, color=DARK_TEXT, family="sans-serif", zorder=4)

        draw_card(ax, 0.65, 0.80, 14.70, 2.00, title="Objetivo Geral e Objetivos Específicos", border=CARD_BORDER)
        add_bullet_list(ax, 0.90, 2.05, [
            "Objetivo Geral: Propor e validar um pipeline híbrido sem vazamento (Pré-Filtro VIF + TreeSHAP Out-of-Fold + shap-select com p < 0,05 + Optuna) para redução dimensional.",
            "Objetivos Específicos: (1) Comparar 9 configurações das 4 famílias em Holdout e 5-Fold CV (ceteris paribus e otimizado); (2) Executar ablação de 40 a 2 atributos; (3) Auditar o LIME em instância crítica e coorte de fronteira (N=20)."
        ], width_chars=116, fontsize=9.9, line_gap=0.24, item_gap=0.14)

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SLIDE 4: TRABALHOS RELACIONADOS
        # =====================================================================
        fig, ax = create_slide(
            title="Trabalhos Relacionados e Fundamentação Teórica",
            subtitle="Taxonomia clássica de seleção de atributos vs. seleção inferencial guiada por XAI",
            section_tag="2. Trabalhos Relacionados",
            slide_num=4, time_hint="3:00 – 4:30"
        )

        works = [
            (
                0.65, 3.85, 7.15, 3.65,
                "1. Taxonomia Clássica: Filter, Wrapper e Embedded",
                [
                    "Kohavi & John (1997) [2]: definiram as famílias Filter (independentes do modelo) e Wrapper (orientados por reajustes sucessivos do preditor).",
                    "Filtros (ANOVA F, Mutual Info): velozes (milissegundos), mas avaliam cada coluna isoladamente e retêm variáveis redundantes colineares.",
                    "Wrappers — Guyon et al. (2002) [3] (RFE): eliminam recursivamente atributos de menor peso; sofrem com colinearidade ao dividir importância no ensemble.",
                    "Embedded (Lasso L1 / Impureza MDI): seleção intrínseca durante o treino; vulneráveis a viés de impureza e instabilidade sob multivariáveis correlacionadas."
                ],
                BLUE_BORDER
            ),
            (
                8.20, 3.85, 7.15, 3.65,
                "2. Explicabilidade Formal: SHAP e LIME",
                [
                    "Lundberg & Lee (2017) [4] (SHAP / TreeSHAP): atribuições exatas baseadas nos axiomas da Teoria dos Jogos Cooperativos de Shapley.",
                    "Ribeiro et al. (2016) [5] (LIME): aproximação linear local ponderada por perturbações na vizinhança de uma instância específica.",
                    "Alvarez-Melis & Jaakkola (2018) [7]: provaram que explicadores locais por perturbação sofrem forte degradação e inclusão de ruído em alta dimensão."
                ],
                EMERALD_BORDER
            )
        ]
        for wx, wy, ww, wh, wtitle, witems, wborder in works:
            draw_card(ax, wx, wy, ww, wh, title=wtitle, border=wborder)
            add_bullet_list(ax, wx + 0.25, wy + wh - 0.68, witems, width_chars=63, fontsize=9.5, line_gap=0.22, item_gap=0.12)

        draw_card(ax, 0.65, 0.80, 14.70, 2.80, title="3. Seleção Guiada por XAI e o Avanço Metodológico deste Trabalho", bg=BLUE_BG, border=BLUE_BORDER)
        add_bullet_list(ax, 0.90, 2.92, [
            "Marcílio & Eler (2020) [6]: propuseram seleção via ranking SHAP, porém dependente de um número Top-K fixado manualmente pelo usuário e vulnerável a variáveis redundantes.",
            "Limitação do shap-select In-Sample: regredir y contra valores SHAP no mesmo conjunto de treino faz árvores profundas memorizarem os dados, gerando p < 0,05 falso-positivo.",
            "Contribuição deste Trabalho: combinação de Pré-Filtro VIF (≤ 10) + TreeSHAP Out-of-Fold (3-Fold OOF) + Regressão de significância (p < 0,05), recuperando 10/10 variáveis informativas automaticamente sem fixar K!"
        ], width_chars=120, fontsize=9.7, line_gap=0.23, item_gap=0.13)

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SLIDE 5: METODOLOGIA (PARTE 1) — DADOS E BASELINE
        # =====================================================================
        fig, ax = create_slide(
            title="Metodologia: Base Sintética Controlada e Modelo Baseline",
            subtitle="Protocolo experimental com Holdout estratificado (75/25) e Validação Cruzada de 5 Folds",
            section_tag="3. Metodologia",
            slide_num=5, time_hint="4:30 – 5:45"
        )

        draw_card(ax, 0.65, 0.85, 6.80, 6.65, title="Descrição dos Dados e Configuração do Baseline", border=BLUE_BORDER)
        add_bullet_list(ax, 0.90, 6.72, [
            "Justificativa do Benchmark Sintético: em bases reais, o conjunto verdadeiro de variáveis puramente ruidosas é desconhecido. Com make_classification (shuffle=False), auditamos com exatidão matemática a taxa de recuperação do ground truth.",
            "Dimensão: N = 2.000 observações e D = 40 atributos contínuos (10 informativos, 10 redundantes por combinação linear e 20 ruídos estocásticos).",
            "Ruído de Rótulo e Proporção: classes 60% / 40% com 3% de inversão aleatória de rótulos (flip_y = 0,03) para simular incerteza de fronteira.",
            "Protocolo Duplo de Avaliação: Holdout estratificado 75% treino (1.500) / 25% teste (500) + Validação Cruzada Estratificada de 5 Folds no treino (Média ± DP).",
            f"Desempenho Baseline (40 Atributos): F1 Holdout = {fmt_dec(baseline['F1_Score'])} | F1 CV (5-Fold) = {fmt_dec(baseline['F1_CV_Media'])} ± {fmt_dec(baseline['F1_CV_Std'])} | ROC-AUC = {fmt_dec(baseline['ROC_AUC'])} | Treino = {fmt_dec(baseline['Tempo_Treino_ms'], 1)} ms."
        ], width_chars=54, fontsize=9.8, line_gap=0.24, item_gap=0.16)

        draw_card(ax, 7.75, 0.85, 7.60, 6.65, title="Diagnóstico Experimental do Modelo Baseline (40 Atributos)", border=CARD_BORDER)
        add_image_panel(
            fig, "modulo1_baseline_metrics.png",
            [0.50, 0.16, 0.44, 0.56],
            caption="Figura gerada pelo pipeline: Matriz de Confusão e Curva ROC (AUC = 0,9655) do Baseline no Holdout."
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SLIDE 6: METODOLOGIA (PARTE 2) — PIPELINE PROPOSTO
        # =====================================================================
        fig, ax = create_slide(
            title="Metodologia: Pipeline Proposto (VIF + shap-select Out-of-Fold)",
            subtitle="Eliminação de colinearidade multivariada e seleção inferencial p < 0,05 sem vazamento in-sample",
            section_tag="3. Metodologia",
            slide_num=6, time_hint="5:45 – 7:00"
        )

        stages = [
            (
                0.65, "Etapa 1: Pré-Filtro (Pearson + VIF ≤ 10)",
                [
                    "VarianceThreshold (< 0,01) e Pearson (|r| > 0,90): reduz de 40 para 33 colunas.",
                    "VIF Iterativo (VIF = 1/(1-R²_j) ≤ 10): detecta e elimina todas as 10 combinações lineares redundantes.",
                    "Retém 30 colunas ortogonais (10 inform. + 20 ruídos)."
                ],
                BLUE_BG, BLUE_BORDER
            ),
            (
                5.65, "Etapa 2: TreeSHAP OOF + shap-select",
                [
                    "Calcula TreeSHAP em 3-Fold Stratified Out-of-Fold (OOF) sobre X_train_pre (zero memorização).",
                    "Regressão Logit de y_train sobre Φ_OOF: aprova apenas β > 0 e p-valor < 0,05 (sem corte Top-K!).",
                    "Seleciona exatamente os 10 informativos (10/10) e 0 ruídos!"
                ],
                EMERALD_BG, EMERALD_BORDER
            ),
            (
                10.65, "Etapa 3: Avaliação Ceteris Paribus e Optuna",
                [
                    "Avalia o subconjunto selecionado sob o mesmo RF padrão (100 árvores) dos demais métodos.",
                    "Aplica Optuna TPE (20 trials, 5-Fold CV) para refinar hiperparâmetros no espaço limpo de 10 atributos.",
                    f"Audita LIME na Obs. #{int(float(proposed['Paciente_Limiar']))} e na Coorte N=20."
                ],
                AMBER_BG, AMBER_BORDER
            )
        ]
        for sx, stitle, sitems, sbg, sborder in stages:
            draw_card(ax, sx, 4.85, 4.70, 2.65, title=stitle, bg=sbg, border=sborder)
            add_bullet_list(ax, sx + 0.20, 6.75, sitems, width_chars=42, fontsize=9.3, line_gap=0.22, item_gap=0.11)

        draw_card(ax, 0.65, 0.80, 7.20, 3.80, title="TreeSHAP no Baseline (Diluição com Redundantes em Laranja)", border=CARD_BORDER)
        add_image_panel(fig, "modulo2_shap_summary.png", [0.055, 0.11, 0.41, 0.31])

        draw_card(ax, 8.15, 0.80, 7.20, 3.80, title="shap-select Out-of-Fold: 10 Informativos Aprovados (Verde) vs. Ruído (Vermelho)", border=CARD_BORDER)
        add_image_panel(fig, "modulo5_shap_select_analysis.png", [0.525, 0.11, 0.41, 0.31])

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SLIDE 7: EXPERIMENTOS E RESULTADOS — BENCHMARK MULTI-FAMÍLIAS
        # =====================================================================
        fig, ax = create_slide(
            title="Resultados: Benchmark Multi-Famílias (Holdout e 5-Fold CV)",
            subtitle="Comparação justa ceteris paribus (sem Optuna) e otimizada (+ Optuna) frente às 4 famílias",
            section_tag="4. Experimentos, Resultados e Discussão",
            slide_num=7, time_hint="7:00 – 8:30"
        )

        draw_card(ax, 0.65, 2.75, 14.70, 4.75, title="Tabela I — Recuperação do Ground Truth, Desempenho Preditivo e Custo Computacional", border=BLUE_BORDER)

        headers = ["Método / Abordagem", "Família", "Inform.", "Redund.", "Ruído", "F1 Holdout", "F1 CV (5-Fold)", "Acurácia", "ROC-AUC", "Seleção (s)", "Treino"]
        col_x = [0.82, 4.35, 5.65, 6.45, 7.25, 8.15, 9.35, 10.95, 11.95, 13.00, 14.20]

        hdr_bg = Rectangle((0.78, 6.48), 14.44, 0.38, facecolor=NAVY, edgecolor="none", zorder=3)
        ax.add_patch(hdr_bg)
        for cx, htext in zip(col_x, headers):
            ax.text(cx, 6.61, htext, fontsize=8.6, fontweight="bold", color="#FFFFFF", family="sans-serif", zorder=4)

        method_order = [
            "XAI Proposto (shap-select OOF + Optuna)",
            "XAI Proposto (shap-select OOF Puro)",
            "Wrapper (RFE)",
            "Embedded (Lasso L1)",
            "Embedded (Tree MDI)",
            "Filter (ANOVA F)",
            "XAI (TreeSHAP)",
            "Filter (Mutual Info)",
            "Baseline Completo (40 Atributos)"
        ]

        y_row = 6.10
        for idx, mname in enumerate(method_order):
            r = benchmark[mname]
            is_prop = (idx in [0, 1])
            is_base = (mname == "Baseline Completo (40 Atributos)")
            row_bg_color = "#DCFCE7" if is_prop else ("#FEF3C7" if is_base else ("#F8FAFC" if idx % 2 == 1 else "#FFFFFF"))
            r_rect = Rectangle((0.78, y_row - 0.10), 14.44, 0.38, facecolor=row_bg_color, edgecolor="#E2E8F0", linewidth=0.6, zorder=3)
            ax.add_patch(r_rect)

            cv_txt = f"{fmt_dec(r['F1_CV_Media'], 4)} ± {fmt_dec(r['F1_CV_Std'], 3)}"
            vals = [
                mname,
                r["Familia"],
                f'{int(float(r["Biomarcadores_Retidos (max 10)"]))}/10',
                f'{int(float(r["Redundantes_Retidos (max 10)"]))}/10',
                f'{int(float(r["Ruidos_Selecionados (max 20)"]))}/20',
                fmt_dec(r["F1_Score"]),
                cv_txt,
                fmt_dec(r["Acuracia"]),
                fmt_dec(r["ROC_AUC"]),
                f'{fmt_dec(r["Tempo_Selecao_s"], 2)} s',
                f'{fmt_dec(r["Tempo_Treino_ms"], 1)} ms'
            ]
            fw = "bold" if (is_prop or is_base) else "normal"
            tcol = EMERALD_TEXT if is_prop else (AMBER_TEXT if is_base else DARK_TEXT)
            for cx, val in zip(col_x, vals):
                ax.text(cx, y_row + 0.03, val, fontsize=8.3, fontweight=fw, color=tcol, family="sans-serif", zorder=4)
            y_row -= 0.39

        draw_card(ax, 0.65, 0.78, 14.70, 1.80, title="Análise Crítica da Tabela I (Validação Rigorosa de H1 e H2)", bg=EMERALD_BG, border=EMERALD_BORDER, title_color=EMERALD_TEXT)
        add_bullet_list(ax, 0.90, 1.88, [
            f"H1 Superada (Ganho Preditivo com 75% de Redução): Mesmo SEM Optuna (ceteris paribus), o XAI Proposto atingiu F1 = {fmt_dec(prop_puro['F1_Score'])} (superando o Baseline de 40 atributos, {fmt_dec(baseline['F1_Score'])}, e todos os métodos clássicos). Com Optuna, saltou para F1 = {fmt_dec(proposed['F1_Score'])} e ROC-AUC = {fmt_dec(proposed['ROC_AUC'])}.",
            f"H2 Confirmada em Pureza (10/10 Informativos) e Trade-off de Custo: Único método a recuperar 10/10 informativos (0 redundantes e 0 ruídos); RFE e ANOVA pararam em 7/10 por reterem 3 redundantes. O tempo de seleção OOF foi {fmt_dec(prop_puro['Tempo_Selecao_s'], 2)} s vs. {fmt_dec(benchmark['Wrapper (RFE)']['Tempo_Selecao_s'], 2)} s do RFE."
        ], width_chars=118, fontsize=9.4, line_gap=0.22, item_gap=0.10, bullet_color=EMERALD_TEXT)

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SLIDE 8: ESTUDO DE ABLAÇÃO PROGRESSIVA
        # =====================================================================
        fig, ax = create_slide(
            title="Resultados: Estudo de Ablação Progressiva (40 a 2 Atributos)",
            subtitle="Comprovação visual do Fenômeno de Hughes e da superioridade da trajetória OOF + VIF",
            section_tag="4. Experimentos, Resultados e Discussão",
            slide_num=8, time_hint="8:30 – 9:30"
        )

        draw_card(ax, 0.65, 0.85, 9.20, 6.65, title="Curvas de Ablação Multivariada e Economia de Treinamento", border=CARD_BORDER)
        add_image_panel(
            fig, "modulo4_ablation_curves.png",
            [0.055, 0.14, 0.54, 0.59],
            caption="Figura 3 do Artigo: F1-Score no Holdout (esquerda) e Tempo de Treino (direita) sob poda de 40 até 2 atributos."
        )

        draw_card(ax, 10.10, 0.85, 5.25, 6.65, title="Principais Achados da Ablação", border=BLUE_BORDER)
        add_bullet_list(ax, 10.35, 6.72, [
            "Pico no Subespaço Verdadeiro (K = 10): a curva verde-escura (XAI Proposto OOF + VIF) atinge o pico de desempenho exatamente em K = 10 atributos, pois os 10 primeiros do ranking são os 10 informativos puros.",
            "Colapso abaixo de K < 8: quando a poda passa a remover atributos informativos reais (abaixo de 8 variáveis), todas as famílias sofrem queda acentuada de F1-Score.",
            f"Redução do Tempo de Treino do Estimador: treinar o Random Forest padrão com 10 atributos leva {fmt_dec(prop_puro['Tempo_Treino_ms'], 1)} ms contra {fmt_dec(baseline['Tempo_Treino_ms'], 1)} ms no Baseline.",
            "Custo Offline vs. Online: o tempo da seleção OOF ocorre apenas uma vez na engenharia; o modelo produtivo opera indefinidamente com 75% menos variáveis."
        ], width_chars=43, fontsize=9.7, line_gap=0.24, item_gap=0.18)

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SLIDE 9: AUDITORIA LOCAL LIME E DASHBOARD EXECUTIVO
        # =====================================================================
        obs_id = int(float(proposed['Paciente_Limiar']))
        fig, ax = create_slide(
            title="Discussão: Auditoria Local LIME (Instância e Coorte) e Dashboard",
            subtitle=f"Avaliação na Observação Limítrofe #{obs_id} (P = 50%) e na Coorte de Fronteira (N = 20)",
            section_tag="4. Experimentos, Resultados e Discussão",
            slide_num=9, time_hint="9:30 – 10:45"
        )

        draw_card(ax, 0.65, 3.20, 7.20, 4.30, title=f"Auditoria Local LIME (Observação #{obs_id} e Coorte N=20)", border=CARD_BORDER)
        add_image_panel(fig, "modulo3_lime_local.png", [0.055, 0.37, 0.41, 0.35])

        draw_card(ax, 8.15, 3.20, 7.20, 4.30, title="Dashboard Executivo Comparativo (Baseline vs. XAI Puro vs. Optuna)", border=CARD_BORDER)
        add_image_panel(fig, "dashboard_final_comparativo.png", [0.525, 0.37, 0.41, 0.35])

        draw_card(ax, 0.65, 0.80, 14.70, 2.15, title="Interpretação Rigorosa da Auditoria Local LIME (Verificação de H3)", bg=BLUE_BG, border=BLUE_BORDER)
        add_bullet_list(ax, 0.90, 2.22, [
            f"Fidelidade Local (R²) na Instância #{obs_id} e na Coorte (N=20): Na observação #{obs_id} (P = 50,0%), o R² do LIME subiu de {fmt_dec(proposed['R2_LIME_Baseline'])} para {fmt_dec(proposed['R2_LIME_Reduzido'])}. Na coorte das 20 observações mais incertas da fronteira, o R² médio foi de {fmt_dec(proposed['R2_LIME_Coorte_Base_Media'], 3)}±{fmt_dec(proposed['R2_LIME_Coorte_Base_Std'], 2)} (Baseline) e {fmt_dec(proposed['R2_LIME_Coorte_Red_Media'], 3)}±{fmt_dec(proposed['R2_LIME_Coorte_Red_Std'], 2)} (Reduzido), refletindo a forte não-linearidade do Random Forest na zona P ≈ 50%.",
            f"Eliminação de Atributos Não-Informativos nas Explicações Locais: No Baseline, as explicações Top-10 do LIME na coorte incluíram em média {fmt_dec(proposed['Ruidos_LIME_Coorte_Base'], 2)} regras não-informativas (redundantes/ruído) por instância; no modelo reduzido, essa contaminação caiu para {fmt_dec(proposed['Ruidos_LIME_Coorte_Red'], 2)} (100% informativos!)."
        ], width_chars=118, fontsize=9.4, line_gap=0.22, item_gap=0.11)

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SLIDE 10: CONCLUSÃO — RESPOSTA À QUESTÃO DE PESQUISA E LIMITAÇÕES
        # =====================================================================
        fig, ax = create_slide(
            title="Conclusão: Resposta à Questão de Pesquisa e Limitações",
            subtitle="Síntese das evidências quantitativas, julgamento das hipóteses e fronteiras epistemológicas",
            section_tag="5. Conclusão e Limitações",
            slide_num=10, time_hint="10:45 – 11:45"
        )

        draw_card(ax, 0.65, 3.65, 7.15, 3.85, title="Resposta Direta à Questão de Pesquisa", bg=EMERALD_BG, border=EMERALD_BORDER, title_color=EMERALD_TEXT)
        add_bullet_list(ax, 0.90, 6.82, [
            f"1. Desempenho Preditivo (H1 Confirmada com Superação): A poda de 75% (40 → 10 atributos) elevou o F1 no Holdout de {fmt_dec(baseline['F1_Score'])} para {fmt_dec(prop_puro['F1_Score'])} (sem Optuna) e {fmt_dec(proposed['F1_Score'])} (com Optuna), além de superar na validação cruzada de 5 folds ({fmt_dec(proposed['F1_CV_Media'])} vs. {fmt_dec(baseline['F1_CV_Media'])}).",
            f"2. Recuperação do Subespaço e Custo (H2 Confirmada em Pureza): Único método a atingir 10/10 informativos, 0/10 redundantes e 0/20 ruídos automaticamente via p < 0,05, com tempo de seleção de {fmt_dec(prop_puro['Tempo_Selecao_s'], 2)} s frente a {fmt_dec(benchmark['Wrapper (RFE)']['Tempo_Selecao_s'], 2)} s do RFE.",
            f"3. Auditoria Local LIME (H3 Confirmada): Elevou o R² na instância crítica ({fmt_dec(proposed['R2_LIME_Baseline'])} → {fmt_dec(proposed['R2_LIME_Reduzido'])}) e zerou a presença de ruído nas regras locais da coorte de fronteira (N=20)."
        ], width_chars=63, fontsize=9.5, line_gap=0.22, item_gap=0.13, bullet_color=EMERALD_TEXT)

        draw_card(ax, 8.20, 3.65, 7.15, 3.85, title="Limitações do Trabalho (Transparência Científica)", bg=AMBER_BG, border=AMBER_BORDER, title_color=AMBER_TEXT)
        add_bullet_list(ax, 8.45, 6.82, [
            "Benchmark Sintético Controlado: Os resultados provam a eficácia matemática do algoritmo sob ground truth conhecido, mas não substituem validação clínica ou em dados reais multimodais.",
            "Associação Preditiva ≠ Causalidade: Valores TreeSHAP e pesos LIME descrevem o comportamento funcional do estimador e não implicam relação causal física.",
            "Custo Computacional O(K · TLD²): O cálculo TreeSHAP em 3 folds OOF exige maior tempo inicial de seleção que filtros univariados e RFE, sendo indicado para seleção offline.",
            "Ausência de Testes Cognitivos com Humanos: A referência de Miller (7 ± 2) é teórica; não foram realizados experimentos com usuários."
        ], width_chars=63, fontsize=9.5, line_gap=0.21, item_gap=0.12, bullet_color=AMBER_TEXT)

        draw_card(ax, 0.65, 0.80, 14.70, 2.60, title="Trabalhos Futuros", border=CARD_BORDER)
        add_bullet_list(ax, 0.90, 2.72, [
            "Aplicação em Bases Reais de Alta Dimensionalidade: Estender o protocolo Pré-Filtro VIF + shap-select OOF a datasets públicos de genômica e bioinformática.",
            "Exploração de Explicadores Não-Lineares Locais: Comparar o substituto linear do LIME com explicadores baseados em âncoras (Anchors) e contrafactuais na fronteira P ≈ 50%.",
            "Estudo Experimental Centrado no Usuário: Avaliar empiricamente o impacto da redução de atributos na tomada de decisão de especialistas."
        ], width_chars=120, fontsize=9.8, line_gap=0.22, item_gap=0.12)

        pdf.savefig(fig)
        plt.close(fig)

        # =====================================================================
        # SLIDE 11: REFERÊNCIAS BIBLIOGRÁFICAS E ENCERRAMENTO
        # =====================================================================
        fig, ax = create_slide(
            title="Referências Bibliográficas",
            subtitle="Literatura fundamental que sustenta a metodologia, o benchmark e o artigo IEEE",
            section_tag="6. Referências",
            slide_num=11, time_hint="11:45 – 12:00 (+ 3 min Perguntas)"
        )

        draw_card(ax, 0.65, 2.25, 14.70, 5.25, title="Referências Citadas no Artigo e na Apresentação (Formato IEEE)", border=BLUE_BORDER)
        refs = [
            "[1] C. Molnar, Interpretable Machine Learning: A Guide for Making Black Box Models Explainable, 2nd ed. Leanpub, 2022.",
            "[2] R. Kohavi and G. H. John, \"Wrappers for feature subset selection,\" Artificial Intelligence, vol. 97, no. 1–2, pp. 273–324, 1997.",
            "[3] I. Guyon, J. Weston, S. Barnhill, and V. Vapnik, \"Gene selection for cancer classification using support vector machines,\" Machine Learning, vol. 46, no. 1–3, pp. 389–422, 2002.",
            "[4] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 4765–4774.",
            "[5] M. T. Ribeiro, S. Singh, and C. Guestrin, \"'Why should I trust you?': Explaining the predictions of any classifier,\" in Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD), 2016, pp. 1135–1144.",
            "[6] W. E. Marcílio and D. M. Eler, \"From explainability to model improvement: A survey on XAI-based feature selection,\" IEEE Trans. Knowl. Data Eng., vol. 34, no. 11, pp. 5500–5515, 2020.",
            "[7] D. Alvarez-Melis and T. S. Jaakkola, \"Towards robust interpretability with self-explaining neural networks,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 31, 2018, pp. 7775–7784.",
            "[8] G. A. Miller, \"The magical number seven, plus or minus two: Some limits on our capacity for processing information,\" Psychological Review, vol. 63, no. 2, pp. 81–97, 1956."
        ]
        curr_y = 6.80
        for ref in refs:
            lines = textwrap.wrap(ref, width=122)
            for idx, line in enumerate(lines):
                ax.text(0.95 + (0.28 if idx > 0 else 0), curr_y, line, fontsize=9.5, color=DARK_TEXT, family="sans-serif", zorder=4)
                curr_y -= 0.21
            curr_y -= 0.11

        q_banner = FancyBboxPatch(
            (0.65, 0.80), 14.70, 1.30,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=NAVY, edgecolor="#3B82F6", linewidth=1.5, zorder=3
        )
        ax.add_patch(q_banner)
        ax.text(
            8.0, 1.58,
            "Obrigado pela atenção!  •  Sessão aberta para arguição e perguntas (3 minutos)",
            fontsize=15, fontweight="bold", color="#FFFFFF", ha="center", family="sans-serif", zorder=4
        )
        ax.text(
            8.0, 1.10,
            "Eduardo Lázaro Roesler de Oliveira  —  eduardo.roesler@univem.edu.br  —  UNIVEM",
            fontsize=11, color="#93C5FD", ha="center", family="sans-serif", zorder=4
        )

        pdf.savefig(fig)
        plt.close(fig)

    os.makedirs(os.path.dirname(DOCS_PDF_OUT), exist_ok=True)
    shutil.copyfile(PDF_OUT, DOCS_PDF_OUT)
    print(f"[+] Apresentação PDF gerada com sucesso em: {PDF_OUT}")
    print(f"[+] Cópia sincronizada em: {DOCS_PDF_OUT}")


if __name__ == "__main__":
    build_presentation()
