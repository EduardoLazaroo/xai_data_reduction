"""
===================================================================================
GERADOR DA APRESENTAÇÃO EXECUTIVA EM POWERPOINT (.PPTX — MÁXIMO 10 MINUTOS)
===================================================================================
Projeto: Redução de Dimensionalidade Guiada por Explicabilidade (XAI)
Autor: Eduardo Lázaro Roesler de Oliveira
Instituição: UNESP — Programa de Pós-Graduação (Mestrado)

Objetivo deste script:
  Gerar uma apresentação limpa, visual e leve em PowerPoint (.pptx), com baixa
  densidade textual por slide e notas detalhadas do apresentador (script cronometrado
  para ~9 min 30 s, respeitando o teto de 10 minutos), seguindo a estrutura:
    1. Introdução (contextualização, motivação, problema, questão e objetivo)
    2. Trabalhos relacionados
    3. Metodologia (dados, técnicas e procedimento experimental)
    4. Experimentos, resultados e discussão
    5. Conclusão (principais resultados, resposta à questão e limitações)
    6. Referências bibliográficas
===================================================================================
"""

import csv
import os
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CSV_PATH = os.path.join(ASSETS_DIR, "tabela_comparativa_completa.csv")
PPTX_OUT = os.path.join(BASE_DIR, "apresentacao_projeto_xai_10min.pptx")

FONT_NAME = "Segoe UI"

# Paleta Corporativa/Acadêmica Limpa
NAVY = RGBColor(15, 41, 74)          # #0F294A
BLUE = RGBColor(37, 99, 235)         # #2563EB
LIGHT_BLUE = RGBColor(147, 197, 253) # #93C5FD
SLATE_DARK = RGBColor(30, 41, 59)    # #1E293B
SLATE_MUTED = RGBColor(100, 116, 139)# #64748B
BG_LIGHT = RGBColor(248, 250, 252)   # #F8FAFC
WHITE = RGBColor(255, 255, 255)
BORDER_GRAY = RGBColor(203, 213, 225)# #CBD5E1
LINE_GRAY = RGBColor(226, 232, 240)  # #E2E8F0

EMERALD = RGBColor(5, 150, 105)      # #059669
EMERALD_DARK = RGBColor(6, 95, 70)   # #065F46
EMERALD_BG = RGBColor(236, 253, 245) # #ECFDF5

AMBER = RGBColor(217, 119, 6)        # #D97706
AMBER_DARK = RGBColor(146, 64, 14)   # #92400E
AMBER_BG = RGBColor(255, 251, 235)   # #FFFBEB

RED = RGBColor(220, 38, 38)          # #DC2626
RED_DARK = RGBColor(153, 27, 27)     # #991B1B
RED_BG = RGBColor(254, 242, 242)     # #FEF2F2

BLUE_BG = RGBColor(239, 246, 255)    # #EFF6FF
BLUE_DARK = RGBColor(30, 58, 138)    # #1E3A8A
NAVY_SOFT = RGBColor(24, 58, 102)    # #183A66


def load_benchmark():
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows, {r["Metodo"]: r for r in rows}


def fmt(val, dec=4):
    return f"{float(val):.{dec}f}".replace(".", ",")


def build_helper_images(tmp_dir, rows):
    """Cria recortes limpos e um gráfico comparativo de alta legibilidade em pasta temporária."""
    paths = {}

    # 1. Recorte limpo da Curva ROC (painel direito de modulo1_baseline_metrics.png)
    im1 = Image.open(os.path.join(ASSETS_DIR, "modulo1_baseline_metrics.png"))
    w1, h1 = im1.size
    roc_path = os.path.join(tmp_dir, "crop_roc_baseline.png")
    im1.crop((int(w1 * 0.502), 0, w1, h1)).save(roc_path)
    paths["roc"] = roc_path

    # 2. Recorte limpo do Bar Plot SHAP (painel esquerdo de modulo2_shap_summary.png)
    im2 = Image.open(os.path.join(ASSETS_DIR, "modulo2_shap_summary.png"))
    w2, h2 = im2.size
    shap_bar_path = os.path.join(tmp_dir, "crop_shap_bar.png")
    im2.crop((0, 0, int(w2 * 0.468), h2)).save(shap_bar_path)
    paths["shap_bar"] = shap_bar_path

    # 3. Gráfico limpo de Recuperação do Ground Truth (para Slide 9)
    # Excluir Baseline (40 col) para comparar apenas os seletores K=10
    selectors = [r for r in rows if "Baseline" not in r["Metodo"]]
    label_map = {
        "XAI Proposto (shap-select OOF + Optuna)": "XAI OOF + Optuna",
        "XAI Proposto (shap-select OOF Puro)": "XAI OOF Puro",
        "Wrapper (RFE)": "Wrapper (RFE)",
        "Embedded (Lasso L1)": "Embedded (Lasso L1)",
        "Embedded (Tree MDI)": "Embedded (Tree MDI)",
        "Filter (ANOVA F)": "Filter (ANOVA F)",
        "XAI (TreeSHAP)": "XAI (TreeSHAP direto)",
        "Filter (Mutual Info)": "Filter (Mutual Info)",
    }
    labels = [label_map.get(r["Metodo"], r["Metodo"]) for r in selectors][::-1]
    inf_vals = np.array([int(r["Biomarcadores_Retidos (max 10)"]) for r in selectors][::-1])
    red_vals = np.array([int(r["Redundantes_Retidos (max 10)"]) for r in selectors][::-1])
    noi_vals = np.array([int(r["Ruidos_Selecionados (max 20)"]) for r in selectors][::-1])

    fig, ax = plt.subplots(figsize=(6.2, 4.9), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    y = np.arange(len(labels))
    h = 0.62

    ax.barh(y, inf_vals, height=h, color="#059669", label="Informativos (meta: 10)")
    ax.barh(y, red_vals, left=inf_vals, height=h, color="#F59E0B", label="Redundantes")
    ax.barh(y, noi_vals, left=inf_vals + red_vals, height=h, color="#DC2626", label="Ruído")

    for i, (iv, rv, nv) in enumerate(zip(inf_vals, red_vals, noi_vals)):
        ax.text(iv / 2, i, f"{iv}/10", va="center", ha="center", color="white", fontweight="bold", fontsize=10.5)
        if rv > 0:
            ax.text(iv + rv / 2, i, f"{rv}", va="center", ha="center", color="#1E293B", fontweight="bold", fontsize=9.5)
        if nv > 0:
            ax.text(iv + rv + nv / 2, i, f"{nv}", va="center", ha="center", color="white", fontweight="bold", fontsize=9.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, fontweight="bold", color="#1E293B")
    ax.set_xlim(0, 10.6)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Composição dos 10 Atributos Selecionados", fontsize=10.5, fontweight="bold", color="#334155")
    ax.set_title("Pureza na Recuperação do Ground Truth", fontsize=12.5, fontweight="bold", color="#0F294A", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.48, -0.14), ncol=3, frameon=False, fontsize=9.5)

    plt.tight_layout()
    bench_path = os.path.join(tmp_dir, "chart_benchmark_recuperacao.png")
    plt.savefig(bench_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["bench_recovery"] = bench_path

    return paths


def set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, x, y, w, h, fill_color=WHITE, border_color=None, border_pt=1.0, rounded=False, radius=0.04):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded and len(shp.adjustments) > 0:
        shp.adjustments[0] = radius
    if fill_color is not None:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill_color
    else:
        shp.fill.background()
    if border_color is not None:
        shp.line.color.rgb = border_color
        shp.line.width = Pt(border_pt)
    else:
        shp.line.fill.background()
    return shp


def add_accent_card(slide, x, y, w, h, fill_color=WHITE, border_color=LINE_GRAY, accent_color=BLUE, accent_side="top", accent_thick=0.06):
    card = add_rect(slide, x, y, w, h, fill_color=fill_color, border_color=border_color, border_pt=1.0, rounded=True, radius=0.025)
    if accent_color is not None:
        if accent_side == "top":
            add_rect(slide, x + 0.08, y, w - 0.16, accent_thick, fill_color=accent_color, border_color=None)
        elif accent_side == "left":
            add_rect(slide, x, y + 0.08, accent_thick, h - 0.16, fill_color=accent_color, border_color=None)
    return card


def add_textbox(slide, x, y, w, h, text="", font_size=15, bold=False, italic=False, color=SLATE_DARK, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = valign
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = FONT_NAME
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def add_rich_bullets(slide, x, y, w, h, items, font_size=14.5, space_after=10, bullet_char="▸ ", bullet_color=BLUE, text_color=SLATE_DARK, line_spacing=1.18):
    """
    items: lista de tuplas (lead_bold, rest_text) ou strings simples.
    """
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)

    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing

        if bullet_char:
            r_b = p.add_run()
            r_b.text = bullet_char
            r_b.font.name = FONT_NAME
            r_b.font.size = Pt(font_size)
            r_b.font.bold = True
            r_b.font.color.rgb = bullet_color

        if isinstance(item, tuple):
            lead, rest = item
            r1 = p.add_run()
            r1.text = lead
            r1.font.name = FONT_NAME
            r1.font.size = Pt(font_size)
            r1.font.bold = True
            r1.font.color.rgb = NAVY

            r2 = p.add_run()
            r2.text = rest
            r2.font.name = FONT_NAME
            r2.font.size = Pt(font_size)
            r2.font.color.rgb = text_color
        else:
            r1 = p.add_run()
            r1.text = item
            r1.font.name = FONT_NAME
            r1.font.size = Pt(font_size)
            r1.font.color.rgb = text_color

    return tb


def set_notes(slide, text):
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = text.strip()


def add_picture_fit(slide, img_path, box_x, box_y, box_w, box_h):
    """Insere a imagem centralizada dentro da caixa mantendo exatamente a proporção original."""
    with Image.open(img_path) as im:
        iw, ih = im.size
    aspect = iw / ih
    box_aspect = box_w / box_h
    if aspect >= box_aspect:
        w = box_w
        h = w / aspect
    else:
        h = box_h
        w = h * aspect
    x = box_x + (box_w - w) / 2.0
    y = box_y + (box_h - h) / 2.0
    slide.shapes.add_picture(img_path, Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def init_content_slide(prs, section_tag, title, slide_num, total_slides=15):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, BG_LIGHT)

    # Faixa superior sutil
    add_rect(slide, 0, 0, 13.333, 0.06, fill_color=NAVY)
    add_rect(slide, 0, 0, 3.10, 0.06, fill_color=BLUE)

    # Eyebrow da seção + Título único (sem subtítulo pesado)
    add_textbox(slide, 0.65, 0.18, 10.5, 0.25, section_tag.upper(), font_size=9.5, bold=True, color=BLUE)
    add_textbox(slide, 0.65, 0.42, 12.03, 0.46, title, font_size=22, bold=True, color=NAVY)
    add_rect(slide, 0.65, 0.94, 12.03, 0.015, fill_color=BORDER_GRAY)

    # Rodapé minimalista
    add_rect(slide, 0.65, 7.05, 12.03, 0.01, fill_color=LINE_GRAY)
    add_textbox(slide, 0.65, 7.12, 8.8, 0.25, "Eduardo Lázaro Roesler de Oliveira  •  UNESP (Mestrado)", font_size=9, color=SLATE_MUTED)
    add_textbox(slide, 10.8, 7.12, 1.88, 0.25, f"{slide_num} / {total_slides}", font_size=9, bold=True, color=NAVY, align=PP_ALIGN.RIGHT)
    return slide


def build_presentation():
    rows, by_method = load_benchmark()
    baseline = by_method["Baseline Completo (40 Atributos)"]
    prop_puro = by_method["XAI Proposto (shap-select OOF Puro)"]
    prop_opt = by_method["XAI Proposto (shap-select OOF + Optuna)"]
    rfe = by_method["Wrapper (RFE)"]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    total = 15

    with tempfile.TemporaryDirectory() as tmp_dir:
        imgs = build_helper_images(tmp_dir, rows)

        # =====================================================================
        # SLIDE 1 — CAPA
        # =====================================================================
        s1 = prs.slides.add_slide(prs.slide_layouts[6])
        set_slide_bg(s1, NAVY)
        add_rect(s1, 0, 0, 0.20, 7.5, fill_color=BLUE)

        # Pill institucional
        add_rect(s1, 0.85, 0.85, 6.2, 0.42, fill_color=NAVY_SOFT, border_color=BLUE, border_pt=1.0, rounded=True, radius=0.25)
        add_textbox(s1, 1.02, 0.91, 5.9, 0.30, "APRESENTAÇÃO DE PROJETO  •  UNESP (MESTRADO)", font_size=10.5, bold=True, color=LIGHT_BLUE)

        # Título principal e subtítulo conciso
        add_textbox(
            s1, 0.85, 1.65, 11.4, 1.55,
            "Redução de Dimensionalidade Guiada por\nExplicabilidade (XAI) em Machine Learning",
            font_size=34, bold=True, color=WHITE
        )
        add_textbox(
            s1, 0.85, 3.35, 11.2, 0.65,
            "Seleção Out-of-Fold (shap-select + VIF), Benchmark Multi-Famílias e Auditoria Local LIME",
            font_size=17, italic=True, color=LIGHT_BLUE
        )

        # Autor
        add_rect(s1, 0.85, 4.25, 2.2, 0.03, fill_color=BLUE)
        add_textbox(s1, 0.85, 4.48, 10.5, 0.38, "Eduardo Lázaro Roesler de Oliveira", font_size=16, bold=True, color=WHITE)
        add_textbox(s1, 0.85, 4.86, 10.5, 0.35, "Universidade Estadual Paulista \"Júlio de Mesquita Filho\" (UNESP)  •  eduardo.roesler@unesp.br", font_size=12, color=RGBColor(203, 213, 225))

        # 3 cards KPI de preview no rodapé da capa
        kpis_cover = [
            ("40 → 10 atributos", "Redução dimensional de 75%", BLUE),
            ("10/10 informativos", "100% de rejeição ao ruído sintético", EMERALD),
            (f"F1 = {fmt(prop_opt['F1_Score'])}", f"Supera o Baseline ({fmt(baseline['F1_Score'])}) no Holdout", LIGHT_BLUE),
        ]
        for idx, (val, desc, col) in enumerate(kpis_cover):
            kx = 0.85 + idx * 3.92
            add_rect(s1, kx, 5.75, 3.68, 1.10, fill_color=NAVY_SOFT, border_color=RGBColor(59, 130, 246), border_pt=0.8, rounded=True, radius=0.06)
            add_rect(s1, kx, 5.77, 0.08, 1.06, fill_color=col)
            add_textbox(s1, kx + 0.22, 5.88, 3.35, 0.42, val, font_size=20, bold=True, color=WHITE)
            add_textbox(s1, kx + 0.22, 6.33, 3.35, 0.42, desc, font_size=11.5, color=RGBColor(203, 213, 225))

        set_notes(s1, """[0:00 – 0:20 | 20s]
Bom dia/boa tarde a todos. Meu nome é Eduardo Oliveira, mestrando na UNESP.
Hoje apresento meu projeto de pesquisa sobre Redução de Dimensionalidade Guiada por Explicabilidade (XAI), integrando pré-filtragem VIF, seleção Out-of-Fold com shap-select, benchmark contra famílias clássicas e auditoria local com LIME.
Em resumo, conseguimos cortar 75% dos atributos (40 para 10), recuperar 100% do sinal verdadeiro sem ruído e ainda superar o modelo completo em F1-Score.""")

        # =====================================================================
        # SLIDE 2 — 1. INTRODUÇÃO: CONTEXTUALIZAÇÃO E MOTIVAÇÃO
        # =====================================================================
        s2 = init_content_slide(prs, "1. Introdução", "Alta Dimensionalidade Degrada Modelos; XAI Pode Podar Atributos", 2, total)

        # Card Esquerdo: O Desafio
        add_accent_card(s2, 0.65, 1.25, 5.82, 4.45, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=AMBER, accent_side="top")
        add_textbox(s2, 0.92, 1.48, 5.3, 0.40, "O Problema do Excesso de Dados", font_size=18, bold=True, color=AMBER_DARK)
        add_rich_bullets(s2, 0.92, 2.02, 5.3, 3.45, [
            ("Acúmulo de irrelevância: ", "Bases tabulares agregam variáveis redundantes e ruído estocástico."),
            ("Fenômeno de Hughes: ", "Com amostra fixa, dimensões extras diluem o sinal, elevam o overfitting e encarecem o treino."),
            ("Dilema em ensembles: ", "Árvores (Random Forest) toleram ruído, mas diluem importância entre colunas correlacionadas."),
        ], font_size=15.5, space_after=16, bullet_color=AMBER)

        # Card Direito: A Motivação
        add_accent_card(s2, 6.86, 1.25, 5.82, 4.45, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=BLUE, accent_side="top")
        add_textbox(s2, 7.13, 1.48, 5.3, 0.40, "XAI como Motor de Seleção", font_size=18, bold=True, color=BLUE_DARK)
        add_rich_bullets(s2, 7.13, 2.02, 5.3, 3.45, [
            ("Uso tradicional limitado: ", "SHAP e LIME costumam servir apenas como auditoria cosmética pós-treino."),
            ("Poda guiada por Shapley: ", "Usar contribuições exatas (TreeSHAP) para eliminar variáveis sem poder preditivo."),
            ("Fim do Top-K manual: ", "Substituir cortes arbitrários por um teste estatístico objetivo (p < 0,05)."),
        ], font_size=15.5, space_after=16, bullet_color=BLUE)

        # Síntese inferior
        add_accent_card(s2, 0.65, 5.90, 12.03, 0.90, fill_color=BLUE_BG, border_color=RGBColor(147, 197, 253), accent_color=BLUE, accent_side="left")
        add_rich_bullets(s2, 0.92, 6.08, 11.5, 0.56, [
            ("Motivação central: ", "comprimir bases tabulares com rigor estatístico, preservando o poder preditivo e a clareza do modelo.")
        ], font_size=14.8, space_after=0, bullet_char="", text_color=BLUE_DARK)

        set_notes(s2, """[0:20 – 1:00 | 40s]
Começando pela contextualização: em problemas tabulares, é comum acumular dezenas de variáveis redundantes e ruído puro. Pelo Fenômeno de Hughes, esse excesso dilui a densidade amostral, favorece overfitting e aumenta o custo de treino.
Ensembles como Random Forest são robustos, mas espalham a importância entre colunas colineares.
A motivação deste projeto é inverter o uso da XAI: em vez de explicar o modelo só no final, usar atribuições de Shapley (TreeSHAP) como seletor estatístico objetivo, eliminando a escolha manual do número K de atributos.""")

        # =====================================================================
        # SLIDE 3 — 1. INTRODUÇÃO: PROBLEMAS CRÍTICOS TRATADOS
        # =====================================================================
        s3 = init_content_slide(prs, "1. Introdução", "Por que a Seleção Ingênua via SHAP Falha?", 3, total)

        # Esquerda: 2 Problemas + Solução
        add_accent_card(s3, 0.65, 1.18, 5.75, 1.88, fill_color=RED_BG, border_color=RGBColor(252, 165, 165), accent_color=RED, accent_side="left")
        add_textbox(s3, 0.90, 1.28, 5.3, 0.34, "1. Colinearidade Multivariada Oculta", font_size=16, bold=True, color=RED_DARK)
        add_rich_bullets(s3, 0.90, 1.66, 5.3, 1.30, [
            ("Filtros simples falham: ", "combinações lineares têm |r| par-a-par < 0,85 e driblam correlação bivariada."),
            ("Massa SHAP dividida: ", "variáveis redundantes roubam o topo do ranking."),
        ], font_size=13.5, space_after=6, bullet_color=RED)

        add_accent_card(s3, 0.65, 3.20, 5.75, 1.68, fill_color=AMBER_BG, border_color=RGBColor(252, 211, 77), accent_color=AMBER, accent_side="left")
        add_textbox(s3, 0.90, 3.30, 5.3, 0.34, "2. Vazamento Interno (In-Sample)", font_size=16, bold=True, color=AMBER_DARK)
        add_rich_bullets(s3, 0.90, 3.68, 5.3, 1.10, [
            ("Memorização da árvore: ", "extrair SHAP nos mesmos dados de treino gera p < 0,05 falso para ruído."),
        ], font_size=13.5, space_after=4, bullet_color=AMBER)

        add_accent_card(s3, 0.65, 5.02, 5.75, 1.74, fill_color=EMERALD_BG, border_color=RGBColor(110, 231, 183), accent_color=EMERALD, accent_side="left")
        add_textbox(s3, 0.90, 5.14, 5.3, 0.34, "Solução Proposta Neste Projeto", font_size=16, bold=True, color=EMERALD_DARK)
        add_rich_bullets(s3, 0.90, 5.52, 5.3, 1.12, [
            ("Pré-filtro VIF ≤ 10: ", "remove colinearidade multivariada."),
            ("TreeSHAP 3-Fold OOF: ", "zera memorização in-sample."),
        ], font_size=13.5, space_after=8, bullet_color=EMERALD)

        # Direita: Gráfico Bar Plot SHAP mostrando a invasão das redundantes laranjas
        add_rect(s3, 6.62, 1.20, 6.06, 5.55, fill_color=WHITE, border_color=BORDER_GRAY, rounded=True, radius=0.03)
        add_picture_fit(s3, imgs["shap_bar"], 6.72, 1.30, 5.86, 4.78)
        add_textbox(
            s3, 6.78, 6.18, 5.74, 0.48,
            "Evidência no Baseline: variáveis redundantes (laranja) invadem o Top-3 SHAP e competem com os biomarcadores reais (azul).",
            font_size=11.5, italic=True, color=SLATE_MUTED, align=PP_ALIGN.CENTER
        )

        set_notes(s3, """[1:00 – 1:40 | 40s]
Por que simplesmente rodar SHAP e pegar o Top-K não funciona? Este gráfico à direita mostra o problema real no nosso Baseline:
As barras laranjas (exame_redundante_5 e 4) invadiram o segundo e o terceiro lugar no ranking TreeSHAP!
Identificamos duas armadilhas: primeiro, a colinearidade multivariada oculta — combinações lineares têm correlação par-a-par moderada, escapam de filtros simples e dividem a massa SHAP dos atributos verdadeiros. Segundo, o vazamento in-sample: calcular SHAP nos mesmos dados onde a floresta foi ajustada infla falsos positivos no teste de significância.
Nossa solução combina pré-filtro VIF <= 10 com TreeSHAP Out-of-Fold.""")

        # =====================================================================
        # SLIDE 4 — 1. INTRODUÇÃO: QUESTÃO DE PESQUISA, OBJETIVO E HIPÓTESES
        # =====================================================================
        s4 = init_content_slide(prs, "1. Introdução", "Questão de Pesquisa, Objetivo e Hipóteses", 4, total)

        # Questão de Pesquisa
        add_accent_card(s4, 0.65, 1.18, 12.03, 1.62, fill_color=BLUE_BG, border_color=RGBColor(147, 197, 253), accent_color=BLUE, accent_side="left")
        add_textbox(s4, 0.92, 1.28, 4.0, 0.30, "QUESTÃO DE PESQUISA NORTEADORA", font_size=10.5, bold=True, color=BLUE)
        add_textbox(
            s4, 0.92, 1.60, 11.55, 1.08,
            "\"Em um benchmark sintético controlado, como a seleção via VIF + TreeSHAP Out-of-Fold (shap-select) se compara a métodos filter, wrapper e embedded em recuperação do subespaço informativo, rejeição de ruído, desempenho preditivo e custo — e o que a auditoria LIME na fronteira revela sobre fidelidade local?\"",
            font_size=14.5, italic=True, color=NAVY
        )

        # Objetivo Geral
        add_accent_card(s4, 0.65, 2.98, 12.03, 1.02, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=NAVY, accent_side="left")
        add_rich_bullets(s4, 0.92, 3.16, 11.55, 0.72, [
            ("Objetivo do Trabalho: ", "propor e validar um pipeline sem vazamento (VIF + TreeSHAP OOF + shap-select p < 0,05 + Optuna), confrontando-o com 4 famílias no Holdout e 5-Fold CV, e auditando o LIME na fronteira de decisão.")
        ], font_size=14.5, space_after=0, bullet_char="")

        # 3 Hipóteses Formais
        hyps = [
            ("H1 • Desempenho Preditivo", "Cortar ≥ 50% dos atributos mantendo ou superando o F1 do Baseline (queda máxima ≤ 0,05) no Holdout e CV.", BLUE),
            ("H2 • Pureza e Custo", "Recuperar mais sinal verdadeiro com 100% de rejeição de ruído frente às 4 famílias, avaliando o tempo vs. RFE.", EMERALD),
            ("H3 • Auditoria Local LIME", "Na fronteira (P ≈ 50%, instância crítica e coorte N=20), zerar ruído nas regras locais sem degradar o R².", AMBER),
        ]
        for idx, (htitle, hbody, hcol) in enumerate(hyps):
            hx = 0.65 + idx * 4.08
            add_accent_card(s4, hx, 4.22, 3.86, 2.52, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=hcol, accent_side="top")
            add_textbox(s4, hx + 0.20, 4.45, 3.46, 0.42, htitle, font_size=15.5, bold=True, color=NAVY)
            add_textbox(s4, hx + 0.20, 4.98, 3.46, 1.55, hbody, font_size=14, color=SLATE_DARK)

        set_notes(s4, """[1:40 – 2:25 | 45s]
Nossa questão de pesquisa investiga como a combinação VIF + TreeSHAP Out-of-Fold (shap-select) se compara a métodos clássicos Filter, Wrapper e Embedded em quatro eixos: recuperação do sinal real, rejeição de ruído, desempenho preditivo e custo computacional — além da fidelidade local do LIME na fronteira de decisão.
Testamos três hipóteses quantitativas formais:
H1 prevê redução mínima de 50% mantendo ou superando o F1-Score do Baseline.
H2 prevê maior recuperação do subespaço real, 100% de rejeição ao ruído e compara o custo temporal contra o RFE.
H3 prevê que o LIME na fronteira P ≈ 50% elimina contaminação por ruído nas regras locais sem queda de R².""")

        # =====================================================================
        # SLIDE 5 — 2. TRABALHOS RELACIONADOS
        # =====================================================================
        s5 = init_content_slide(prs, "2. Trabalhos Relacionados", "Seleção Clássica vs. Explicabilidade (XAI)", 5, total)

        rel_cards = [
            ("Família Filter", "Kohavi & John (1997)", [
                ("Técnicas: ", "ANOVA F e Mutual Information."),
                ("Força: ", "execução em milissegundos."),
                ("Limite: ", "análise univariada ignora interações e retém redundantes.")
            ], SLATE_MUTED),
            ("Wrapper & Embedded", "Guyon et al. (2002)", [
                ("Técnicas: ", "RFE, Lasso L1 e Tree MDI."),
                ("Força: ", "consideram o preditor."),
                ("Limite: ", "RFE é custoso; Lasso e MDI oscilam sob colinearidade.")
            ], AMBER),
            ("Explicabilidade Formal", "Lundberg & Lee (2017); Ribeiro et al. (2016)", [
                ("SHAP: ", "atribuições axiomáticas exatas (TreeSHAP)."),
                ("LIME: ", "aproximação linear local; sofre em alta dimensão (Alvarez-Melis & Jaakkola, 2018).")
            ], BLUE),
            ("Seleção via SHAP", "Marcílio & Eler (2020)", [
                ("Avanço: ", "usa ranking |SHAP| para podar variáveis."),
                ("Limite: ", "exige corte Top-K manual e divide peso entre colineares.")
            ], EMERALD),
        ]
        for idx, (ctitle, cite, bullets, col) in enumerate(rel_cards):
            cx = 0.65 + idx * 3.07
            add_accent_card(s5, cx, 1.22, 2.84, 4.10, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=col, accent_side="top")
            add_textbox(s5, cx + 0.16, 1.40, 2.52, 0.36, ctitle, font_size=15.5, bold=True, color=NAVY)
            add_textbox(s5, cx + 0.16, 1.78, 2.52, 0.44, cite, font_size=11, italic=True, color=col)
            add_rect(s5, cx + 0.16, 2.28, 2.52, 0.015, fill_color=LINE_GRAY)
            add_rich_bullets(s5, cx + 0.14, 2.42, 2.56, 2.75, bullets, font_size=13, space_after=10, bullet_color=col)

        # Lacuna preenchida
        add_accent_card(s5, 0.65, 5.54, 12.03, 1.22, fill_color=EMERALD_BG, border_color=RGBColor(110, 231, 183), accent_color=EMERALD, accent_side="left")
        add_textbox(s5, 0.92, 5.66, 4.5, 0.30, "CONTRIBUIÇÃO DESTE TRABALHO", font_size=10.5, bold=True, color=EMERALD_DARK)
        add_rich_bullets(s5, 0.92, 5.98, 11.55, 0.65, [
            ("Preenchimento da lacuna: ", "acoplar pré-filtro VIF ≤ 10 + TreeSHAP 3-Fold OOF + regressão Logit (p < 0,05) elimina o corte Top-K manual, neutraliza a redundância e evita vazamento in-sample.")
        ], font_size=14.2, space_after=0, bullet_char="", text_color=SLATE_DARK)

        set_notes(s5, """[2:25 – 3:15 | 50s]
Na literatura, Kohavi & John organizam a seleção clássica em três famílias: filtros como ANOVA e Mutual Info são muito rápidos, mas univariados; wrappers como o RFE de Guyon et al. capturam interações via retreinamento iterativo, porém com alto custo; e métodos embutidos como Lasso e MDI sofrem instabilidade sob colinearidade.
Na área de XAI, Lundberg & Lee formalizaram o SHAP via Teoria dos Jogos e Ribeiro et al. propuseram o LIME, cuja fragilidade sob perturbações em alta dimensão foi demonstrada por Alvarez-Melis & Jaakkola.
Marcílio & Eler (2020) mostraram que o ranking SHAP melhora a seleção, mas ainda depende de um corte Top-K arbitrário. Nossa contribuição avança ao combinar VIF <= 10 com TreeSHAP Out-of-Fold e teste p < 0,05.""")

        # =====================================================================
        # SLIDE 6 — 3. METODOLOGIA: DADOS E BASELINE
        # =====================================================================
        s6 = init_content_slide(prs, "3. Metodologia", "Base Sintética Controlada e Modelo Baseline", 6, total)

        # Esquerda: Dados + Barra de Composição + Protocolo
        add_accent_card(s6, 0.65, 1.18, 5.85, 2.48, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=BLUE, accent_side="top")
        add_textbox(s6, 0.90, 1.34, 5.35, 0.34, "Benchmark Sintético (make_classification)", font_size=16, bold=True, color=NAVY)
        add_rich_bullets(s6, 0.90, 1.74, 5.35, 1.80, [
            ("Dimensão: ", "N = 2.000 observações e D = 40 atributos contínuos (seed 42)."),
            ("Ruído de fronteira: ", "classes 60% / 40% com 3% de inversão de rótulo (flip_y = 0,03)."),
            ("Justificativa: ", "permite auditar o ground truth exato (sem uso clínico real)."),
        ], font_size=13.2, space_after=6, bullet_color=BLUE)

        # Barra visual proporcional dos 40 atributos (10 + 10 + 20)
        add_textbox(s6, 0.65, 3.80, 5.85, 0.28, "Composição exata dos 40 atributos (Ground Truth):", font_size=12, bold=True, color=SLATE_MUTED)
        # 5.85 in total -> 10/40 = 1.4625 in, 10/40 = 1.4625 in, 20/40 = 2.925 in
        add_rect(s6, 0.65, 4.10, 1.46, 0.66, fill_color=BLUE)
        add_textbox(s6, 0.65, 4.16, 1.46, 0.54, "10 Inform.\n(25%)", font_size=11.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        add_rect(s6, 2.11, 4.10, 1.46, 0.66, fill_color=AMBER)
        add_textbox(s6, 2.11, 4.16, 1.46, 0.54, "10 Redund.\n(25%)", font_size=11.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
        add_rect(s6, 3.57, 4.10, 2.93, 0.66, fill_color=RED)
        add_textbox(s6, 3.57, 4.16, 2.93, 0.54, "20 Ruídos Gaussianos (50%)", font_size=11.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)

        # Protocolo e Baseline
        add_accent_card(s6, 0.65, 4.92, 5.85, 1.86, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=NAVY, accent_side="left")
        add_textbox(s6, 0.90, 5.05, 5.4, 0.34, "Protocolo Duplo & Baseline (RF 100 árvores)", font_size=15.5, bold=True, color=NAVY)
        add_rich_bullets(s6, 0.90, 5.45, 5.4, 1.25, [
            ("Partição: ", "75% treino (1.500) / 25% holdout (500) + Validação Cruzada 5-Fold."),
            ("Baseline (40 col.): ", f"F1 Holdout = {fmt(baseline['F1_Score'])} | F1 CV = {fmt(baseline['F1_CV_Media'])} | Treino = {fmt(baseline['Tempo_Treino_ms'], 1)} ms."),
        ], font_size=13.2, space_after=6, bullet_color=NAVY)

        # Direita: Curva ROC do Baseline
        add_rect(s6, 6.72, 1.20, 5.96, 5.58, fill_color=WHITE, border_color=BORDER_GRAY, rounded=True, radius=0.03)
        add_picture_fit(s6, imgs["roc"], 6.84, 1.32, 5.72, 4.72)
        add_textbox(
            s6, 6.84, 6.18, 5.72, 0.45,
            f"Desempenho Baseline (40 atributos): ROC-AUC = {fmt(baseline['ROC_AUC'])} e Acurácia = {fmt(baseline['Acuracia'])} no Holdout.",
            font_size=11.5, italic=True, color=SLATE_MUTED, align=PP_ALIGN.CENTER
        )

        set_notes(s6, """[3:15 – 3:55 | 40s]
Em bases reais, raramente sabemos com certeza quais colunas são ruído puro. Por isso, construímos um benchmark sintético controlado via make_classification (semente 42) com 2.000 amostras e exatamente 40 variáveis: 10 informativas, 10 redundantes por combinação linear e 20 ruídos gaussianos, com 3% de flip_y para simular incerteza na fronteira.
Os dados foram divididos de forma estratificada em 75% treino e 25% holdout, além de validação cruzada de 5 folds.
O modelo Baseline (Random Forest com 100 árvores usando todas as 40 colunas) atingiu F1 de 0,8856 no holdout, 0,8649 na CV e AUC de 0,9655, com treino em 878,8 ms.""")

        # =====================================================================
        # SLIDE 7 — 3. METODOLOGIA: PIPELINE PROPOSTO
        # =====================================================================
        s7 = init_content_slide(prs, "3. Metodologia", "Pipeline Proposto em 4 Estágios (Sem Vazamento em X_train)", 7, total)

        stages = [
            ("1. Pré-Filtro Ortogonal", "Variance (< 0,01), Pearson (|r| > 0,90) e VIF ≤ 10 iterativo.", "40 → 30 colunas", "Elimina 100% das redundantes", AMBER),
            ("2. TreeSHAP 3-Fold OOF", "Calcula valores Shapley fora da amostra sobre X_train_pre.", "Matriz Φ_OOF (30 col.)", "Zero memorização in-sample", BLUE),
            ("3. shap-select (Logit)", "Regressão y_train ~ Φ_OOF; aprova apenas β > 0 e p < 0,05.", "30 → 10 informativos", "Automático, sem Top-K", EMERALD),
            ("4. Avaliação & LIME", "Teste ceteris paribus, Optuna TPE (20 trials) e auditoria local.", "Holdout + 5-Fold CV", "Fronteira P ≈ 50%", NAVY),
        ]
        for idx, (st_title, st_desc, st_badge, st_note, col) in enumerate(stages):
            sx = 0.65 + idx * 3.08
            add_accent_card(s7, sx, 1.35, 2.78, 3.35, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=col, accent_side="top")
            add_textbox(s7, sx + 0.16, 1.56, 2.46, 0.42, st_title, font_size=15, bold=True, color=NAVY)
            add_textbox(s7, sx + 0.16, 2.06, 2.46, 1.15, st_desc, font_size=13.5, color=SLATE_DARK)
            add_rect(s7, sx + 0.16, 3.38, 2.46, 0.48, fill_color=BG_LIGHT, border_color=col, border_pt=0.9, rounded=True, radius=0.15)
            add_textbox(s7, sx + 0.20, 3.45, 2.38, 0.34, st_badge, font_size=12.5, bold=True, color=col, align=PP_ALIGN.CENTER)
            add_textbox(s7, sx + 0.16, 4.02, 2.46, 0.48, st_note, font_size=11.5, italic=True, color=SLATE_MUTED, align=PP_ALIGN.CENTER)

            if idx < 3:
                add_textbox(s7, sx + 2.78, 2.75, 0.30, 0.45, "▶", font_size=16, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

        # Dois cards embaixo: Ceteris Paribus vs Optuna
        add_accent_card(s7, 0.65, 5.02, 5.85, 1.70, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=EMERALD, accent_side="left")
        add_textbox(s7, 0.90, 5.16, 5.4, 0.34, "Avaliação Justa (Ceteris Paribus)", font_size=15.5, bold=True, color=EMERALD_DARK)
        add_textbox(s7, 0.90, 5.56, 5.4, 1.00, "O subconjunto XAI Puro e os 6 seletores clássicos (ANOVA, MI, RFE, Lasso, MDI, TreeSHAP) usam exatamente o mesmo Random Forest padrão (100 árvores).", font_size=13.5, color=SLATE_DARK)

        add_accent_card(s7, 6.83, 5.02, 5.85, 1.70, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=BLUE, accent_side="left")
        add_textbox(s7, 7.08, 5.16, 5.4, 0.34, "Refinamento Bayesiano (+ Optuna TPE)", font_size=15.5, bold=True, color=BLUE_DARK)
        add_textbox(s7, 7.08, 5.56, 5.4, 1.00, "Após isolar os 10 atributos limpos, o Optuna executa 20 trials TPE (5-Fold CV em X_train) para calibrar hiperparâmetros no subespaço compacto.", font_size=13.5, color=SLATE_DARK)

        set_notes(s7, """[3:55 – 4:40 | 45s]
Todo o pipeline roda estritamente sobre X_train para evitar vazamento de dados:
No estágio 1, VarianceThreshold, Pearson e o Fator de Inflação da Variância (VIF <= 10) reduzem de 40 para 30 colunas, removendo exatamente as 10 combinações lineares redundantes.
No estágio 2, calculamos TreeSHAP em 3-Fold Out-of-Fold: cada amostra recebe atribuições de uma floresta que não a viu no treino.
No estágio 3, ajustamos uma regressão logística (shap-select) de y sobre a matriz SHAP OOF, aprovando apenas variáveis com beta positivo e p < 0,05. Isso retém os 10 informativos sem fixar K manualmente.
Por fim, comparamos a proposta de forma justa (mesmo RF padrão de 100 árvores) e também após otimização Optuna, além da auditoria LIME.""")

        # =====================================================================
        # SLIDE 8 — 3. METODOLOGIA: DIAGNÓSTICO SHAP-SELECT OOF
        # =====================================================================
        s8 = init_content_slide(prs, "3. Metodologia", "Diagnóstico shap-select OOF: Separação Automática de Sinal e Ruído", 8, total)

        add_rect(s8, 0.65, 1.22, 8.20, 5.50, fill_color=WHITE, border_color=BORDER_GRAY, rounded=True, radius=0.03)
        add_picture_fit(s8, os.path.join(ASSETS_DIR, "modulo5_shap_select_analysis.png"), 0.78, 1.36, 7.94, 4.90)
        add_textbox(s8, 0.80, 6.32, 7.90, 0.32, "Regressão Logit sobre a matriz TreeSHAP Out-of-Fold (3-Fold Stratified).", font_size=11, italic=True, color=SLATE_MUTED, align=PP_ALIGN.CENTER)

        insights_s8 = [
            ("Critério Duplo", "Aprova apenas coeficiente β > 0 e p-valor < 0,05. Dispensa escolher K na mão.", BLUE),
            ("10/10 Informativos", "Todos os 10 atributos reais foram aprovados com alta significância (p ≤ 0,035).", EMERALD),
            ("0/20 Ruídos", "Nenhum ruído estocástico passou no teste inferencial (todos com p ≥ 0,116 ou β ≤ 0).", RED),
        ]
        for idx, (ititle, ibody, icol) in enumerate(insights_s8):
            iy = 1.22 + idx * 1.90
            add_accent_card(s8, 9.08, iy, 3.60, 1.70, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=icol, accent_side="left")
            add_textbox(s8, 9.30, iy + 0.18, 3.22, 0.36, ititle, font_size=16.5, bold=True, color=icol)
            add_textbox(s8, 9.30, iy + 0.60, 3.22, 0.95, ibody, font_size=13.8, color=SLATE_DARK)

        set_notes(s8, """[4:40 – 5:15 | 35s]
Este diagnóstico visual mostra exatamente como o shap-select Out-of-Fold decide quais variáveis ficam.
Em verde, os 10 atributos informativos verdadeiros obtiveram coeficiente beta positivo e p-valor abaixo de 0,05 (chegando a 10^-28).
Em vermelho, todo o ruído estocástico foi rejeitado por não atingir significância estatística ou ter direção negativa. Ou seja, 100% do sinal recuperado automaticamente.""")

        # =====================================================================
        # SLIDE 9 — 4. EXPERIMENTOS, RESULTADOS E DISCUSSÃO: BENCHMARK MULTI-FAMÍLIAS
        # =====================================================================
        s9 = init_content_slide(prs, "4. Experimentos, Resultados e Discussão", "Benchmark Multi-Famílias: Pureza Exata e Ganho Preditivo", 9, total)

        # 3 KPI Pills no topo
        top_kpis = [
            ("10 / 10 Informativos", "0 redundantes  •  0 ruídos", EMERALD_BG, EMERALD_DARK, EMERALD),
            (f"F1 Holdout: {fmt(prop_opt['F1_Score'])}", f"vs. {fmt(baseline['F1_Score'])} Baseline  ({fmt(prop_puro['F1_Score'])} Puro)", BLUE_BG, BLUE_DARK, BLUE),
            (f"F1 CV (5f): {fmt(prop_puro['F1_CV_Media'])}", f"vs. {fmt(baseline['F1_CV_Media'])} Baseline  (AUC {fmt(prop_opt['ROC_AUC'])})", BLUE_BG, BLUE_DARK, BLUE),
        ]
        for idx, (kt, ks, kbg, kfg, kbd) in enumerate(top_kpis):
            kx = 0.65 + idx * 4.08
            add_rect(s9, kx, 1.12, 3.87, 0.78, fill_color=kbg, border_color=kbd, border_pt=1.0, rounded=True, radius=0.10)
            add_textbox(s9, kx + 0.14, 1.17, 3.60, 0.36, kt, font_size=15.5, bold=True, color=kfg, align=PP_ALIGN.CENTER)
            add_textbox(s9, kx + 0.14, 1.52, 3.60, 0.30, ks, font_size=11, color=SLATE_DARK, align=PP_ALIGN.CENTER)

        # Tabela enxuta e legível à esquerda (6 colunas)
        headers = ["Método", "Inf.", "Red+Ruído", "F1 Hold.", "F1 CV (5f)", "Sel. (s)"]
        col_widths = [2.46, 0.64, 1.12, 0.88, 0.98, 0.77]
        table_shape = s9.shapes.add_table(len(rows) + 1, len(headers), Inches(0.65), Inches(2.10), Inches(sum(col_widths)), Inches(4.70))
        tbl = table_shape.table
        for ci, cw in enumerate(col_widths):
            tbl.columns[ci].width = Inches(cw)

        short_names = {
            "XAI Proposto (shap-select OOF + Optuna)": "XAI OOF + Optuna",
            "XAI Proposto (shap-select OOF Puro)": "XAI OOF Puro",
            "Wrapper (RFE)": "Wrapper (RFE)",
            "Embedded (Lasso L1)": "Embedded (Lasso L1)",
            "Embedded (Tree MDI)": "Embedded (Tree MDI)",
            "Filter (ANOVA F)": "Filter (ANOVA F)",
            "XAI (TreeSHAP)": "XAI (TreeSHAP direto)",
            "Filter (Mutual Info)": "Filter (Mutual Info)",
            "Baseline Completo (40 Atributos)": "Baseline (40 atrib.)",
        }

        for ci, htext in enumerate(headers):
            cell = tbl.cell(0, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = htext
            r.font.name = FONT_NAME
            r.font.size = Pt(11)
            r.font.bold = True
            r.font.color.rgb = WHITE

        for ri, rdata in enumerate(rows, start=1):
            m = rdata["Metodo"]
            is_prop = "XAI Proposto" in m
            is_base = "Baseline" in m
            bg_c = EMERALD_BG if is_prop else (AMBER_BG if is_base else (WHITE if ri % 2 == 1 else BG_LIGHT))
            fg_c = EMERALD_DARK if is_prop else (AMBER_DARK if is_base else SLATE_DARK)

            inf_n = int(rdata["Biomarcadores_Retidos (max 10)"])
            red_n = int(rdata["Redundantes_Retidos (max 10)"])
            noi_n = int(rdata["Ruidos_Selecionados (max 20)"])
            vals = [
                short_names.get(m, m),
                f"{inf_n}/10",
                f"{red_n} + {noi_n}",
                fmt(rdata["F1_Score"], 4),
                fmt(rdata["F1_CV_Media"], 3),
                f"{fmt(rdata['Tempo_Selecao_s'], 2)}",
            ]
            for ci, val in enumerate(vals):
                cell = tbl.cell(ri, ci)
                cell.fill.solid()
                cell.fill.fore_color.rgb = bg_c
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                p = cell.text_frame.paragraphs[0]
                p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
                r = p.add_run()
                r.text = val
                r.font.name = FONT_NAME
                r.font.size = Pt(10.8)
                r.font.bold = is_prop or is_base
                r.font.color.rgb = fg_c

        # Direita: Gráfico de recuperação do ground truth
        add_rect(s9, 7.68, 2.10, 5.00, 4.70, fill_color=WHITE, border_color=BORDER_GRAY, rounded=True, radius=0.03)
        add_picture_fit(s9, imgs["bench_recovery"], 7.76, 2.18, 4.84, 4.54)

        set_notes(s9, """[5:15 – 6:05 | 50s]
Comparando nossa proposta contra seis seletores clássicos fixados em K=10 e contra o Baseline de 40 variáveis:
Pelo gráfico à direita, a proposta (OOF + VIF) foi a única a recuperar 10/10 atributos informativos com zero redundantes e zero ruídos. RFE, ANOVA, MDI e até o TreeSHAP direto estacionaram em 7/10 porque capturaram 3 variáveis redundantes; Lasso e Mutual Info deixaram passar ruído puro.
Na tabela à esquerda, mesmo SEM Optuna (ceteris paribus), o XAI OOF Puro atingiu F1 = 0,8861 no holdout e 0,8873 na validação cruzada — já superando o Baseline (0,8856 e 0,8649). Com Optuna, o holdout sobe para 0,8872 e AUC de 0,9734.
O ponto de atenção está na última coluna: o tempo de seleção offline (11,04 s puro; 79,30 s com Optuna) é maior que os 4,55 s do RFE.""")

        # =====================================================================
        # SLIDE 10 — 4. EXPERIMENTOS, RESULTADOS E DISCUSSÃO: ABLAÇÃO PROGRESSIVA
        # =====================================================================
        s10 = init_content_slide(prs, "4. Experimentos, Resultados e Discussão", "Ablação Progressiva (40 → 2): Pico em K = 10 e Custo", 10, total)

        add_rect(s10, 0.65, 1.15, 12.03, 3.88, fill_color=WHITE, border_color=BORDER_GRAY, rounded=True, radius=0.03)
        add_picture_fit(s10, os.path.join(ASSETS_DIR, "modulo4_ablation_curves.png"), 0.78, 1.24, 11.77, 3.70)

        ab_cards = [
            ("1. Pico Exato em K = 10", "A trajetória OOF + VIF (verde escuro) atinge seu máximo em 10 atributos, superando o modelo de 40 variáveis (Hughes mitigado).", EMERALD),
            ("2. Colapso Abaixo de K < 8", "Quando a poda remove variáveis informativas verdadeiras, todas as famílias sofrem queda acentuada no F1-Score.", RED),
            ("3. Custo Offline vs. Online", f"O treino final cai 40,2% ({fmt(baseline['Tempo_Treino_ms'], 1)} → {fmt(prop_puro['Tempo_Treino_ms'], 1)} ms), compensando a seleção offline única ({fmt(prop_puro['Tempo_Selecao_s'], 2)} s puro).", BLUE),
        ]
        for idx, (atitle, abody, acol) in enumerate(ab_cards):
            ax = 0.65 + idx * 4.08
            add_accent_card(s10, ax, 5.20, 3.86, 1.62, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=acol, accent_side="top")
            add_textbox(s10, ax + 0.18, 5.36, 3.50, 0.34, atitle, font_size=14.5, bold=True, color=NAVY)
            add_textbox(s10, ax + 0.18, 5.74, 3.50, 1.00, abody, font_size=12.8, color=SLATE_DARK)

        set_notes(s10, """[6:05 – 6:50 | 45s]
Na ablação progressiva podando de 40 até 2 atributos:
À esquerda, vemos que a curva OOF+VIF atinge o pico exatamente em K=10 (a linha pontilhada vermelha do subespaço real). Abaixo de 8 atributos, todas as famílias colapsam porque começam a perder sinal verdadeiro.
À direita, o tempo de treino do Random Forest cai 40,2% — de 878,8 ms para 525,7 ms no modelo enxuto puro. Embora a seleção offline leve mais tempo que o RFE, ela ocorre uma única vez na engenharia, enquanto o modelo produtivo opera continuamente com 75% menos variáveis.""")

        # =====================================================================
        # SLIDE 11 — 4. EXPERIMENTOS, RESULTADOS E DISCUSSÃO: AUDITORIA LOCAL LIME
        # =====================================================================
        s11 = init_content_slide(prs, "4. Experimentos, Resultados e Discussão", "Auditoria Local LIME na Fronteira (P ≈ 50%): Zero Ruído", 11, total)

        add_rect(s11, 0.65, 1.15, 12.03, 3.74, fill_color=WHITE, border_color=BORDER_GRAY, rounded=True, radius=0.03)
        add_picture_fit(s11, os.path.join(ASSETS_DIR, "modulo3_lime_local.png"), 0.78, 1.22, 11.77, 3.60)

        obs_id = int(float(prop_opt["Paciente_Limiar"]))
        add_accent_card(s11, 0.65, 5.05, 5.88, 1.76, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=BLUE, accent_side="left")
        add_textbox(s11, 0.90, 5.17, 5.45, 0.34, f"Instância Mais Incerta: Observação #{obs_id} (P = 50,0%)", font_size=15, bold=True, color=BLUE_DARK)
        add_rich_bullets(s11, 0.90, 5.55, 5.45, 1.15, [
            ("Fidelidade local R²: ", f"subiu de {fmt(prop_opt['R2_LIME_Baseline'])} → {fmt(prop_opt['R2_LIME_Reduzido'])}."),
            ("Regras contaminadas no Top-10: ", "caíram de 3/10 (redundantes) para 0/10."),
        ], font_size=13.2, space_after=6, bullet_color=BLUE)

        add_accent_card(s11, 6.80, 5.05, 5.88, 1.76, fill_color=EMERALD_BG, border_color=RGBColor(110, 231, 183), accent_color=EMERALD, accent_side="left")
        add_textbox(s11, 7.05, 5.17, 5.45, 0.34, "Coorte de Fronteira (N = 20 mais incertas)", font_size=15, bold=True, color=EMERALD_DARK)
        add_rich_bullets(s11, 7.05, 5.55, 5.45, 1.15, [
            ("Pureza local absoluta: ", f"regras não-informativas caíram de {fmt(prop_opt['Ruidos_LIME_Coorte_Base'], 2)} → {fmt(prop_opt['Ruidos_LIME_Coorte_Red'], 2)} / 10."),
            ("Ajuste R² médio: ", f"mantido ({fmt(prop_opt['R2_LIME_Coorte_Base_Media'], 3)} → {fmt(prop_opt['R2_LIME_Coorte_Red_Media'], 3)}), limitado pela não-linearidade do RF."),
        ], font_size=13.2, space_after=6, bullet_color=EMERALD)

        set_notes(s11, f"""[6:50 – 7:35 | 45s]
Para auditar a explicabilidade local (H3), aplicamos o LIME na fronteira de máxima incerteza (P ≈ 50%).
Na observação crítica #{obs_id}, o modelo Baseline de 40 variáveis incluía 3 atributos redundantes entre as 10 principais regras locais (barras à esquerda). No modelo reduzido (à direita), as 10 regras são exclusivamente biomarcadores reais, e o R² local subiu de 0,3066 para 0,3851.
Expandindo para uma coorte das 20 instâncias mais incertas do holdout, o R² médio manteve-se moderado (0,248 para 0,255) pela própria não-linearidade do Random Forest na zona de transição, mas a contaminação por variáveis não-informativas caiu de 3,80 para 0,00 por instância.""")

        # =====================================================================
        # SLIDE 12 — 4. EXPERIMENTOS, RESULTADOS E DISCUSSÃO: SÍNTESE QUANTITATIVA
        # =====================================================================
        s12 = init_content_slide(prs, "4. Experimentos, Resultados e Discussão", "Síntese Comparativa: Baseline (40) vs. Proposta XAI (10)", 12, total)

        kpi_grid = [
            (
                "Compressão & Pureza", "40 → 10 atributos (-75%)",
                [
                    ("Ground truth: ", "10/10 informativos e 0% ruído."),
                    ("Massa SHAP: ", f"{fmt(float(prop_opt['Massa_SHAP_Retida_pct'])*100, 1)}% retida nos informativos (sem diluição)."),
                ],
                EMERALD, EMERALD_DARK
            ),
            (
                "Qualidade Preditiva", f"F1 Holdout: {fmt(prop_opt['F1_Score'])}",
                [
                    ("Sem Optuna (puro): ", f"F1 = {fmt(prop_puro['F1_Score'])} | CV = {fmt(prop_puro['F1_CV_Media'])}."),
                    ("Com Optuna: ", f"Acurácia = {fmt(prop_opt['Acuracia'])} | ROC-AUC = {fmt(prop_opt['ROC_AUC'])}."),
                ],
                BLUE, BLUE_DARK
            ),
            (
                "Eficiência Computacional", "Treino -40,2% (RF Puro)",
                [
                    ("Uso contínuo (puro): ", f"treino {fmt(baseline['Tempo_Treino_ms'], 1)} → {fmt(prop_puro['Tempo_Treino_ms'], 1)} ms; inf. {fmt(prop_puro['Latencia_Inf_ms'], 1)} ms."),
                    ("Custo offline único: ", f"{fmt(prop_puro['Tempo_Selecao_s'], 2)} s (puro) / {fmt(prop_opt['Tempo_Selecao_s'], 2)} s (+Optuna)."),
                ],
                NAVY, NAVY
            ),
            (
                "Auditoria Local LIME", "3,80 → 0,00 regras ruidosas",
                [
                    (f"Obs. #{obs_id} (P = 50%): ", f"R² local {fmt(prop_opt['R2_LIME_Baseline'])} → {fmt(prop_opt['R2_LIME_Reduzido'])}."),
                    ("Coorte fronteira (N=20): ", "100% das regras locais baseadas em sinal real."),
                ],
                AMBER, AMBER_DARK
            ),
        ]

        for idx, (kcat, kbig, kbullets, kcol, ktcol) in enumerate(kpi_grid):
            col_i = idx % 2
            row_i = idx // 2
            gx = 0.65 + col_i * 6.18
            gy = 1.25 + row_i * 2.82
            add_accent_card(s12, gx, gy, 5.85, 2.58, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=kcol, accent_side="top")
            add_textbox(s12, gx + 0.24, gy + 0.18, 5.35, 0.30, kcat.upper(), font_size=10.5, bold=True, color=kcol)
            add_textbox(s12, gx + 0.24, gy + 0.50, 5.35, 0.55, kbig, font_size=24, bold=True, color=ktcol)
            add_rect(s12, gx + 0.24, gy + 1.15, 5.35, 0.015, fill_color=LINE_GRAY)
            add_rich_bullets(s12, gx + 0.22, gy + 1.28, 5.40, 1.18, kbullets, font_size=13.5, space_after=8, bullet_color=kcol)

        set_notes(s12, """[7:35 – 8:10 | 35s]
Este painel consolida os quatro pilares do experimento comparando o Baseline de 40 atributos com o modelo reduzido a 10 variáveis:
Primeiro, compressão de 75% com 100% de pureza.
Segundo, superação preditiva tanto na comparação pura quanto após Optuna (AUC de 0,9734).
Terceiro, redução de 40,2% no treino do classificador final no modelo puro, mantendo o custo maior restrito à seleção offline.
Quarto, na auditoria LIME, eliminação total de regras espúrias na coorte de fronteira (de 3,80 para 0,00).""")

        # =====================================================================
        # SLIDE 13 — 5. CONCLUSÃO: PRINCIPAIS RESULTADOS E RESPOSTA À QUESTÃO
        # =====================================================================
        s13 = init_content_slide(prs, "5. Conclusão", "Principais Resultados e Resposta à Questão de Pesquisa", 13, total)

        verdicts = [
            (
                "H1 • Desempenho", "CONFIRMADA", EMERALD_BG, EMERALD, EMERALD_DARK,
                f"Corte de 75% elevou o F1 no Holdout ({fmt(baseline['F1_Score'])} → {fmt(prop_opt['F1_Score'])}) e na CV 5-Fold ({fmt(baseline['F1_CV_Media'])} → {fmt(prop_puro['F1_CV_Media'])}), mitigando o Fenômeno de Hughes."
            ),
            (
                "H2 • Pureza & Custo", "PUREZA SIM | CUSTO NÃO", AMBER_BG, AMBER, AMBER_DARK,
                f"Único método a atingir 10/10 informativos e 0% ruído sem corte Top-K. Porém, a seleção offline ({fmt(prop_puro['Tempo_Selecao_s'], 2)} s puro; {fmt(prop_opt['Tempo_Selecao_s'], 2)} s +Optuna) foi mais lenta que RFE ({fmt(rfe['Tempo_Selecao_s'], 2)} s)."
            ),
            (
                "H3 • Auditoria LIME", "CONFIRMADA", EMERALD_BG, EMERALD, EMERALD_DARK,
                f"Não houve queda de R² ({fmt(prop_opt['R2_LIME_Baseline'])} → {fmt(prop_opt['R2_LIME_Reduzido'])} na Obs. #{obs_id}; {fmt(prop_opt['R2_LIME_Coorte_Base_Media'], 3)} → {fmt(prop_opt['R2_LIME_Coorte_Red_Media'], 3)} na coorte), zerando o ruído local (3,80 → 0,00)."
            ),
        ]
        for idx, (vh, vbadge, vbg, vcol, vtcol, vtext) in enumerate(verdicts):
            vy = 1.20 + idx * 1.44
            add_accent_card(s13, 0.65, vy, 12.03, 1.26, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=vcol, accent_side="left")
            add_textbox(s13, 0.90, vy + 0.16, 2.55, 0.34, vh, font_size=15, bold=True, color=NAVY)
            add_rect(s13, 0.90, vy + 0.56, 2.40, 0.44, fill_color=vbg, border_color=vcol, border_pt=1.0, rounded=True, radius=0.18)
            add_textbox(s13, 0.94, vy + 0.62, 2.32, 0.32, vbadge, font_size=11, bold=True, color=vtcol, align=PP_ALIGN.CENTER)
            add_textbox(s13, 3.55, vy + 0.22, 8.90, 0.88, vtext, font_size=14, color=SLATE_DARK, valign=MSO_ANCHOR.MIDDLE)

        # Resposta direta à Questão de Pesquisa
        add_rect(s13, 0.65, 5.56, 12.03, 1.26, fill_color=NAVY, rounded=True, radius=0.05)
        add_textbox(s13, 0.92, 5.66, 5.5, 0.28, "RESPOSTA À QUESTÃO DE PESQUISA", font_size=10, bold=True, color=LIGHT_BLUE)
        add_textbox(
            s13, 0.92, 5.96, 11.50, 0.78,
            "VIF ≤ 10 + TreeSHAP Out-of-Fold (shap-select) supera filter, wrapper e embedded em recuperação exata do sinal (10/10), F1-Score e pureza das explicações locais LIME, ao custo de maior tempo apenas na etapa única de seleção offline.",
            font_size=14, bold=True, color=WHITE
        )

        set_notes(s13, """[8:10 – 8:55 | 45s]
Respondendo diretamente à questão de pesquisa e julgando as hipóteses:
H1 foi confirmada com superação: cortar 75% das variáveis elevou o F1 tanto no holdout quanto na validação cruzada.
H2 teve resultado misto transparente: foi confirmada em pureza (único com 10/10 informativos e 0% ruído sem Top-K manual), mas rejeitada em velocidade de seleção offline frente ao RFE (11,04 s vs. 4,55 s).
H3 foi confirmada: o R² local do LIME não caiu e a contaminação por ruído nas regras locais caiu a zero.
Portanto, o método proposto supera as quatro famílias em qualidade e pureza, sendo ideal quando o custo de seleção offline é aceitável.""")

        # =====================================================================
        # SLIDE 14 — 5. CONCLUSÃO: LIMITAÇÕES E TRABALHOS FUTUROS
        # =====================================================================
        s14 = init_content_slide(prs, "5. Conclusão", "Limitações do Trabalho e Próximos Passos", 14, total)

        add_accent_card(s14, 0.65, 1.25, 5.82, 5.45, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=AMBER, accent_side="top")
        add_textbox(s14, 0.92, 1.48, 5.3, 0.40, "Limitações Metodológicas", font_size=18, bold=True, color=AMBER_DARK)
        add_rich_bullets(s14, 0.92, 2.04, 5.3, 4.45, [
            ("Benchmark sintético: ", "comprova eficácia matemática sob ground truth conhecido, mas não valida uso clínico nem dados reais."),
            ("Associação ≠ causalidade: ", "SHAP e LIME descrevem o comportamento do modelo preditivo, não relações causais físicas."),
            ("Custo computacional O(K · TLD²): ", "a extração OOF em 3 folds é mais lenta que filtros univariados e RFE (indicada para seleção offline)."),
            ("Ausência de estudo humano: ", "a carga cognitiva de Miller (7 ± 2) é referência teórica; não houve testes com usuários."),
        ], font_size=14.5, space_after=16, bullet_color=AMBER)

        add_accent_card(s14, 6.86, 1.25, 5.82, 5.45, fill_color=WHITE, border_color=BORDER_GRAY, accent_color=BLUE, accent_side="top")
        add_textbox(s14, 7.13, 1.48, 5.3, 0.40, "Trabalhos Futuros", font_size=18, bold=True, color=BLUE_DARK)
        add_rich_bullets(s14, 7.13, 2.04, 5.3, 4.45, [
            ("Bases reais e múltiplas sementes: ", "estender o protocolo OOF + VIF a datasets públicos de alta dimensão (ex.: genômica) com intervalos de incerteza."),
            ("Explicadores locais não-lineares: ", "comparar LIME com métodos de âncoras (Anchors) e contrafactuais na fronteira P ≈ 50%."),
            ("Avaliação centrada no usuário: ", "mensurar empiricamente o ganho de interpretabilidade e velocidade de decisão com especialistas."),
        ], font_size=14.5, space_after=18, bullet_color=BLUE)

        set_notes(s14, """[8:55 – 9:25 | 30s]
Por transparência científica, destaco quatro limitações: os dados são sintéticos e em partição única holdout/5-Fold CV, não validando aplicação clínica; valores SHAP e LIME expressam associação estatística no estimador e não causalidade; o custo computacional OOF restringe a seleção ao ambiente offline; e a referência cognitiva de Miller (7 ± 2) não foi testada com humanos.
Como trabalhos futuros, propomos testar múltiplas sementes em bases reais de alta dimensionalidade, comparar LIME com Anchors e contrafactuais, e conduzir estudos empíricos com usuários.""")

        # =====================================================================
        # SLIDE 15 — 6. REFERÊNCIAS BIBLIOGRÁFICAS E ENCERRAMENTO
        # =====================================================================
        s15 = init_content_slide(prs, "6. Referências Bibliográficas", "Referências Bibliográficas", 15, total)

        refs_left = [
            "[1] C. Molnar, Interpretable Machine Learning: A Guide for Making Black Box Models Explainable, 2nd ed. Leanpub, 2022.",
            "[2] R. Kohavi and G. H. John, \"Wrappers for feature subset selection,\" Artificial Intelligence, vol. 97, no. 1–2, pp. 273–324, 1997.",
            "[3] I. Guyon, J. Weston, S. Barnhill, and V. Vapnik, \"Gene selection for cancer classification using support vector machines,\" Machine Learning, vol. 46, no. 1–3, pp. 389–422, 2002.",
            "[4] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017, pp. 4765–4774.",
        ]
        refs_right = [
            "[5] M. T. Ribeiro, S. Singh, and C. Guestrin, \"'Why should I trust you?': Explaining the predictions of any classifier,\" in Proc. ACM SIGKDD, 2016, pp. 1135–1144.",
            "[6] W. E. Marcílio and D. M. Eler, \"From explanations to feature selection: assessing SHAP values as feature selection mechanism,\" in 33rd SIBGRAPI, IEEE, 2020, pp. 340–347.",
            "[7] D. Alvarez-Melis and T. S. Jaakkola, \"On the robustness of interpretability methods,\" in ICML Workshop on Human Interpretability in ML (WHI), 2018.",
            "[8] G. A. Miller, \"The magical number seven, plus or minus two: Some limits on our capacity for processing information,\" Psychological Review, vol. 63, no. 2, pp. 81–97, 1956.",
        ]

        add_rect(s15, 0.65, 1.20, 5.85, 4.25, fill_color=WHITE, border_color=BORDER_GRAY, rounded=True, radius=0.03)
        add_rich_bullets(s15, 0.85, 1.38, 5.45, 3.95, refs_left, font_size=11.5, space_after=12, bullet_char="", line_spacing=1.18)

        add_rect(s15, 6.83, 1.20, 5.85, 4.25, fill_color=WHITE, border_color=BORDER_GRAY, rounded=True, radius=0.03)
        add_rich_bullets(s15, 7.03, 1.38, 5.45, 3.95, refs_right, font_size=11.5, space_after=12, bullet_char="", line_spacing=1.18)

        # Banner de encerramento
        add_rect(s15, 0.65, 5.68, 12.03, 1.15, fill_color=NAVY, border_color=BLUE, border_pt=1.0, rounded=True, radius=0.06)
        add_textbox(s15, 0.90, 5.82, 11.53, 0.42, "Obrigado pela atenção!  •  Espaço aberto para perguntas", font_size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_textbox(s15, 0.90, 6.28, 11.53, 0.35, "Eduardo Lázaro Roesler de Oliveira  —  eduardo.roesler@unesp.br  —  UNESP (Mestrado)", font_size=12, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)

        set_notes(s15, """[9:25 – 9:35 | 10s]
Aqui estão as principais referências bibliográficas que fundamentam o trabalho.
Muito obrigado pela atenção e fico à disposição para perguntas.""")

    prs.save(PPTX_OUT)
    print(f"[+] Apresentação PowerPoint gerada com sucesso em: {PPTX_OUT}")


if __name__ == "__main__":
    build_presentation()
