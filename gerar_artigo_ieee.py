"""
===================================================================================
GERADOR DO ARTIGO CIENTÍFICO NO PADRÃO IEEE (EXATAMENTE 3 PÁGINAS)
===================================================================================
Projeto: Redução de Dimensionalidade Guiada por Explicabilidade (XAI)
Instituição: Centro Universitário Eurípides de Marília (UNIVEM)
Autor: Eduardo Lázaro Roesler de Oliveira

Formatação:
- Padrão IEEE (Two-Column Conference Format, A4)
- Margens: Sup/Inf 0.70 pol (1.8 cm), Esq/Dir 0.65 pol (1.65 cm)
- Tipografia: Times New Roman 9.5pt corpo, entrelinhas 1.05, títulos em caixa alta/itálico
- Seção 1 (Full-width 1 coluna): Título, Autores, Resumo e Palavras-chave
- Seção 2 (2 colunas, espaçamento 0.24 pol): Corpo do artigo, tabelas, figuras e referências
- Limite estrito: Exatamente até 3 páginas (validado via pypdf)
===================================================================================
"""

import os
import shutil
import csv
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
import docx2pdf
import pypdf

BASE_DIR = r"c:\Users\eduar\projetos\xai_data_reduction"
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DOCX_OUTPUT = os.path.join(BASE_DIR, "artigo_cientifico_ieee.docx")
PDF_OUTPUT = os.path.join(BASE_DIR, "artigo_cientifico_ieee.pdf")
DOCS_PDF_OUTPUT = os.path.join(BASE_DIR, "docs", "artigo_cientifico_ieee.pdf")


def load_benchmark_rows():
    path = os.path.join(ASSETS_DIR, "tabela_comparativa_completa.csv")
    with open(path, newline="", encoding="utf-8-sig") as file:
        return {row["Metodo"]: row for row in csv.DictReader(file)}


def format_decimal(value, places=4):
    return f"{float(value):.{places}f}".replace(".", ",")


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=35, bottom=35, left=45, right=45):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_ieee_heading_1(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(9.8)
    run.font.color.rgb = RGBColor(16, 44, 87)
    return p


def add_ieee_heading_2(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.italic = True
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(9.2)
    run.font.color.rgb = RGBColor(40, 40, 40)
    return p


def add_ieee_paragraph(doc, text, bold_prefix="", italic_prefix="", indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.line_spacing = 1.05
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.18)

    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.bold = True
        r_b.font.name = 'Times New Roman'
        r_b.font.size = Pt(9.3)

    if italic_prefix:
        r_i = p.add_run(italic_prefix)
        r_i.italic = True
        r_i.font.name = 'Times New Roman'
        r_i.font.size = Pt(9.3)

    r_text = p.add_run(text)
    r_text.font.name = 'Times New Roman'
    r_text.font.size = Pt(9.3)
    r_text.font.color.rgb = RGBColor(20, 20, 20)
    return p


def add_ieee_equation(doc, eq_text, eq_num):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r_eq = p.add_run(f"{eq_text}    ({eq_num})")
    r_eq.font.name = 'Times New Roman'
    r_eq.font.size = Pt(8.8)
    r_eq.italic = True
    return p


def add_ieee_image(doc, filename, caption, width_inches=3.35):
    img_path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(img_path):
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_img.paragraph_format.space_before = Pt(4)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(img_path, width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_cap.paragraph_format.space_before = Pt(1)
    p_cap.paragraph_format.space_after = Pt(4)
    p_cap.paragraph_format.keep_with_next = False
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(7.8)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(60, 60, 60)


def add_ieee_table_1(doc, benchmark):
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_cap.paragraph_format.space_before = Pt(4)
    p_cap.paragraph_format.space_after = Pt(2)
    p_cap.paragraph_format.keep_with_next = True
    r_cap = p_cap.add_run("TABELA I: BENCHMARK EXPERIMENTAL MULTI-FAMÍLIAS (K = 10 ATRIBUTOS)")
    r_cap.bold = True
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(8.0)
    r_cap.font.color.rgb = RGBColor(16, 44, 87)

    method_order = [
        "XAI Proposto (shap-select + Optuna)",
        "Embedded (Lasso L1)",
        "Wrapper (RFE)",
        "XAI (TreeSHAP)",
        "Filter (ANOVA F)",
        "Embedded (Tree MDI)",
        "Filter (Mutual Info)",
        "Baseline Completo (40 Atributos)"
    ]
    table_data = [[
        "Método / Abordagem", "Família", "Atrib.", "Ruído", "Rej. Ruído",
        "F1-Score", "Acurácia", "ROC-AUC", "Treino"
    ]]
    for method in method_order:
        row = benchmark[method]
        noise_total = 20
        table_data.append([
            method,
            row["Familia"],
            f'{int(float(row["Biomarcadores_Retidos (max 10)"]))}/10',
            f'{int(float(row["Ruidos_Selecionados (max 20)"]))}/{noise_total}',
            row["Taxa_Rejeicao_Ruido"],
            format_decimal(row["F1_Score"]),
            format_decimal(row["Acuracia"]),
            format_decimal(row["ROC_AUC"]),
            f'{format_decimal(row["Tempo_Treino_ms"], 1)} ms'
        ])

    col_widths = [1.14, 0.46, 0.24, 0.26, 0.31, 0.31, 0.31, 0.31, 0.35]
    table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for row_idx, row in enumerate(table_data):
        for col_idx, text in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.width = Inches(col_widths[col_idx])
            set_cell_margins(cell, top=22, bottom=22, left=26, right=26)
            p = cell.paragraphs[0]
            p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT if col_idx == 0 else WD_PARAGRAPH_ALIGNMENT.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(6.6)

            if row_idx == 0:
                set_cell_background(cell, "EBF2FA")
                r.bold = True
                r.font.color.rgb = RGBColor(16, 44, 87)
            elif row_idx == 1:
                set_cell_background(cell, "F0FFF4")
                r.bold = True
                r.font.color.rgb = RGBColor(20, 80, 20)
            elif row_idx % 2 == 0:
                set_cell_background(cell, "F9FAFB")


def add_ieee_table_2(doc, benchmark):
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_cap.paragraph_format.space_before = Pt(4)
    p_cap.paragraph_format.space_after = Pt(2)
    p_cap.paragraph_format.keep_with_next = True
    r_cap = p_cap.add_run("TABELA II: AUDITORIA FORENSE E INTERPRETABILIDADE MULTIDIMENSIONAL")
    r_cap.bold = True
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(8.0)
    r_cap.font.color.rgb = RGBColor(16, 44, 87)

    proposed = benchmark["XAI Proposto (shap-select + Optuna)"]
    baseline = benchmark["Baseline Completo (40 Atributos)"]
    n_reduced = int(float(proposed["N_Atributos"]))
    reduction = 100 * (1 - n_reduced / int(float(baseline["N_Atributos"])))
    noise_rejected = proposed["Taxa_Rejeicao_Ruido"]
    table_data = [
        ["Dimensão Avaliada", "Baseline (40)", "XAI Reduzido", "Leitura / Limitação"],
        [
            "Número de atributos",
            f'{int(float(baseline["N_Atributos"]))} (100%)',
            f'{n_reduced} ({100 - reduction:.1f}%)',
            f"Redução de {reduction:.1f}% no conjunto sintético"
        ],
        [
            "Rejeição de ruído",
            baseline["Taxa_Rejeicao_Ruido"],
            noise_rejected,
            "Contagem conhecida apenas por construção sintética"
        ],
        [
            "Fidelidade LIME (R²)",
            format_decimal(proposed["R2_LIME_Baseline"]),
            format_decimal(proposed["R2_LIME_Reduzido"]),
            "Uma instância; valores baixos indicam ajuste local limitado"
        ],
        [
            "Massa SHAP retida",
            "100% (referência)",
            f'{float(proposed["Massa_SHAP_Retida_pct"]):.1%}',
            "Atribuição do baseline; não é medida de causalidade"
        ],
        [
            "Heurística de Miller",
            "40 atributos",
            f"{n_reduced} atributos",
            "Referência teórica; não foi realizado teste cognitivo"
        ]
    ]

    col_widths = [1.05, 0.90, 1.05, 0.70]
    table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for row_idx, row in enumerate(table_data):
        for col_idx, text in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.width = Inches(col_widths[col_idx])
            set_cell_margins(cell, top=22, bottom=22, left=24, right=24)
            p = cell.paragraphs[0]
            p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT if col_idx in [0, 3] else WD_PARAGRAPH_ALIGNMENT.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(6.5)

            if row_idx == 0:
                set_cell_background(cell, "EBF2FA")
                r.bold = True
                r.font.color.rgb = RGBColor(16, 44, 87)
            elif row_idx % 2 == 1:
                set_cell_background(cell, "F9FAFB")


def build_ieee_article():
    benchmark = load_benchmark_rows()
    proposed = benchmark["XAI Proposto (shap-select + Optuna)"]
    baseline = benchmark["Baseline Completo (40 Atributos)"]
    n_reduced = int(float(proposed["N_Atributos"]))
    n_original = int(float(baseline["N_Atributos"]))
    reduction = 100 * (1 - n_reduced / n_original)
    f1_delta = float(proposed["F1_Score"]) - float(baseline["F1_Score"])
    train_saving = 100 * (
        1 - float(proposed["Tempo_Treino_ms"]) / float(baseline["Tempo_Treino_ms"])
    )
    doc = Document()

    # Configuração de Margens Estilo IEEE Conference A4
    sec1 = doc.sections[0]
    sec1.top_margin = Inches(0.70)
    sec1.bottom_margin = Inches(0.70)
    sec1.left_margin = Inches(0.65)
    sec1.right_margin = Inches(0.65)
    sec1.page_width = Inches(8.27)  # A4
    sec1.page_height = Inches(11.69)

    # =========================================================================
    # SEÇÃO 1: CABEÇALHO IEEE (LARGURA TOTAL / 1 COLUNA)
    # =========================================================================
    p_title = doc.add_paragraph()
    p_title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    r_title = p_title.add_run("Redução de Dimensionalidade Guiada por XAI:\nBenchmark Sintético e Auditoria Local")
    r_title.bold = True
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(16.5)
    r_title.font.color.rgb = RGBColor(16, 44, 87)

    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_auth.paragraph_format.space_before = Pt(0)
    p_auth.paragraph_format.space_after = Pt(5)
    r_name = p_auth.add_run("Eduardo Lázaro Roesler de Oliveira\n")
    r_name.bold = True
    r_name.font.name = 'Times New Roman'
    r_name.font.size = Pt(10.0)
    r_affil = p_auth.add_run("Centro Universitário Eurípides de Marília (UNIVEM) — Marília, SP, Brasil\n")
    r_affil.font.name = 'Times New Roman'
    r_affil.font.size = Pt(8.8)
    r_mail = p_auth.add_run("eduardo.roesler@univem.edu.br")
    r_mail.font.name = 'Times New Roman'
    r_mail.font.size = Pt(8.5)
    r_mail.font.italic = True
    r_mail.font.color.rgb = RGBColor(80, 80, 80)

    # Resumo
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    p_abs.paragraph_format.left_indent = Inches(0.20)
    p_abs.paragraph_format.right_indent = Inches(0.20)
    p_abs.paragraph_format.space_before = Pt(0)
    p_abs.paragraph_format.space_after = Pt(2.5)
    p_abs.paragraph_format.line_spacing = 1.03
    r_abs_lbl = p_abs.add_run("Resumo— ")
    r_abs_lbl.bold = True
    r_abs_lbl.font.name = 'Times New Roman'
    r_abs_lbl.font.size = Pt(8.5)
    r_abs_txt = p_abs.add_run(
        f"Este estudo avalia a seleção de atributos guiada por explicabilidade em dados sintéticos, sem pretensão de validação clínica. "
        f"Foram geradas 2.000 observações e 40 colunas (10 informativas, 10 redundantes e 20 de ruído), com divisão estratificada treino/teste de 75/25. "
        f"Comparam-se métodos filter, wrapper, embedded e uma proposta que combina TreeSHAP, pré-filtro, regressão sobre contribuições SHAP (shap-select) e ajuste Optuna. "
        f"No holdout desta execução, a proposta reteve {n_reduced} atributos ({reduction:.1f}% de redução), obteve F1 {format_decimal(proposed['F1_Score'])} "
        f"frente a {format_decimal(baseline['F1_Score'])} no baseline e rejeitou {proposed['Taxa_Rejeicao_Ruido']} do ruído sintético. "
        f"O LIME foi avaliado em uma única observação; seus valores de R² são reportados como diagnóstico local, não como evidência de validade clínica ou benefício cognitivo."
    )
    r_abs_txt.font.name = 'Times New Roman'
    r_abs_txt.font.size = Pt(8.5)

    # Palavras-chave
    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    p_kw.paragraph_format.left_indent = Inches(0.20)
    p_kw.paragraph_format.right_indent = Inches(0.20)
    p_kw.paragraph_format.space_before = Pt(0)
    p_kw.paragraph_format.space_after = Pt(7)
    r_kw_lbl = p_kw.add_run("Termos para Indexação— ")
    r_kw_lbl.bold = True
    r_kw_lbl.font.name = 'Times New Roman'
    r_kw_lbl.font.size = Pt(8.5)
    r_kw_txt = p_kw.add_run("Inteligência Artificial Explicável (XAI), Redução de Dimensionalidade, Seleção de Características, TreeSHAP, LIME, Dados Sintéticos, Benchmark.")
    r_kw_txt.font.name = 'Times New Roman'
    r_kw_txt.font.size = Pt(8.5)

    # =========================================================================
    # SEÇÃO 2: CORPO EM DUAS COLUNAS (FORMATO IEEE)
    # =========================================================================
    sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
    sectPr2 = sec2._sectPr
    cols = parse_xml(r'<w:cols xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:num="2" w:space="360"/>')
    sectPr2.append(cols)

    # I. INTRODUÇÃO E QUESTÃO DE PESQUISA
    add_ieee_heading_1(doc, "I. INTRODUÇÃO E QUESTÃO DE PESQUISA")
    add_ieee_paragraph(
        doc,
        "A seleção de atributos busca reduzir a dimensionalidade sem comprometer o desempenho preditivo. "
        "A interpretação de modelos pode orientar esse processo, mas atribuições explicativas descrevem o comportamento do estimador e não identificam relações causais. "
        "Este trabalho avalia uma estratégia que usa TreeSHAP e shap-select em um benchmark sintético controlado."
    )
    add_ieee_paragraph(
        doc,
        "A proposta combina importância global TreeSHAP, pré-filtro de variância e colinearidade, regressão sobre contribuições SHAP (shap-select) e otimização Optuna. "
        "O LIME é usado apenas para examinar localmente uma observação do holdout."
    )
    add_ieee_paragraph(
        doc,
        "\"Em um benchmark sintético controlado, como uma seleção de atributos informada por TreeSHAP e shap-select se compara a métodos filter, wrapper e embedded em redução de ruído, desempenho preditivo e custo de seleção, e o que uma auditoria LIME de uma observação permite concluir sobre fidelidade local?\"",
        bold_prefix="Questão de Pesquisa Norteadora: "
    )
    add_ieee_paragraph(
        doc,
        "As hipóteses são avaliadas apenas nesta execução e partição: H1 prevê redução de ao menos 50% com queda de F1 não superior a 0,05; "
        "H2 prevê maior rejeição do ruído sintético que os métodos comparados e menor tempo de seleção que RFE; "
        "H3 prevê que o R² do LIME não diminua na observação selecionada. Uma única partição e uma explicação local não permitem inferências clínicas ou cognitivas."
    )

    # II. TRABALHOS RELACIONADOS
    add_ieee_heading_1(doc, "II. TRABALHOS RELACIONADOS")
    add_ieee_paragraph(
        doc,
        "A taxonomia clássica de seleção de atributos proposta por Kohavi e John [2] classifica as estratégias em três vertentes canônicas: "
        "filtros (avaliadores univariados independentes do modelo, como ANOVA e Mutual Information), wrappers (que usam o estimador como função objetivo, como o Recursive Feature Elimination - RFE de Guyon et al. [3]) "
        "e embedded (seleção intrínseca durante o treino, como penalização Lasso L1 e importância por impureza MDI/Gini). "
        "Filtros univariados podem deixar de capturar interações; wrappers podem ter custo elevado conforme o estimador e o espaço de busca; e métodos embedded podem ser instáveis diante de atributos correlacionados."
    )
    add_ieee_paragraph(
        doc,
        "No campo de XAI, Lundberg e Lee [4] formalizaram o SHAP com garantias axiomáticas da Teoria dos Jogos Cooperativos (eficiência, simetria, dummy e aditividade). "
        "Ribeiro et al. [5] introduziram o LIME para aproximações locais lineares interpretáveis. "
        "Marcílio e Eler [6] identificaram que atribuições explicáveis possuem potencial inexplorado para simplificação estrutural de modelos, lacuna que este trabalho aborda formalmente."
    )

    # III. METODOLOGIA EXPERIMENTAL
    add_ieee_heading_1(doc, "III. METODOLOGIA EXPERIMENTAL")
    add_ieee_heading_2(doc, "A. Síntese do Conjunto de Dados e Baseline")
    add_ieee_paragraph(
        doc,
        "Gerou-se uma base sintética com 2.000 observações e 40 atributos por make_classification: 10 informativos, 10 redundantes e 20 de ruído, com 3% de ruído nos rótulos. "
        "Com shuffle=False, os nomes das colunas correspondem à ordem dos grupos gerados; os nomes não representam biomarcadores reais nem causalidade. "
        "Usou-se uma divisão estratificada de 75% treino e 25% holdout. O baseline foi Random Forest com 100 árvores."
    )

    add_ieee_heading_2(doc, "B. Explicabilidade Global e Demarcação Teórica")
    add_ieee_paragraph(
        doc,
        "O algoritmo TreeSHAP calcula os valores de Shapley exatos φ_i para cada atributo i através da soma ponderada sobre todas as coalizões de atributos S:"
    )
    add_ieee_equation(doc, "φ_i = ∑_{S ⊆ F \\ {i}} (|S|!(|F|-|S|-1)! / |F|!) [f(S ∪ {i}) - f(S)]", "1")
    add_ieee_paragraph(
        doc,
        "O TreeSHAP resume atribuições do modelo para orientar um ranking global; isso não implica causalidade. "
        "O LIME aproxima localmente as previsões do modelo e foi aplicado a uma observação do holdout cuja probabilidade do baseline está mais próxima de 0,5. O R² mede o ajuste local do substituto, não a correção clínica:"
    )
    add_ieee_equation(doc, "ξ(x) = argmin_{g ∈ G} L(f, g, π_x) + Ω(g)", "2")

    add_ieee_heading_2(doc, "C. Pipeline Proposto: Pré-Filtro, shap-select e Optuna")
    add_ieee_paragraph(
        doc,
        "O pipeline proposto opera em três etapas sequenciais integradas: "
        "(1) O pré-filtro remove variância abaixo de 0,01 e colinearidade com correlação absoluta de Pearson acima de 0,90. "
        "(2) O shap-select ajusta Logit aos valores SHAP e seleciona até 10 atributos com coeficiente positivo e p-valor < 0,05; se o Logit falha nas exceções previstas, usa OLS como fallback explícito. "
        "(3) O Optuna executa 15 tentativas TPE com validação cruzada de 3 folds no conjunto de treino. "
        "As métricas finais vêm de um único holdout e não incluem validação externa."
    )

    add_ieee_heading_2(doc, "D. Métricas de Avaliação Multidimensional")
    add_ieee_paragraph(
        doc,
        "Além das métricas preditivas (F1-Score, Acurácia, ROC-AUC) e computacionais (tempo de treino e latência de inferência), a interpretabilidade foi formalizada em quatro eixos auditáveis: "
        "(1) Fração da massa SHAP do baseline retida pelos atributos selecionados; "
        "(2) Fidelidade Local do Modelo Substituto LIME (R²); "
        "(3) Taxa de Rejeição de Ruído Puro (blindagem contra ruído); e "
        "(4) Número de atributos, tratado apenas como proxy de complexidade: não houve avaliação com usuários. A regra de Miller [8] é uma heurística e não comprova capacidade cognitiva neste projeto."
    )

    # IV. EXPERIMENTOS, RESULTADOS E DISCUSSÃO
    add_ieee_heading_1(doc, "IV. EXPERIMENTOS, RESULTADOS E DISCUSSÃO")
    add_ieee_heading_2(doc, "A. Benchmark Comparativo Multi-Famílias")
    add_ieee_paragraph(
        doc,
        "Para comparar as famílias, fixou-se o mesmo teto de K = 10 atributos nas seleções por ranking; a proposta estatística pode reter menos, até esse limite. O baseline usa os 40 atributos. A Tabela I sintetiza o benchmark deste holdout."
    )

    # Tabela I
    add_ieee_table_1(doc, benchmark)

    add_ieee_paragraph(
        doc,
        f"Na Tabela I, a proposta reduziu {reduction:.1f}% dos atributos; o F1 passou de "
        f"{format_decimal(baseline['F1_Score'])} para {format_decimal(proposed['F1_Score'])} "
        f"(delta {format_decimal(f1_delta, 4)}). O criterio numerico de H1 foi atendido nesta particao "
        f"({n_reduced} atributos e queda de F1 nao superior a 0,05), sem demonstrar generalizacao alem do holdout."
    )

    # Figura 1: Ablação
    add_ieee_image(
        doc,
        "modulo4_ablation_curves.png",
        "Fig. 1. Estudo de ablação progressiva multivariada (40 a 2 atributos) comparando a acurácia e F1-score entre as quatro famílias."
    )

    add_ieee_heading_2(doc, "B. Pureza de Seleção e Rejeição de Ruído Estocástico")
    add_ieee_paragraph(
        doc,
        f"A proposta reteve {int(float(proposed['Ruidos_Selecionados (max 20)']))} atributos rotulados como ruido "
        f"({proposed['Taxa_Rejeicao_Ruido']} de rejeicao por construcao do dataset). Esse resultado empata com "
        "alguns metodos do benchmark; nao demonstra superioridade universal. Os nomes informativos, redundantes "
        "e de ruido sao definidos por construcao sintetica e nao correspondem a causas clinicas."
    )

    add_ieee_heading_2(doc, "C. Custo Computacional e Estudo de Ablação")
    add_ieee_paragraph(
        doc,
        f"Em uma execucao nesta maquina, o treino do modelo reduzido levou "
        f"{format_decimal(proposed['Tempo_Treino_ms'], 1)} ms frente a "
        f"{format_decimal(baseline['Tempo_Treino_ms'], 1)} ms no baseline "
        f"({train_saving:.1f}% menos); a latencia foi {format_decimal(proposed['Latencia_Inf_ms'], 1)} ms. "
        f"A selecao proposta levou {format_decimal(proposed['Tempo_Selecao_s'], 2)} s, contra "
        f"{format_decimal(benchmark['Wrapper (RFE)']['Tempo_Selecao_s'], 2)} s para RFE. "
        "A proposta nao foi mais rapida nesta medicao; tempos de uma execucao dependem do hardware."
    )

    # Quebra de página controlada para equilibrar perfeitamente as páginas 2 e 3
    doc.add_page_break()

    # Figura 2: LIME Local (Topo da Página 3)
    add_ieee_image(
        doc,
        "modulo3_lime_local.png",
        "Fig. 2. Auditoria forense local via LIME no paciente crítico limítrofe (P ≈ 50%): comparação de pesos e fidelidade R² entre Baseline e Modelo Reduzido."
    )

    add_ieee_heading_2(doc, "D. Auditoria LIME Local e Limites de Interpretacao")
    add_ieee_paragraph(
        doc,
        f"Para a observacao {int(float(proposed['Paciente_Limiar']))}, selecionada por ter a probabilidade do baseline mais proxima de 0,5 "
        f"({float(proposed['Probabilidade_Baseline_Limiar']):.1%}), o R² do LIME foi "
        f"{format_decimal(proposed['R2_LIME_Baseline'])} no baseline e "
        f"{format_decimal(proposed['R2_LIME_Reduzido'])} no modelo reduzido. O ajuste local permanece limitado "
        f"e a diferenca descreve apenas esta observacao. Os atributos selecionados retiveram "
        f"{float(proposed['Massa_SHAP_Retida_pct']):.1%} da massa SHAP do baseline. "
        f"O conjunto tem {n_reduced} atributos, acima do limite superior 9 da heuristica 7 ± 2; "
        "nao houve teste de compreensao com pessoas, portanto nao se conclui reducao de carga cognitiva."
    )

    # Tabela II
    add_ieee_table_2(doc, benchmark)

    # V. CONCLUSÕES E LIMITAÇÕES
    add_ieee_heading_1(doc, "V. CONCLUSÕES E LIMITAÇÕES")
    add_ieee_paragraph(
        doc,
        f"Neste benchmark sintetico, a proposta reteve {n_reduced} de {n_original} atributos e atendeu ao limite de queda de F1 definido em H1 nesta particao. "
        "Entretanto, nao superou todos os metodos em desempenho e sua selecao foi mais lenta que RFE nesta medicao. "
        "A rejeicao de ruido e a fidelidade LIME referem-se a dados sinteticos e a uma unica particao/observacao; nao constituem validacao clinica nem evidencia de causalidade."
    )
    add_ieee_paragraph(
        doc,
        "Como limitações epistemológicas, destaca-se que as atribuições Shapley refletem importância preditiva sob o modelo e não inferência causal biológica pura. "
        "Ademais, o cálculo do TreeSHAP impõe custo inicial O(TLD²), justificando o pré-filtro híbrido adotado. "
        "Trabalhos futuros contemplarão a extensão para dados clínicos multimodais e verificação em bases hospitalares multicêntricas reais."
    )

    # REFERÊNCIAS BIBLIOGRÁFICAS
    add_ieee_heading_1(doc, "REFERÊNCIAS")
    refs = [
        "[1] C. Molnar, Interpretable Machine Learning: A Guide for Making Black Box Models Explainable, 2nd ed. Leanpub, 2022.",
        "[2] R. Kohavi and G. H. John, \"Wrappers for feature subset selection,\" Artificial Intelligence, vol. 97, no. 1-2, pp. 273-324, 1997.",
        "[3] I. Guyon, J. Weston, S. Barnhill, and V. Vapnik, \"Gene selection for cancer classification using support vector machines,\" Machine Learning, vol. 46, no. 1-3, pp. 389-422, 2002.",
        "[4] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 4765-4774.",
        "[5] M. T. Ribeiro, S. Singh, and C. Guestrin, \"'Why should I trust you?': Explaining the predictions of any classifier,\" in Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD), 2016, pp. 1135-1144.",
        "[6] W. E. Marcílio and D. M. Eler, \"From explainability to model improvement: A survey on XAI-based feature selection,\" IEEE Trans. Knowl. Data Eng., vol. 34, no. 11, pp. 5500-5515, 2020.",
        "[7] D. Alvarez-Melis and T. S. Jaakkola, \"Towards robust interpretability with self-explaining neural networks,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 31, 2018, pp. 7775-7784.",
        "[8] G. A. Miller, \"The magical number seven, plus or minus two: Some limits on our capacity for processing information,\" Psychological Review, vol. 63, no. 2, pp. 81-97, 1956."
    ]

    for ref in refs:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
        p_ref.paragraph_format.left_indent = Inches(0.18)
        p_ref.paragraph_format.first_line_indent = Inches(-0.18)
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(1.5)
        p_ref.paragraph_format.line_spacing = 1.02
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = 'Times New Roman'
        r_ref.font.size = Pt(7.4)
        r_ref.font.color.rgb = RGBColor(40, 40, 40)

    doc.save(DOCX_OUTPUT)
    print(f"[+] Artigo IEEE em Word salvo em: {DOCX_OUTPUT}")

    # Conversão para PDF
    print("[*] Convertendo Artigo IEEE para PDF...")
    docx2pdf.convert(DOCX_OUTPUT, PDF_OUTPUT)
    print(f"[+] Artigo IEEE em PDF gerado em: {PDF_OUTPUT}")

    # Cópia para a pasta docs/
    os.makedirs(os.path.dirname(DOCS_PDF_OUTPUT), exist_ok=True)
    shutil.copyfile(PDF_OUTPUT, DOCS_PDF_OUTPUT)
    print(f"[+] Cópia sincronizada em: {DOCS_PDF_OUTPUT}")

    # Verificação estrita de páginas via pypdf
    reader = pypdf.PdfReader(PDF_OUTPUT)
    total_pages = len(reader.pages)
    print(f"\n=======================================================")
    print(f"VERIFICAÇÃO DE PÁGINAS DO ARTIGO IEEE:")
    print(f"Total de páginas geradas: {total_pages} páginas")
    print(f"Critério exigido: ATÉ 3 PÁGINAS")
    if total_pages <= 3:
        print(f"[SUCESSO TOTAL] O artigo atende rigorosamente ao limite de até 3 páginas ({total_pages}/3)!")
    else:
        print(f"[AVISO] O documento excedeu o limite ({total_pages} páginas). Ajuste necessário.")
    print(f"=======================================================\n")
    return total_pages


if __name__ == "__main__":
    build_ieee_article()
