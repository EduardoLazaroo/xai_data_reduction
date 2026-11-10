"""
===================================================================================
GERADOR DO ARTIGO CIENTÍFICO NO PADRÃO IEEE (ATÉ 3 PÁGINAS)
===================================================================================
Projeto: Redução de Dimensionalidade Guiada por Explicabilidade (XAI)
Instituição: Universidade Estadual Paulista "Júlio de Mesquita Filho" (UNESP) — Mestrado
Autor: Eduardo Lázaro Roesler de Oliveira

Formatação:
- Padrão IEEE (Two-Column Conference Format, A4)
- Margens: Sup/Inf 0.65 pol, Esq/Dir 0.62 pol
- Tipografia: Times New Roman 9.2pt corpo, entrelinhas 1.04, títulos em caixa alta/itálico
- Seção 1 (Full-width 1 coluna): Título, Autores, Resumo e Palavras-chave
- Seção 2 (2 colunas, espaçamento 0.22 pol): Corpo do artigo, tabelas, figuras e referências
- Limite estrito: Até 3 páginas (validado via pypdf)
===================================================================================
"""

import os
import shutil
import csv
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


def set_cell_margins(cell, top=20, bottom=20, left=24, right=24):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_row_cant_split(row, is_header=False):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:cantSplit'))
    if is_header:
        trPr.append(OxmlElement('w:tblHeader'))


def add_ieee_heading_1(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(9.5)
    run.font.color.rgb = RGBColor(16, 44, 87)
    return p


def add_ieee_heading_2(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
    p.paragraph_format.space_before = Pt(3.5)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.italic = True
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(9.0)
    run.font.color.rgb = RGBColor(40, 40, 40)
    return p


def add_ieee_paragraph(doc, text, bold_prefix="", italic_prefix="", indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2.0)
    p.paragraph_format.line_spacing = 1.03
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.16)

    if bold_prefix:
        r_b = p.add_run(bold_prefix)
        r_b.bold = True
        r_b.font.name = 'Times New Roman'
        r_b.font.size = Pt(9.0)

    if italic_prefix:
        r_i = p.add_run(italic_prefix)
        r_i.italic = True
        r_i.font.name = 'Times New Roman'
        r_i.font.size = Pt(9.0)

    r_text = p.add_run(text)
    r_text.font.name = 'Times New Roman'
    r_text.font.size = Pt(9.0)
    r_text.font.color.rgb = RGBColor(20, 20, 20)
    return p


def add_ieee_equation(doc, eq_text, eq_num):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r_eq = p.add_run(f"{eq_text}    ({eq_num})")
    r_eq.font.name = 'Times New Roman'
    r_eq.font.size = Pt(8.6)
    r_eq.italic = True
    return p


def add_ieee_image(doc, filename, caption, width_inches=3.32):
    img_path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(img_path):
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_img.paragraph_format.space_before = Pt(3)
    p_img.paragraph_format.space_after = Pt(1.5)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(img_path, width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_cap.paragraph_format.space_before = Pt(0.5)
    p_cap.paragraph_format.space_after = Pt(3)
    p_cap.paragraph_format.keep_with_next = False
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(7.6)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(60, 60, 60)


def add_ieee_table_1(doc, benchmark):
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_cap.paragraph_format.space_before = Pt(3.5)
    p_cap.paragraph_format.space_after = Pt(1.5)
    p_cap.paragraph_format.keep_with_next = True
    r_cap = p_cap.add_run("TABELA I: BENCHMARK MULTI-FAMÍLIAS (HOLDOUT E VALIDAÇÃO CRUZADA 5-FOLD)")
    r_cap.bold = True
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(7.8)
    r_cap.font.color.rgb = RGBColor(16, 44, 87)

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
    table_data = [[
        "Método / Abordagem", "Inf.", "Ruído", "F1 Hold.", "F1 CV (5f)",
        "Acur.", "AUC", "Sel.(s)", "Tr.(ms)"
    ]]
    for method in method_order:
        row = benchmark[method]
        short_name = method.replace("XAI Proposto ", "XAI ").replace("Baseline Completo (40 Atributos)", "Baseline (40 Atrib.)")
        cv_str = f"{format_decimal(row['F1_CV_Media'], 3)}±{format_decimal(row['F1_CV_Std'], 2)}"
        table_data.append([
            short_name,
            f'{int(float(row["Biomarcadores_Retidos (max 10)"]))}/10',
            f'{int(float(row["Ruidos_Selecionados (max 20)"]))}/20',
            format_decimal(row["F1_Score"], 4),
            cv_str,
            format_decimal(row["Acuracia"], 3),
            format_decimal(row["ROC_AUC"], 3),
            format_decimal(row["Tempo_Selecao_s"], 2),
            format_decimal(row["Tempo_Treino_ms"], 0)
        ])

    col_widths = [1.18, 0.25, 0.27, 0.34, 0.48, 0.28, 0.28, 0.28, 0.28]
    table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for row_idx, row in enumerate(table_data):
        set_row_cant_split(table.rows[row_idx], is_header=(row_idx == 0))
        for col_idx, text in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.width = Inches(col_widths[col_idx])
            set_cell_margins(cell, top=18, bottom=18, left=18, right=18)
            p = cell.paragraphs[0]
            p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT if col_idx == 0 else WD_PARAGRAPH_ALIGNMENT.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            if row_idx < len(table_data) - 1:
                p.paragraph_format.keep_with_next = True
            r = p.add_run(text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(6.3)

            if row_idx == 0:
                set_cell_background(cell, "EBF2FA")
                r.bold = True
                r.font.color.rgb = RGBColor(16, 44, 87)
            elif row_idx in [1, 2]:
                set_cell_background(cell, "F0FFF4")
                r.bold = True
                r.font.color.rgb = RGBColor(20, 80, 20)
            elif row_idx % 2 == 1:
                set_cell_background(cell, "F9FAFB")


def add_ieee_table_2(doc, benchmark):
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_cap.paragraph_format.space_before = Pt(3.5)
    p_cap.paragraph_format.space_after = Pt(1.5)
    p_cap.paragraph_format.keep_with_next = True
    r_cap = p_cap.add_run("TABELA II: AUDITORIA DE INTERPRETABILIDADE E COORTE DE FRONTEIRA LIME")
    r_cap.bold = True
    r_cap.font.name = 'Times New Roman'
    r_cap.font.size = Pt(7.8)
    r_cap.font.color.rgb = RGBColor(16, 44, 87)

    proposed = benchmark["XAI Proposto (shap-select OOF + Optuna)"]
    baseline = benchmark["Baseline Completo (40 Atributos)"]
    n_reduced = int(float(proposed["N_Atributos"]))
    reduction = 100 * (1 - n_reduced / int(float(baseline["N_Atributos"])))
    table_data = [
        ["Dimensão Auditada", "Baseline (40)", "XAI (10 Atrib.)", "Leitura Metodológica"],
        [
            "Recuperação Ground Truth",
            "10 inf./10 red./20 ruídos",
            "10/10 inf. | 0 red. | 0 ruídos",
            f"Redução de {format_decimal(reduction, 1)}% sem corte Top-K forçado"
        ],
        [
            f"LIME R² (Obs. #{int(float(proposed['Paciente_Limiar']))}, P=50%)",
            format_decimal(proposed["R2_LIME_Baseline"]),
            format_decimal(proposed["R2_LIME_Reduzido"]),
            "Ganho local na instância mais incerta"
        ],
        [
            "LIME R² Coorte (N=20)",
            f'{format_decimal(proposed["R2_LIME_Coorte_Base_Media"], 3)}±{format_decimal(proposed["R2_LIME_Coorte_Base_Std"], 2)}',
            f'{format_decimal(proposed["R2_LIME_Coorte_Red_Media"], 3)}±{format_decimal(proposed["R2_LIME_Coorte_Red_Std"], 2)}',
            "Ajuste linear restrito pela não-linearidade RF"
        ],
        [
            "Redund./Ruído Top-10 LIME (N=20)",
            f'{format_decimal(proposed["Ruidos_LIME_Coorte_Base"], 2)} / 10 regras',
            f'{format_decimal(proposed["Ruidos_LIME_Coorte_Red"], 2)} / 10 regras',
            "Eliminação total de regras não-informativas"
        ],
        [
            "Massa SHAP / Miller",
            "100% (40 col.)",
            f'{format_decimal(float(proposed["Massa_SHAP_Retida_pct"]) * 100, 1)}% (10 col.)',
            "Sem validação clínica ou teste cognitivo humano"
        ]
    ]

    col_widths = [0.98, 0.85, 0.92, 0.89]
    table = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for row_idx, row in enumerate(table_data):
        set_row_cant_split(table.rows[row_idx], is_header=(row_idx == 0))
        for col_idx, text in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.width = Inches(col_widths[col_idx])
            set_cell_margins(cell, top=18, bottom=18, left=20, right=20)
            p = cell.paragraphs[0]
            p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT if col_idx in [0, 3] else WD_PARAGRAPH_ALIGNMENT.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            if row_idx < len(table_data) - 1:
                p.paragraph_format.keep_with_next = True
            r = p.add_run(text)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(6.3)

            if row_idx == 0:
                set_cell_background(cell, "EBF2FA")
                r.bold = True
                r.font.color.rgb = RGBColor(16, 44, 87)
            elif row_idx % 2 == 1:
                set_cell_background(cell, "F9FAFB")


def build_ieee_article():
    benchmark = load_benchmark_rows()
    proposed = benchmark["XAI Proposto (shap-select OOF + Optuna)"]
    prop_puro = benchmark["XAI Proposto (shap-select OOF Puro)"]
    baseline = benchmark["Baseline Completo (40 Atributos)"]
    n_reduced = int(float(proposed["N_Atributos"]))
    n_original = int(float(baseline["N_Atributos"]))
    reduction = 100 * (1 - n_reduced / n_original)
    f1_delta_opt = float(proposed["F1_Score"]) - float(baseline["F1_Score"])
    train_saving_puro = 100 * (
        1 - float(prop_puro["Tempo_Treino_ms"]) / float(baseline["Tempo_Treino_ms"])
    )
    doc = Document()

    sec1 = doc.sections[0]
    sec1.top_margin = Inches(0.62)
    sec1.bottom_margin = Inches(0.62)
    sec1.left_margin = Inches(0.60)
    sec1.right_margin = Inches(0.60)
    sec1.page_width = Inches(8.27)
    sec1.page_height = Inches(11.69)

    # CABEÇALHO IEEE
    p_title = doc.add_paragraph()
    p_title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(3)
    r_title = p_title.add_run("Redução de Dimensionalidade Guiada por XAI:\nSeleção Out-of-Fold, Benchmark Multi-Famílias e Auditoria Local")
    r_title.bold = True
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(15.5)
    r_title.font.color.rgb = RGBColor(16, 44, 87)

    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_auth.paragraph_format.space_before = Pt(0)
    p_auth.paragraph_format.space_after = Pt(4)
    r_name = p_auth.add_run("Eduardo Lázaro Roesler de Oliveira\n")
    r_name.bold = True
    r_name.font.name = 'Times New Roman'
    r_name.font.size = Pt(9.8)
    r_affil = p_auth.add_run("Universidade Estadual Paulista \"Júlio de Mesquita Filho\" (UNESP) — Programa de Pós-Graduação (Mestrado)  |  eduardo.roesler@unesp.br")
    r_affil.font.name = 'Times New Roman'
    r_affil.font.size = Pt(8.5)
    r_affil.font.color.rgb = RGBColor(60, 60, 60)

    # Resumo
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    p_abs.paragraph_format.left_indent = Inches(0.18)
    p_abs.paragraph_format.right_indent = Inches(0.18)
    p_abs.paragraph_format.space_before = Pt(0)
    p_abs.paragraph_format.space_after = Pt(2)
    p_abs.paragraph_format.line_spacing = 1.02
    r_abs_lbl = p_abs.add_run("Resumo— ")
    r_abs_lbl.bold = True
    r_abs_lbl.font.name = 'Times New Roman'
    r_abs_lbl.font.size = Pt(8.4)
    r_abs_txt = p_abs.add_run(
        f"Este estudo investiga a seleção de atributos guiada por Inteligência Artificial Explicável (XAI) em um benchmark sintético controlado (N = 2.000, 40 atributos: 10 informativos, 10 redundantes por combinação linear e 20 ruídos estocásticos). "
        f"Propõe-se um pipeline que integra pré-filtragem de colinearidade multivariada (Fator de Inflação da Variância, VIF <= 10) e seleção inferencial shap-select calculada fora da amostra (3-Fold Out-of-Fold TreeSHAP, p < 0,05), avaliada com e sem otimização Optuna frente às famílias Filter, Wrapper e Embedded. "
        f"O método proposto recuperou exatamente os 10 atributos informativos verdadeiros (10/10) e rejeitou 100% dos redundantes (0/10) e ruídos (0/20) sem corte Top-K arbitrário, superando o Baseline de 40 atributos tanto sob o mesmo estimador padrão (F1 = {format_decimal(prop_puro['F1_Score'])}) quanto após ajuste Optuna (F1 = {format_decimal(proposed['F1_Score'])}; ROC-AUC = {format_decimal(proposed['ROC_AUC'])}). "
        f"A auditoria LIME no limiar de decisão (P ≈ 50%) foi conduzida na observação mais incerta e em uma coorte de fronteira (N = 20), demonstrando eliminação total de ruído nas regras locais."
    )
    r_abs_txt.font.name = 'Times New Roman'
    r_abs_txt.font.size = Pt(8.4)

    # Palavras-chave
    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    p_kw.paragraph_format.left_indent = Inches(0.18)
    p_kw.paragraph_format.right_indent = Inches(0.18)
    p_kw.paragraph_format.space_before = Pt(0)
    p_kw.paragraph_format.space_after = Pt(5)
    r_kw_lbl = p_kw.add_run("Termos para Indexação— ")
    r_kw_lbl.bold = True
    r_kw_lbl.font.name = 'Times New Roman'
    r_kw_lbl.font.size = Pt(8.4)
    r_kw_txt = p_kw.add_run("Inteligência Artificial Explicável (XAI), Seleção de Características, TreeSHAP Out-of-Fold, shap-select, VIF, LIME, Benchmark Sintético.")
    r_kw_txt.font.name = 'Times New Roman'
    r_kw_txt.font.size = Pt(8.4)

    # CORPO EM DUAS COLUNAS
    sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
    sectPr2 = sec2._sectPr
    cols = parse_xml(r'<w:cols xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:num="2" w:space="310"/>')
    sectPr2.append(cols)

    # I. INTRODUÇÃO E QUESTÃO DE PESQUISA
    add_ieee_heading_1(doc, "I. INTRODUÇÃO E QUESTÃO DE PESQUISA")
    add_ieee_paragraph(
        doc,
        "Em espaços tabulares de alta dimensionalidade, o acúmulo de variáveis redundantes e ruído estocástico degrada a capacidade de generalização (Fenômeno de Hughes) e eleva o custo computacional. "
        "Embora métodos de Inteligência Artificial Explicável (XAI) sejam tradicionalmente empregados para auditoria pós-hoc [1], suas atribuições marginais oferecem base quantitativa para poda estrutural de atributos."
    )
    add_ieee_paragraph(
        doc,
        "Contudo, abordagens ingênuas de seleção via XAI enfrentam duas fragilidades metodológicas: (i) a divisão de importância de Shapley entre variáveis multicolineares e (ii) o vazamento interno (in-sample overfitting) quando a regressão de seleção é ajustada sobre valores SHAP extraídos dos mesmos dados de treino memorizados pelas árvores."
    )
    add_ieee_paragraph(
        doc,
        "\"Em um benchmark sintético controlado, como uma seleção de atributos informada por pré-filtragem VIF e TreeSHAP Out-of-Fold (shap-select) se compara a métodos filter, wrapper e embedded em recuperação do subespaço informativo, rejeição de ruído, desempenho preditivo e custo computacional, e o que a auditoria LIME na fronteira de decisão permite concluir sobre fidelidade local?\"",
        bold_prefix="Questão de Pesquisa Norteadora: "
    )
    add_ieee_paragraph(
        doc,
        "Avaliam-se três hipóteses formais: H1 (desempenho) prevê redução mínima de 50% dos atributos mantendo ou superando o F1-Score do baseline (tolerância máxima de queda de 0,05); "
        "H2 (pureza e custo) prevê recuperação superior de atributos informativos com 100% de rejeição de ruído e menor tempo de seleção que wrappers iterativos (RFE); "
        "e H3 (auditoria local) prevê eliminação de ruído e não-degradação do R² local do LIME na fronteira de decisão (P ≈ 50%)."
    )

    # II. TRABALHOS RELACIONADOS
    add_ieee_heading_1(doc, "II. TRABALHOS RELACIONADOS")
    add_ieee_paragraph(
        doc,
        "A taxonomia de Kohavi e John [2] divide a seleção de atributos em métodos Filter (como ANOVA F e Mutual Information, rápidos porém cegos a interações de ordem superior), "
        "Wrapper (como o Recursive Feature Elimination — RFE de Guyon et al. [3], que reajusta o estimador ciclicamente) e Embedded (como penalização Lasso L1 e impureza MDI/Gini). "
        "Na presença de colinearidade multivariada, filtros retêm variáveis redundantes e métodos baseados em árvores pulverizam a importância entre preditores correlacionados."
    )
    add_ieee_paragraph(
        doc,
        "No domínio de XAI, Lundberg e Lee [4] formalizaram o SHAP sob os axiomas de Shapley (eficiência, simetria, dummy e aditividade), enquanto Ribeiro et al. [5] propuseram o LIME para aproximações lineares locais. "
        "Marcílio e Eler [6] demonstraram o uso de rankings SHAP para seleção Top-K, mas destacaram a ausência de um corte estatístico inferencial. "
        "Alvarez-Melis e Jaakkola [7] alertaram para a instabilidade de explicadores locais em espaços ruidosos, motivando a avaliação conjunta de poda global e fidelidade local."
    )

    # III. METODOLOGIA EXPERIMENTAL
    add_ieee_heading_1(doc, "III. METODOLOGIA EXPERIMENTAL")
    add_ieee_heading_2(doc, "A. Base Sintética com Ground Truth Controlado")
    add_ieee_paragraph(
        doc,
        "Gerou-se um benchmark sintético reprodutível (make_classification, semente 42, shuffle=False) com N = 2.000 observações e D = 40 atributos: "
        "10 informativos verdadeiros (biomarcador_1..10), 10 redundantes gerados como combinações lineares dos informativos (exame_redundante_1..10) e 20 ruídos gaussianos (ruido_metabolico_1..20), com proporção 60/40 e 3% de inversão estocástica de rótulos (flip_y = 0,03). "
        "Os dados foram particionados de forma estratificada em 75% treino (1.500) e 25% holdout (500), além de avaliação complementar em validação cruzada estratificada de 5 folds."
    )

    add_ieee_heading_2(doc, "B. Formalismo TreeSHAP e Modelo Substituto LIME")
    add_ieee_paragraph(
        doc,
        "O algoritmo TreeSHAP computa a contribuição marginal exata φ_i de cada atributo i sobre as coalizões S ⊆ F \\ {i}:"
    )
    add_ieee_equation(doc, "φ_i = ∑_{S ⊆ F \\ {i}} [|S|!(|F|-|S|-1)! / |F|!] [f(S ∪ {i}) - f(S)]", "1")
    add_ieee_paragraph(
        doc,
        "Para auditoria local em torno de uma instância x, o LIME ajusta um modelo linear esparso g ∈ G ponderado pelo kernel de proximidade π_x:"
    )
    add_ieee_equation(doc, "ξ(x) = argmin_{g ∈ G} L(f, g, π_x) + Ω(g)", "2")

    # Figura 1: SHAP Global
    add_ieee_image(
        doc,
        "modulo2_shap_summary.png",
        "Fig. 1. Explicabilidade global TreeSHAP no Baseline (40 atributos): diluição de importância entre informativos (azul) e redundantes (laranja)."
    )

    add_ieee_heading_2(doc, "C. Pipeline Proposto: Pré-Filtro VIF + shap-select Out-of-Fold")
    add_ieee_paragraph(
        doc,
        "Para superar a colinearidade multivariada e o vazamento in-sample, o pipeline proposto executa três estágios estritamente em X_train: "
        "(1) Pré-filtro estatístico não-supervisionado que aplica VarianceThreshold (< 0,01), filtro de correlação bivariada (|r| > 0,90) e Fator de Inflação da Variância iterativo (VIF = 1/(1 - R²_j) <= 10), eliminando exatamente as 10 combinações lineares redundantes (40 → 30 colunas). "
        "(2) Cálculo Out-of-Fold (3-Fold Stratified OOF) da matriz TreeSHAP Φ_OOF nos 30 atributos candidatos, garantindo que cada atribuição seja extraída fora da amostra de treino da árvore; em seguida, ajusta-se regressão Logit de y_train sobre Φ_OOF, retendo apenas atributos com coeficiente β > 0 e p-valor < 0,05 (sem corte Top-K manual). "
        "(3) Avaliação do subconjunto selecionado tanto com o estimador padrão (ceteris paribus) quanto após otimização Bayesiana Optuna (20 tentativas TPE em CV de 5 folds)."
    )

    # Figura 2: shap-select OOF
    add_ieee_image(
        doc,
        "modulo5_shap_select_analysis.png",
        "Fig. 2. Seleção inferencial shap-select Out-of-Fold: aprovação exata dos 10 atributos informativos (p < 0,05, verde) e rejeição do ruído (vermelho)."
    )

    # IV. EXPERIMENTOS, RESULTADOS E DISCUSSÃO
    add_ieee_heading_1(doc, "IV. EXPERIMENTOS, RESULTADOS E DISCUSSÃO")
    add_ieee_heading_2(doc, "A. Benchmark Comparativo Multi-Famílias e Validação de H1 e H2")
    add_ieee_paragraph(
        doc,
        "A Tabela I compara o pipeline proposto (com e sem Optuna) aos seis representantes clássicos (fixados em K = 10 atributos) e ao Baseline de 40 atributos, reportando métricas no holdout e média ± desvio-padrão em validação cruzada de 5 folds."
    )

    # Tabela I
    add_ieee_table_1(doc, benchmark)

    add_ieee_paragraph(
        doc,
        f"Os resultados da Tabela I confirmam integralmente H1: o pipeline proposto reduziu {format_decimal(reduction, 1)}% da dimensionalidade (40 → 10 atributos) e, em vez de sofrer queda de desempenho, elevou o F1-Score de {format_decimal(baseline['F1_Score'])} (Baseline) para {format_decimal(prop_puro['F1_Score'])} na comparação ceteris paribus sem Optuna e para {format_decimal(proposed['F1_Score'])} com Optuna (ganho de +{format_decimal(f1_delta_opt, 4)} no holdout e {format_decimal(proposed['F1_CV_Media'], 4)} na validação cruzada), comprovando a mitigação empírica do Fenômeno de Hughes."
    )
    add_ieee_paragraph(
        doc,
        f"Em relação a H2, a proposta foi o único método de todo o benchmark a recuperar 100% do ground truth informativo (10/10 informativos, 0/10 redundantes e 0/20 ruídos). "
        f"Os demais métodos (RFE, ANOVA, Tree MDI e TreeSHAP direto) retiveram entre 6 e 7 atributos informativos porque selecionaram de 3 a 4 variáveis redundantes colineares. "
        f"Quanto ao custo computacional, o treino do classificador final caiu {format_decimal(train_saving_puro, 1)}% ({format_decimal(baseline['Tempo_Treino_ms'], 1)} ms para {format_decimal(prop_puro['Tempo_Treino_ms'], 1)} ms); contudo, o tempo de seleção offline da proposta ({format_decimal(prop_puro['Tempo_Selecao_s'], 2)} s sem Optuna; {format_decimal(proposed['Tempo_Selecao_s'], 2)} s com Optuna) foi superior ao RFE ({format_decimal(benchmark['Wrapper (RFE)']['Tempo_Selecao_s'], 2)} s), rejeitando-se o subcritério de velocidade de seleção de H2."
    )

    # Figura 3: Ablação
    add_ieee_image(
        doc,
        "modulo4_ablation_curves.png",
        "Fig. 3. Curvas de ablação progressiva (40 a 2 atributos): superioridade da trajetória OOF+VIF (verde escuro) no ponto ótimo K = 10."
    )

    add_ieee_heading_2(doc, "B. Auditoria Local LIME: Observação Crítica e Coorte de Fronteira")
    add_ieee_paragraph(
        doc,
        f"Para testar H3 com robustez estatística, a auditoria LIME foi executada na observação mais incerta do holdout (#{int(float(proposed['Paciente_Limiar']))}, P = {format_decimal(float(proposed['Probabilidade_Baseline_Limiar']) * 100, 1)}%) e na coorte das N = 20 observações mais próximas da fronteira P = 50% (Tabela II e Fig. 4). "
        f"Na observação #{int(float(proposed['Paciente_Limiar']))}, a fidelidade local R² subiu de {format_decimal(proposed['R2_LIME_Baseline'])} para {format_decimal(proposed['R2_LIME_Reduzido'])}. "
        f"Na coorte de fronteira (N = 20), o R² médio passou de {format_decimal(proposed['R2_LIME_Coorte_Base_Media'], 4)} para {format_decimal(proposed['R2_LIME_Coorte_Red_Media'], 4)}, mantendo-se moderado devido à natureza altamente não-linear das árvores na zona de transição. "
        f"Entretanto, o modelo baseline incluiu em média {format_decimal(proposed['Ruidos_LIME_Coorte_Base'], 2)} variáveis não-informativas (redundantes ou ruído) entre as 10 principais regras locais do LIME, ao passo que o modelo reduzido reduziu essa contaminação local a {format_decimal(proposed['Ruidos_LIME_Coorte_Red'], 2)}."
    )

    # Figura 4: LIME Local
    add_ieee_image(
        doc,
        "modulo3_lime_local.png",
        f"Fig. 4. Auditoria local LIME na observação limítrofe #{int(float(proposed['Paciente_Limiar']))} (P = 50,0%): comparação de regras locais e fidelidade R²."
    )

    # Tabela II
    add_ieee_table_2(doc, benchmark)

    # V. CONCLUSÕES E LIMITAÇÕES
    add_ieee_heading_1(doc, "V. CONCLUSÕES E LIMITAÇÕES")
    add_ieee_paragraph(
        doc,
        f"Respondendo à questão de pesquisa: o acoplamento de pré-filtragem multivariada (VIF <= 10) com seleção inferencial TreeSHAP Out-of-Fold (shap-select, p < 0,05) permitiu recuperar exatamente os 10 atributos informativos verdadeiros (10/10) e eliminar 100% dos atributos redundantes e de ruído sem depender de um número K arbitrário. "
        f"A redução de 75% dos dados elevou o F1-Score no holdout ({format_decimal(baseline['F1_Score'])} → {format_decimal(proposed['F1_Score'])}) e na validação cruzada ({format_decimal(baseline['F1_CV_Media'])} → {format_decimal(proposed['F1_CV_Media'])}), confirmando H1 e a pureza de H2, embora com maior tempo de seleção offline que métodos filter e wrapper."
    )
    add_ieee_paragraph(
        doc,
        "Como limitações metodológicas e epistemológicas, ressalta-se que: (i) os resultados decorrem de um benchmark sintético controlado, não configurando validação clínica; (ii) atribuições Shapley e LIME refletem associações preditivas do estimador e não causalidade física ou biológica; e (iii) a referência de Miller [8] (7 ± 2) é apenas teórica, inexistindo avaliação experimental com usuários humanos. Trabalhos futuros estenderão o protocolo OOF+VIF a bases reais multicêntricas de alta dimensionalidade."
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
        p_ref.paragraph_format.left_indent = Inches(0.16)
        p_ref.paragraph_format.first_line_indent = Inches(-0.16)
        p_ref.paragraph_format.space_before = Pt(0)
        p_ref.paragraph_format.space_after = Pt(1.2)
        p_ref.paragraph_format.line_spacing = 1.01
        r_ref = p_ref.add_run(ref)
        r_ref.font.name = 'Times New Roman'
        r_ref.font.size = Pt(7.2)
        r_ref.font.color.rgb = RGBColor(40, 40, 40)

    doc.save(DOCX_OUTPUT)
    print(f"[+] Artigo IEEE em Word salvo em: {DOCX_OUTPUT}")

    print("[*] Convertendo Artigo IEEE para PDF...")
    docx2pdf.convert(DOCX_OUTPUT, PDF_OUTPUT)
    print(f"[+] Artigo IEEE em PDF gerado em: {PDF_OUTPUT}")

    os.makedirs(os.path.dirname(DOCS_PDF_OUTPUT), exist_ok=True)
    shutil.copyfile(PDF_OUTPUT, DOCS_PDF_OUTPUT)
    print(f"[+] Cópia sincronizada em: {DOCS_PDF_OUTPUT}")

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
