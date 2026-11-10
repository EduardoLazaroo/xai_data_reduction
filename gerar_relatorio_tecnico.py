"""
===================================================================================
GERADOR DO RELATÓRIO TÉCNICO COMPLETO DO PROJETO (DOCX & PDF)
===================================================================================
Projeto: Redução de Dimensionalidade Guiada por Explicabilidade (XAI)
Instituição: Centro Universitário Eurípides de Marília (UNIVEM)
Autor: Eduardo Lázaro Roesler de Oliveira
Entrega: 10/11/2026

Formatação:
- Página padrão (A4, coluna única, padrão acadêmico completo / ABNT)
- Margens: Superior 3cm, Esquerda 3cm, Inferior 2cm, Direita 2cm
- Tipografia: Times New Roman 12pt, entrelinhas 1.15, texto justificado
- Elementos: Capa institucional, Resumo, Fundamentação Matemática, Metodologia,
  Tabelas comparativas, 6 Figuras e Diagnósticos em Alta Resolução (300 DPI),
  Auditoria Forense LIME, Análise Multicritério e Referências Bibliográficas.
===================================================================================
"""

import os
import shutil
import csv
import docx
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls
import docx2pdf
import pypdf

BASE_DIR = r"c:\Users\eduar\projetos\xai_data_reduction"
DOCS_DIR = os.path.join(BASE_DIR, "docs")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

OUTPUT_DOCX = os.path.join(DOCS_DIR, "relatorio_tecnico_projeto.docx")
OUTPUT_PDF = os.path.join(DOCS_DIR, "relatorio_tecnico_projeto.pdf")
ROOT_DOCX = os.path.join(BASE_DIR, "relatorio_tecnico_projeto.docx")
ROOT_PDF = os.path.join(BASE_DIR, "relatorio_tecnico_projeto.pdf")


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


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(17)
    run.font.color.rgb = RGBColor(16, 44, 87)
    return p


def add_subtitle(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(14)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.italic = True
    run.font.color.rgb = RGBColor(53, 89, 143)
    return p


def add_author(doc, name, affiliation, email):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p.paragraph_format.space_after = Pt(18)
    
    r_name = p.add_run(f"{name}\n")
    r_name.bold = True
    r_name.font.name = 'Times New Roman'
    r_name.font.size = Pt(11)
    
    r_affil = p.add_run(f"{affiliation}\n")
    r_affil.font.name = 'Times New Roman'
    r_affil.font.size = Pt(10)
    
    r_email = p.add_run(email)
    r_email.font.name = 'Times New Roman'
    r_email.font.size = Pt(9.5)
    r_email.font.italic = True
    r_email.font.color.rgb = RGBColor(80, 80, 80)
    return p


def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(16, 44, 87)
    return p


def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(53, 89, 143)
    return p


def add_paragraph(doc, text, bold_prefix="", indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.4)
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11)
    
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(30, 30, 30)
    return p


def add_callout_box(doc, text, title=""):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:left w:val="single" w:sz="28" w:space="0" w:color="102C57"/>\n'
        f'  <w:bottom w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(0)
    
    if title:
        rt = p.add_run(f"📌 {title}\n")
        rt.bold = True
        rt.font.name = 'Times New Roman'
        rt.font.size = Pt(10.5)
        rt.font.color.rgb = RGBColor(16, 44, 87)
    
    rtxt = p.add_run(text)
    rtxt.font.name = 'Times New Roman'
    rtxt.font.size = Pt(10)
    rtxt.font.italic = True
    rtxt.font.color.rgb = RGBColor(40, 40, 40)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(6)


def add_image_with_caption(doc, filename, caption, width_inches=6.2):
    img_path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(img_path):
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(img_path, width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)
    p_cap.paragraph_format.keep_with_next = False
    
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = 'Times New Roman'
    run_cap.font.size = Pt(9.5)
    run_cap.font.italic = True
    run_cap.font.color.rgb = RGBColor(70, 70, 70)


def build_full_report():
    benchmark = load_benchmark_rows()
    proposed = benchmark["XAI Proposto (shap-select + Optuna)"]
    baseline = benchmark["Baseline Completo (40 Atributos)"]
    n_reduced = int(float(proposed["N_Atributos"]))
    reduction = 100 * (1 - n_reduced / int(float(baseline["N_Atributos"])))
    f1_delta = float(proposed["F1_Score"]) - float(baseline["F1_Score"])
    doc = Document()

    # Configuração de margens A4
    for s in doc.sections:
        s.top_margin = Inches(1.18)      # 3 cm
        s.left_margin = Inches(1.18)     # 3 cm
        s.bottom_margin = Inches(0.79)   # 2 cm
        s.right_margin = Inches(0.79)    # 2 cm
        s.page_width = Inches(8.27)
        s.page_height = Inches(11.69)

    # =========================================================================
    # CABEÇALHO DO RELATÓRIO TÉCNICO
    # =========================================================================
    add_title(doc, "RELATÓRIO TÉCNICO E MONOGRAFIA EXPERIMENTAL:\nREDUÇÃO DE DADOS GUIADA POR EXPLICABILIDADE (XAI)")
    add_subtitle(doc, "Benchmark Sintético, Seleção de Atributos e Auditoria Local")
    add_author(
        doc,
        name="Eduardo Lázaro Roesler de Oliveira",
        affiliation="Programa de Ciência da Computação / Pós-Graduação\nCentro Universitário Eurípides de Marília (UNIVEM)",
        email="eduardo.roesler@univem.edu.br | Data da Documentação: 10/11/2026"
    )

    # =========================================================================
    # RESUMO EXECUTIVO
    # =========================================================================
    add_heading_1(doc, "Resumo Executivo")
    add_paragraph(
        doc,
        f"Este relatorio documenta um benchmark sintetico de selecao de atributos, sem dados de pacientes ou validacao clinica. "
        f"Foram comparados metodos filter, wrapper, embedded e uma proposta com TreeSHAP, pre-filtro, regressao sobre atribuicoes SHAP (shap-select) e Optuna. "
        f"No holdout desta execucao, a proposta reteve {n_reduced} de 40 atributos ({reduction:.1f}% de reducao); "
        f"o F1 foi {format_decimal(baseline['F1_Score'])} no baseline e {format_decimal(proposed['F1_Score'])} na proposta "
        f"(delta {format_decimal(f1_delta)}). A taxa de rejeicao de ruido foi {proposed['Taxa_Rejeicao_Ruido']}. "
        f"O tempo de selecao foi {format_decimal(proposed['Tempo_Selecao_s'], 2)} s para a proposta e "
        f"{format_decimal(benchmark['Wrapper (RFE)']['Tempo_Selecao_s'], 2)} s para RFE; portanto, a proposta nao foi mais rapida nesta medicao. "
        f"O R² LIME e a massa SHAP retida sao diagnosticos do modelo; uma observacao local nao demonstra fidelidade geral, causalidade ou compreensao humana. "
        "As metricas sao descritivas de uma particao sintetica e nao sustentam conclusoes de uso clinico ou escalabilidade hospitalar.",
        indent=False
    )
    add_paragraph(
        doc,
        "Inteligência Artificial Explicável (XAI); Redução de Dimensionalidade; Seleção de Características; TreeSHAP; LIME; Dados Sintéticos; Benchmark.",
        bold_prefix="Palavras-chave: ",
        indent=False
    )

    # =========================================================================
    # 1. INTRODUÇÃO E CONTEXTUALIZAÇÃO DO PROBLEMA
    # =========================================================================
    add_heading_1(doc, "1. Introdução e Contextualização do Problema")
    add_paragraph(
        doc,
        "O avanço substancial de algoritmos de aprendizado de máquina supervisionado na última década consolidou modelos baseados em comitês de árvores, "
        "em particular o Random Forest e variantes de gradient boosting, como ferramentas padrão ouro para tarefas de classificação complexas. "
        "Na área biomédica e nos sistemas de apoio à decisão clínica, esses modelos têm apresentado elevados índices de precisão diagnóstica. "
        "Contudo, a disseminação desses modelos tem sido acompanhada por uma prática comum e arriscada: a coleta massiva e indiscriminada de dados "
        "de alta dimensionalidade."
    )
    add_paragraph(
        doc,
        "No início da formação em ciência de dados, prevalece a intuição empírica ingênua de que 'quanto mais variáveis forem fornecidas ao modelo, mais "
        "inteligente e robusto ele se tornará'. No contexto médico, essa premissa é fundamentalmente falsa e economicamente contraproducente. "
        "Em um hospital ou unidade de terapia intensiva, cada coluna adicional em uma tabela de dados representa um exame de sangue, uma tomografia "
        "computadorizada, uma biópsia ou um monitoramento contínuo. Tais procedimentos acarretam custos financeiros expressivos, tempo de espera prolongado "
        "para o paciente e desgaste físico desnecessário. Além disso, em dispositivos médicos embarcados de triagem rápida (Edge AI), processar dezenas de "
        "parâmetros introduz uma latência computacional que pode ser fatal."
    )
    add_paragraph(
        doc,
        "Do ponto de vista algorítmico, os modelos supervisionados são caçadores de correlações matemáticas. Quando um conjunto de dados contém dezenas de "
        "atributos que representam puro ruído estocástico, o estimador inevitavelmente identificará associações espúrias por mero acaso probabilístico, "
        "levando a uma falsa sensação de desempenho no conjunto de treino e à degradação severa da generalização no mundo real (overfitting)."
    )
    add_paragraph(
        doc,
        "Paralelamente, a área de Explainable Artificial Intelligence (XAI) tem sido subutilizada no ecossistema de machine learning. Na quase totalidade "
        "das aplicações comerciais e acadêmicas, técnicas de XAI como SHAP (Shapley Additive Explanations) e LIME (Local Interpretable Model-agnostic Explanations) "
        "são tratadas apenas como uma camada cosmética analítica pós-processamento: ajusta-se um modelo complexo de 'caixa-preta' e, posteriormente, usa-se a "
        "explicabilidade apenas para observar como a decisão foi tomada. Este trabalho propõe uma mudança radical de paradigma: transformar a explicabilidade "
        "em um motor ativo de engenharia de dados e seleção de características."
    )

    # =========================================================================
    # 2. QUESTÃO DE PESQUISA, OBJETIVOS E HIPÓTESES FORMAIS
    # =========================================================================
    add_heading_1(doc, "2. Questão de Pesquisa, Objetivos e Hipóteses Formais")
    add_callout_box(
        doc,
        "Questão de Pesquisa Norteadora:\n"
        "\"Em um benchmark sintético controlado, como uma seleção informada por TreeSHAP e shap-select se compara a métodos filter, wrapper e embedded "
        "em desempenho preditivo, rejeição de ruído e custo de seleção, e o que uma auditoria LIME de uma observação permite concluir sobre fidelidade local?\"",
        title="FIO CONDUTOR DO PROJETO"
    )

    add_paragraph(
        doc,
        "Para responder a essa questão norteadora, estabeleceram-se os seguintes objetivos:",
        bold_prefix="Objetivos do Trabalho: "
    )
    add_paragraph(
        doc,
        "1. Objetivo Geral: Comparar uma seleção informada por TreeSHAP, com pré-filtro e shap-select, a métodos clássicos em um benchmark sintético, "
        "considerando redução dimensional, desempenho preditivo, ruído selecionado e custo medido.",
        indent=True
    )
    add_paragraph(
        doc,
        "2. Objetivos Específicos:\n"
        "   a) Treinar um modelo Random Forest Baseline sobre a totalidade dos dados (40 atributos) e registrar sua linha de base preditiva e computacional;\n"
        "   b) Ampliar o horizonte metodológico com baselines representativos das famílias Filter (Mutual Information e ANOVA F), Wrapper (RFE) e Embedded (Lasso L1 e Tree MDI);\n"
        "   c) Usar TreeSHAP como atribuição global e LIME para inspecionar uma observação do holdout próxima de P = 50%;\n"
        "   d) Desenvolver a metodologia shap-select acoplada a pré-filtro híbrido (variância + colinearidade) e regressão com p-valor < 0,05;\n"
        "   e) Executar um estudo de ablação progressiva multivariada (de 40 a 2 atributos) comparando todas as famílias;\n"
        "   f) Reportar proxies do modelo (massa SHAP retida e fidelidade local LIME), sem alegar causalidade ou benefício cognitivo não medido.",
        indent=True
    )

    add_paragraph(
        doc,
        "O estudo foi guiado por três hipóteses científicas testadas empiricamente no protocolo:",
        bold_prefix="Hipóteses de Pesquisa: "
    )
    add_paragraph(
        doc,
        "• Hipótese H1 (Integridade Preditiva): No holdout sintético, a proposta descarta pelo menos 50% dos atributos sem queda de F1 superior a 0,05 frente ao baseline.",
        indent=True
    )
    add_paragraph(
        doc,
        "• Hipótese H2 (Pureza e Eficiência Comparativa): A proposta rejeita mais atributos de ruído que os comparadores e tem menor tempo de seleção que RFE nesta execução.",
        indent=True
    )
    add_paragraph(
        doc,
        "• Hipótese H3 (Auditoria Local): O R² do LIME não diminui para a observação selecionada. Esse teste de uma instância não mede fidelidade global nem cognição humana.",
        indent=True
    )

    # =========================================================================
    # 3. TRABALHOS RELACIONADOS E ESTADO DA ARTE
    # =========================================================================
    add_heading_1(doc, "3. Trabalhos Relacionados e Estado da Arte")
    add_paragraph(
        doc,
        "A taxonomia clássica de seleção de características proposta por Kohavi e John (1997) divide as abordagens tradicionais em três famílias:"
    )
    add_paragraph(
        doc,
        "1. Família Filter (Filtros Univariados): Avaliam cada atributo de forma isolada através de critérios puramente estatísticos ou teóricos da informação "
        "(como teste ANOVA F e Ganho de Informação / Mutual Information). Possuem baixíssimo custo computacional, mas padecem de uma fraqueza congênita: "
        "ignoram interações de segunda e terceira ordem entre variáveis, além de selecionarem atributos altamente colineares entre si.",
        indent=True
    )
    add_paragraph(
        doc,
        "2. Família Wrapper: Utilizam o próprio modelo de aprendizado como função de avaliação para subconjuntos de atributos. O expoente dessa categoria é o "
        "Recursive Feature Elimination (RFE) proposto por Guyon et al. (2002). Embora muito eficaz na preservação do desempenho, apresenta custo computacional "
        "proibitivo, pois exige o retreinamento do estimador centenas de vezes a cada corte dimensional.",
        indent=True
    )
    add_paragraph(
        doc,
        "3. Família Embedded (Embutidos): A seleção ocorre intrinsecamente durante o processo de treinamento do algoritmo. Exemplos notáveis incluem a penalização "
        "L1 (Lasso) em modelos lineares e as métricas de impureza de nós (Mean Decrease Impurity - MDI / Gini) em árvores de decisão. Contudo, em cenários com variáveis "
        "correlacionadas, métodos Lasso tendem a selecionar arbitrariamente apenas uma e descartar as demais, enquanto o MDI sobrevaloriza atributos numéricos contínuos.",
        indent=True
    )
    add_paragraph(
        doc,
        "No campo da explicabilidade, Lundberg e Lee (2017) revolucionaram o estado da arte ao introduzirem o SHAP (Shapley Additive Explanations), fundamentado "
        "na Teoria dos Jogos Cooperativos de Lloyd Shapley (Prêmio Nobel de Economia). O SHAP garante quatro propriedades axiomáticas fundamentais: "
        "Eficiência (a soma das contribuições marginais equivale à diferença entre a previsão e a expectativa base), Simetria (atributos com contribuições idênticas recebem pesos iguais), "
        "Dummy (atributos que não alteram o desfecho recebem atribuição zero) e Aditividade. Ribeiro, Singh e Guestrin (2016) propuseram o LIME, voltado a aproximações lineares locais."
    )
    add_paragraph(
        doc,
        "Marcílio e Eler (2020) demonstraram que a utilização de SHAP para ranqueamento de variáveis pode aprimorar modelos, mas indicaram a necessidade de estudos "
        "comparativos sistemáticos com as famílias clássicas de baselines e critérios rigorosos de auditoria local. Este relatório atende a essa recomendação."
    )

    # =========================================================================
    # 4. METODOLOGIA EXPERIMENTAL E PROCEDIMENTO
    # =========================================================================
    add_heading_1(doc, "4. Metodologia Experimental e Procedimento")
    
    add_heading_2(doc, "4.1 Geração do Conjunto de Dados Sintético")
    add_paragraph(
        doc,
        "Para avaliar a recuperação de grupos conhecidos por construção, geraram-se 2.000 observações sintéticas e 40 atributos com make_classification. "
        "A geracao usa shuffle=False para manter a ordem dos grupos. Os nomes informativos, redundantes e de ruído são rótulos do experimento; não representam exames, pacientes ou causas biológicas:"
    )
    add_paragraph(
        doc,
        "• 10 atributos informativos para a classe sintética;\n"
        "• 10 atributos redundantes gerados pelo make_classification;\n"
        "• 20 atributos de ruído, sem sinal informativo intencional.",
        indent=True
    )
    add_paragraph(
        doc,
        "A base foi dividida de maneira estratificada em 75% para treinamento (1.500 observações) e 25% para holdout (500 observações). O holdout é uma única partição, não validação externa."
    )

    add_heading_2(doc, "4.2 Modelo de Referência (Baseline)")
    add_paragraph(
        doc,
        "Ajustou-se um classificador Random Forest com 100 árvores de decisão sobre a totalidade dos 40 atributos. A Figura 1 apresenta o desempenho da linha de base."
    )
    add_image_with_caption(
        doc,
        "modulo1_baseline_metrics.png",
        "Figura 1. Desempenho Diagnóstico do Modelo Baseline (40 Atributos): (A) Matriz de Confusão em Pacientes de Teste; (B) Curva ROC demonstrando AUC = 0,9475."
    )

    add_heading_2(doc, "4.3 Explicabilidade Global via TreeSHAP")
    add_paragraph(
        doc,
        "Utilizou-se o algoritmo TreeSHAP (Lundberg et al., 2020), que calcula os valores exatos de Shapley para conjuntos baseados em árvores em tempo polinomial O(TLD²). "
        "A Figura 2 consolida a importância aditiva global e o gráfico beeswarm de impacto direcional."
    )
    add_image_with_caption(
        doc,
        "modulo2_shap_summary.png",
        "Figura 2. Diagnóstico Explicativo Global TreeSHAP: (A) Ranking dos 15 atributos mais importantes por média absoluta de Shapley; (B) Gráfico Beeswarm exibindo como valores altos (vermelho) ou baixos (azul) influenciam o risco de patologia."
    )

    add_heading_2(doc, "4.4 Demarcação Teórica: Papel Estrito do LIME como Auditor Forense Local")
    add_paragraph(
        doc,
        "Uma das principais diretrizes metodológicas deste projeto é a separação formal entre os papéis de SHAP e LIME. "
        "O LIME constrói um modelo substituto linear perturbando localmente os dados ao redor de uma única instância. Conforme demonstrado por Alvarez-Melis e Jaakkola (2018), "
        "o LIME sofre de instabilidade por amostragem estocástica quando aplicado à seleção global de variáveis. Logo, determinou-se:"
    )
    add_callout_box(
        doc,
        "• TreeSHAP: Responsável exclusivo pelo ranqueamento e seleção de variáveis no nível populacional global.\n"
        "• LIME: Usado para avaliar o ajuste local do modelo substituto (R²) em uma observação do holdout cuja probabilidade do baseline está próxima de 0,5.",
        title="DEMARCAÇÃO METODOLÓGICA CATEGÓRICA"
    )

    add_heading_2(doc, "4.5 Pré-Filtro Híbrido e Técnica shap-select")
    add_paragraph(
        doc,
        "O pipeline avançado integra um Pré-Filtro Híbrido (VariânciaThreshold com limiar 0,01 e remoção de correlação absoluta de Pearson > 0,90) "
        "com a técnica shap-select: ajusta-se Logit aos valores SHAP e retêm-se até 10 atributos com coeficiente positivo e p < 0,05. "
        "Se o Logit não convergir ou falhar nas exceções previstas, o código informa e usa OLS como fallback; p-valores nesse fallback requerem interpretação cautelosa. "
        "A Figura 3 detalha os p-valores e a distribuição de coeficientes obtidos."
    )
    add_image_with_caption(
        doc,
        "modulo5_shap_select_analysis.png",
        "Figura 3. Coeficientes da regressão sobre as contribuições SHAP; as cores indicam os critérios de seleção estatística usados."
    )

    add_heading_2(doc, "4.6 Sintonização Bayesiana no Espaço Reduzido via Optuna")
    add_paragraph(
        doc,
        f"Após a seleção de {n_reduced} atributos, o Optuna via TPE executou 15 tentativas com validação cruzada de 3 folds no treino. "
        "A semente do sampler é fixada; tempos de execução continuam dependentes da máquina."
    )

    # =========================================================================
    # 5. EXPERIMENTOS, RESULTADOS E DISCUSSÃO COMPARATIVA
    # =========================================================================
    add_heading_1(doc, "5. Experimentos, Resultados e Discussão Comparativa")
    add_paragraph(
        doc,
        "As seleções por ranking foram avaliadas com teto K = 10; a proposta shap-select pode reter até esse limite. "
        "O baseline usa os 40 atributos. A Tabela 1 é gerada diretamente do CSV exportado pelo pipeline."
    )

    # Tabela Mestre
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
        "Método de Seleção", "Família", "Info/10", "Ruído/20",
        "Rejeição", "F1", "Acurácia", "ROC-AUC", "Treino (ms)"
    ]]
    for method in method_order:
        row = benchmark[method]
        table_data.append([
            method,
            row["Familia"],
            f'{int(float(row["Biomarcadores_Retidos (max 10)"]))}/10',
            f'{int(float(row["Ruidos_Selecionados (max 20)"]))}/20',
            row["Taxa_Rejeicao_Ruido"],
            format_decimal(row["F1_Score"]),
            format_decimal(row["Acuracia"]),
            format_decimal(row["ROC_AUC"]),
            format_decimal(row["Tempo_Treino_ms"], 1)
        ])

    t = doc.add_table(rows=len(table_data), cols=len(table_data[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = t.cell(r_idx, c_idx)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT if c_idx == 0 else WD_PARAGRAPH_ALIGNMENT.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(val)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(8.5)
            if r_idx == 0:
                set_cell_background(cell, "102C57")
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
            elif r_idx == 1:
                set_cell_background(cell, "EBF3E8")
                run.bold = True
                run.font.color.rgb = RGBColor(20, 80, 20)
            elif r_idx % 2 == 0:
                set_cell_background(cell, "F8F9FA")

    add_paragraph(doc, "")

    add_heading_2(doc, "5.1 Estudo de Ablação Progressiva")
    add_paragraph(
        doc,
        "Avaliou-se o comportamento de todas as famílias ao longo de uma poda progressiva de 40 até 2 variáveis. A figura descreve este holdout; não demonstra estabilidade entre amostras."
    )
    add_image_with_caption(
        doc,
        "modulo4_ablation_curves.png",
        "Figura 4. Estudo de Ablação Progressiva Multivariada (40 a 2 atributos): Comparação sistemática da degradação de Acurácia e F1-Score em todas as famílias de seleção."
    )

    add_heading_2(doc, "5.2 Auditoria Local de uma Observação (LIME)")
    add_paragraph(
        doc,
        f"Na observacao {int(float(proposed['Paciente_Limiar']))}, selecionada por ter a probabilidade do baseline mais proxima de 0,5 "
        f"({float(proposed['Probabilidade_Baseline_Limiar']):.1%}), o R² do substituto foi "
        f"{format_decimal(proposed['R2_LIME_Baseline'])} no baseline e "
        f"{format_decimal(proposed['R2_LIME_Reduzido'])} no modelo reduzido (Figura 5). "
        "Este valor local, relativamente baixo, não avalia a fidelidade em outras observações."
    )
    add_image_with_caption(
        doc,
        "modulo3_lime_local.png",
        f"Figura 5. Explicação LIME para uma observação do holdout: pesos locais no baseline (40 atributos) e no modelo reduzido ({n_reduced} atributos)."
    )

    add_heading_2(doc, "5.3 Dashboard Executivo Integrador")
    add_paragraph(
        doc,
        "A Figura 6 sintetiza métricas preditivas, tempos medidos e proxies de explicabilidade neste benchmark sintético."
    )
    add_image_with_caption(
        doc,
        "dashboard_final_comparativo.png",
        "Figura 6. Dashboard comparativo: (1) número de atributos; (2) tempos medidos de treino e inferência; (3) métricas preditivas; (4) R² LIME local e massa SHAP do baseline retida."
    )

    # =========================================================================
    # 6. CONCLUSÕES E LIMITAÇÕES DO ESTUDO
    # =========================================================================
    add_heading_1(doc, "6. Conclusões e Limitações do Estudo")
    add_paragraph(
        doc,
        f"Neste benchmark sintetico, a proposta reduziu {reduction:.1f}% dos atributos e o F1 variou "
        f"{format_decimal(f1_delta)} em relacao ao baseline. O criterio de H1 foi atendido nesta particao; "
        f"por outro lado, a selecao levou {format_decimal(proposed['Tempo_Selecao_s'], 2)} s frente a "
        f"{format_decimal(benchmark['Wrapper (RFE)']['Tempo_Selecao_s'], 2)} s para RFE, portanto H2 nao foi atendida em custo."
    )
    add_paragraph(
        doc,
        f"A proposta reteve {n_reduced} atributos, rejeitou {proposed['Taxa_Rejeicao_Ruido']} do ruido sintetico e reteve "
        f"{float(proposed['Massa_SHAP_Retida_pct']):.1%} da massa SHAP global do baseline. O LIME aumentou de "
        f"{format_decimal(proposed['R2_LIME_Baseline'])} para {format_decimal(proposed['R2_LIME_Reduzido'])} nesta observacao, "
        "mas a fidelidade local observada não estabelece qualidade geral de explicações."
    )
    add_paragraph(
        doc,
        "Como limitações, os dados são sintéticos, a avaliação usa uma única divisão treino/holdout e não inclui incerteza estatística, validação externa ou avaliação humana. "
        "Os nomes dos atributos não representam biomarcadores reais e TreeSHAP/LIME não permitem inferir causalidade. "
        "Trabalhos futuros podem repetir a avaliação sob múltiplas sementes, relatar intervalos de incerteza e, só então, investigar dados reais com governança e aprovação apropriadas."
    )

    # =========================================================================
    # REFERÊNCIAS BIBLIOGRÁFICAS
    # =========================================================================
    add_heading_1(doc, "Referências Bibliográficas")
    referencias = [
        "CHANDRASHEKAR, G.; SAHIN, F. A survey on feature selection methods. Computers & Electrical Engineering, v. 40, n. 1, p. 16-28, 2014.",
        "COVERT, I.; LUNDBERG, S. M.; LEE, S.-I. Explaining by removing: A unified framework for model explanations. Journal of Machine Learning Research (JMLR), v. 22, p. 1-90, 2021.",
        "FRYER, D.; STRÜMKE, I.; NGUYEN, H. Shapley values for feature selection: The good, the bad, and the axioms. IEEE Access, v. 9, p. 144352-144360, 2021.",
        "GUYON, I.; ELISSEEFF, A. An introduction to variable and feature selection. Journal of Machine Learning Research (JMLR), v. 3, p. 1157-1182, 2003.",
        "LUNDBERG, S. M.; LEE, S.-I. A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems (NeurIPS), v. 30, p. 4765-4774, 2017.",
        "MARCÍLIO, W. E.; ELER, D. M. From explanations to feature selection: assessing SHAP values for feature selection. In: SIBGRAPI Conference on Graphics, Patterns and Images, 33., 2020. Proceedings [...]. IEEE, 2020. p. 351-358.",
        "MILLER, G. A. The magical number seven, plus or minus two: Some limits on our capacity for processing information. Psychological Review, v. 63, n. 2, p. 81-97, 1956.",
        "MOLNAR, C. Interpretable Machine Learning: A Guide for Making Black Box Models Explainable. 2. ed. Munich: Leanpub, 2022.",
        "NIST - National Institute of Standards and Technology. Artificial Intelligence Risk Management Framework (AI RMF 1.0). NIST Trustworthy and Responsible AI, Washington: U.S. Department of Commerce, 2023. DOI: 10.6028/NIST.CSWP.29.",
        "RIBEIRO, M. T.; SINGH, S.; GUESTRIN, C. 'Why should I trust you?': Explaining the predictions of any classifier. In: ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 22., 2016. Proceedings [...]. New York: ACM, 2016. p. 1135-1144.",
        "SLACK, D. et al. Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods. In: AAAI/ACM Conference on AI, Ethics, and Society (AIES), 2020. Proceedings [...]. New York: ACM, 2020. p. 180-186."
    ]

    for ref in referencias:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.first_line_indent = Inches(0.0)
        r = p_ref.add_run(ref)
        r.font.name = 'Times New Roman'
        r.font.size = Pt(9.5)

    os.makedirs(DOCS_DIR, exist_ok=True)
    doc.save(OUTPUT_DOCX)
    shutil.copyfile(OUTPUT_DOCX, ROOT_DOCX)
    print(f"[+] Relatório Técnico em Word salvo em: {OUTPUT_DOCX}")

    print("[*] Convertendo Relatório Técnico para PDF...")
    docx2pdf.convert(OUTPUT_DOCX, OUTPUT_PDF)
    shutil.copyfile(OUTPUT_PDF, ROOT_PDF)
    print(f"[+] Relatório Técnico em PDF gerado em: {OUTPUT_PDF}")

    reader = pypdf.PdfReader(OUTPUT_PDF)
    total_pages = len(reader.pages)
    print(f"\n=======================================================")
    print(f"VERIFICAÇÃO DO RELATÓRIO TÉCNICO COMPLETO:")
    print(f"Total de páginas geradas: {total_pages} páginas")
    print(f"=======================================================\n")
    return total_pages


if __name__ == "__main__":
    build_full_report()
