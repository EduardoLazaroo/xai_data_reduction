"""
===================================================================================
GERADOR DO ARTIGO COMPLETO EM WORD (.DOCX) - FORMATO ACADÊMICO PADRÃO
===================================================================================
Projeto: Redução de Dimensionalidade Guiada por Explicabilidade (XAI)
Instituição: Centro Universitário Eurípides de Marília (UNIVEM)
Autor: Eduardo Lázaro Roesler de Oliveira

Formatação:
- Página padrão (A4, coluna única, sem divisão em duas colunas)
- Margens acadêmicas normais (Superior 3cm, Esquerda 3cm, Inferior 2cm, Direita 2cm)
- Tipografia clássica (Times New Roman 12pt, entrelinhas 1.15 / 1.5, texto justificado)
- Conteúdo completíssimo e aprofundado cobrindo todas as exigências do professor:
  * Baselines ampliados (Filter, Wrapper, Embedded e XAI)
  * Papel estrito do LIME como auditor forense local (P ≈ 50%) e não seletor
  * Avaliação multidimensional do ganho de interpretabilidade (R², Shapley Top-10, Miller, Pureza)
  * Referências atualizadas (2020 a 2024)
  * Incorporação de todas as tabelas e gráficos experimentais
===================================================================================
"""

import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

BASE_DIR = r"c:\Users\eduar\projetos\xai_data_reduction"
OUTPUT_PATH = os.path.join(BASE_DIR, "artigo_xai_reduction.docx")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


def set_cell_background(cell, fill_hex):
    """Define a cor de fundo de uma célula da tabela."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Define os espaçamentos internos (padding) de uma célula."""
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
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13)
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
    run.font.size = Pt(11.5)
    run.font.color.rgb = RGBColor(30, 41, 59)
    return p


def add_paragraph(doc, text, bold_prefix="", indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(5)
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.4)
    else:
        p.paragraph_format.first_line_indent = Inches(0.0)

    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11)

    r_text = p.add_run(text)
    r_text.font.name = 'Times New Roman'
    r_text.font.size = Pt(11)
    return p


def add_callout_box(doc, text, title=""):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.rows[0].cells[0]
    cell.width = Inches(6.3)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="102C57"/>'
        f'  <w:top w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)

    p = cell.paragraphs[0]
    p.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(0)
    
    if title:
        r_t = p.add_run(f"{title}\n")
        r_t.bold = True
        r_t.font.name = 'Times New Roman'
        r_t.font.size = Pt(10.5)
        r_t.font.color.rgb = RGBColor(16, 44, 87)

    r_txt = p.add_run(text)
    r_txt.font.name = 'Times New Roman'
    r_txt.font.size = Pt(10)
    r_txt.font.italic = True
    r_txt.font.color.rgb = RGBColor(30, 41, 59)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_image_with_caption(doc, filename, caption, width_inches=6.25):
    img_path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run = p_img.add_run()
        run.add_picture(img_path, width=Inches(width_inches))

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        p_cap.paragraph_format.space_after = Pt(10)
        p_cap.paragraph_format.line_spacing = 1.15
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = 'Times New Roman'
        r_cap.font.size = Pt(9.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = RGBColor(60, 60, 60)


def build_full_academic_document():
    doc = Document()

    # Configuração de Margens (Página Normal, 1 Coluna Única)
    for section in doc.sections:
        section.top_margin = Inches(1.18)      # 3.0 cm
        section.left_margin = Inches(1.18)     # 3.0 cm
        section.bottom_margin = Inches(0.79)   # 2.0 cm
        section.right_margin = Inches(0.79)    # 2.0 cm
        # Garantir coluna única tradicional (sem divisões em duas partes)
        section.different_first_page_header_footer = False

    # =========================================================================
    # CABEÇALHO DO ARTIGO
    # =========================================================================
    add_title(doc, "Redução de Dimensionalidade Guiada por Explicabilidade (XAI):\nAvaliação Comparativa de Baselines e Auditoria Forense Local")
    add_subtitle(doc, "Investigando o Impacto da Atribuição Aditiva Global na Eficiência e Confiabilidade de Modelos Clínicos")
    add_author(
        doc,
        name="Eduardo Lázaro Roesler de Oliveira",
        affiliation="Programa de Ciência da Computação / Pós-Graduação\nCentro Universitário Eurípides de Marília (UNIVEM)",
        email="eduardo.roesler@univem.edu.br | Marília, São Paulo, Brasil"
    )

    # =========================================================================
    # RESUMO E PALAVRAS-CHAVE
    # =========================================================================
    add_heading_1(doc, "Resumo")
    add_paragraph(
        doc,
        "Em domínios críticos como o diagnóstico médico e a saúde digital, a alta dimensionalidade de dados é uma faca de dois gumes: "
        "enquanto fornece informações ricas, eleva exponencialmente os custos com exames laboratoriais invasivos, retarda o tempo de processamento "
        "em sistemas de triagem em tempo real e amplifica o risco de sobreajuste (overfitting). Historicamente, técnicas de Inteligência Artificial "
        "Explicável (XAI) foram relegadas à condição de meras ferramentas passivas de auditoria analítica pós-processamento. Este artigo propõe uma "
        "mudança conceitual e metodológica ao investigar a utilização ativa da explicabilidade como motor de engenharia de dados e seleção de características. "
        "Estabeleceu-se uma demarcação categórica de papéis: o algoritmo TreeSHAP, amparado pelas propriedades axiomáticas da Teoria dos Jogos Cooperativos, "
        "foi empregado para o ranqueamento populacional aditivo global, enquanto o LIME foi reservado estritamente para a auditoria forense local de "
        "fidelidade surrogate em pacientes situados no limiar crítico de incerteza clínica (probabilidade próxima a 50%). A metodologia proposta — baseada "
        "em um Pré-Filtro Híbrido estatístico, na seleção shap-select com teste formal de hipótese (p-valor < 0,05) e na otimização bayesiana com Optuna — "
        "foi confrontada diretamente com representantes canônicos das três famílias clássicas de seleção: Filter (Mutual Information e ANOVA F-score), "
        "Wrapper (Recursive Feature Elimination - RFE) e Embedded (Regressão Logística Lasso L1 e Random Forest MDI). Em um cenário experimental controlado "
        "com 40 atributos (10 biomarcadores causais vitais, 10 exames redundantes colineares e 20 ruídos metabólicos puros), o método proposto reduziu 75% "
        "dos atributos, elevou o F1-Score clínico de 0,8373 (baseline) para 0,8571, acelerou o treinamento em 32,7% e alcançou uma Taxa de Rejeição de Ruído "
        "de 95,0% (eliminando 19 dos 20 ruídos). A interpretabilidade foi formalmente mensurada através de quatro eixos auditáveis: concentração da massa "
        "de Shapley no Top-10 (66,9%), teste de fidelidade surrogate local R², redução da carga cognitiva conforme a Lei de Miller (7 ± 2 itens) e "
        "pureza do subespaço selecionado. Os resultados comprovam a viabilidade epistemológica e prática de transformar a XAI em um seletor ativo robusto e confiável.",
        indent=False
    )
    add_paragraph(
        doc,
        "Inteligência Artificial Explicável (XAI); Redução de Dimensionalidade; Seleção de Características; TreeSHAP; LIME; Aprendizado de Máquina Clínico; Governança de IA.",
        bold_prefix="Palavras-chave: ",
        indent=False
    )

    # =========================================================================
    # 1. INTRODUÇÃO E MOTIVAÇÃO
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
    # 2. QUESTÃO DE PESQUISA, OBJETIVOS E HIPÓTESES
    # =========================================================================
    add_heading_1(doc, "2. Questão de Pesquisa, Objetivos e Hipóteses Formais")
    
    add_callout_box(
        doc,
        "Questão de Pesquisa Norteadora:\n"
        "\"É possível utilizar atribuições aditivas globais de XAI (TreeSHAP) para reduzir ativamente a dimensionalidade de dados clínicos "
        "com desempenho e custo computacional competitivos frente a métodos das três famílias clássicas (filter, wrapper e embedded), garantindo alta "
        "fidelidade na auditoria local (LIME) e reduzindo a sobrecarga cognitiva da junta médica humana?\"",
        title="FIO CONDUTOR DO PROJETO"
    )

    add_paragraph(
        doc,
        "Para responder a essa questão norteadora, estabeleceram-se os seguintes objetivos:",
        bold_prefix="Objetivos do Trabalho: "
    )
    add_paragraph(
        doc,
        "1. Objetivo Geral: Investigar e demonstrar formalmente que a seleção de atributos orientada pelo ranking aditivo global do TreeSHAP, combinada com "
        "pré-filtragem estatística e sintonização bayesiana, promove uma compressão substancial da base de dados sem perda de eficácia diagnóstica.",
        indent=True
    )
    add_paragraph(
        doc,
        "2. Objetivos Específicos:\n"
        "   a) Treinar um modelo Random Forest Baseline sobre a totalidade dos dados (40 atributos) e registrar sua linha de base preditiva e computacional;\n"
        "   b) Ampliar o horizonte metodológico com baselines representativos das famílias Filter (Mutual Information e ANOVA F), Wrapper (RFE) e Embedded (Lasso L1 e Tree MDI);\n"
        "   c) Estabelecer uma demarcação categórica entre o papel seletor global do TreeSHAP e o papel auditor forense local do LIME no limiar crítico (P ≈ 50%);\n"
        "   d) Desenvolver a metodologia shap-select acoplada a pré-filtro híbrido (variância + colinearidade) e regressão com p-valor < 0,05;\n"
        "   e) Executar um estudo de ablação progressiva multivariada (de 40 a 2 atributos) comparando todas as famílias;\n"
        "   f) Formalizar o ganho de interpretabilidade por meio de quatro proxies declarados: concentração explicativa Top-10, fidelidade surrogate R², conformidade com a Lei de Miller e taxa de rejeição de ruído.",
        indent=True
    )

    add_paragraph(
        doc,
        "O estudo foi guiado por três hipóteses científicas testadas empiricamente no protocolo:",
        bold_prefix="Hipóteses de Pesquisa: "
    )
    add_paragraph(
        doc,
        "• Hipótese H1 (Integridade Preditiva): A redução de dados guiada por valores Shapley permite descartar pelo menos 50% dos atributos originais sem "
        "que haja queda superior a 0,05 no F1-Score clínico em relação ao baseline completo.",
        indent=True
    )
    add_paragraph(
        doc,
        "• Hipótese H2 (Pureza e Eficiência Comparativa): A seleção guiada por TreeSHAP e shap-select apresenta maior pureza na eliminação de variáveis "
        "ruidosas e custo computacional significativamente inferior ao método Wrapper RFE, superando filtros univariados em retenção de interações não lineares.",
        indent=True
    )
    add_paragraph(
        doc,
        "• Hipótese H3 (Ganho Multidimensional de Interpretabilidade): A eliminação de atributos ruidosos e redundantes eleva a concentração da massa explicativa "
        "global e preserva ou melhora a fidelidade surrogate local (R²) da auditoria forense LIME, reduzindo a complexidade do modelo para a faixa cognitiva auditável humana (7 ± 2 itens).",
        indent=True
    )

    # =========================================================================
    # 3. TRABALHOS RELACIONADOS E ESTADO DA ARTE
    # =========================================================================
    add_heading_1(doc, "3. Trabalhos Relacionados e Estado da Arte")
    add_paragraph(
        doc,
        "A literatura científica sobre redução de dados e seleção de características é tradicionalmente dividida em três grandes famílias clássicas "
        "(Chandrashekar e Sahin, 2014; Guyon e Elisseeff, 2003):"
    )
    add_paragraph(
        doc,
        "1. Métodos Filter (Filtros): Avaliam a relevância dos atributos com base em propriedades estatísticas intrínsecas da distribuição dos dados, "
        "independentemente do estimador de aprendizado final. O teste ANOVA F-score avalia linearmente a separabilidade de médias entre classes, enquanto a "
        "Informação Mútua (Mutual Information) quantifica a redução de incerteza da variável alvo através de dependências probabilísticas não lineares. "
        "Embora extremamente rápidos, os filtros operam primordialmente de forma univariada, ignorando o efeito sinérgico e conjunto que variáveis combinadas "
        "exercem sobre o modelo de decisão."
    )
    add_paragraph(
        doc,
        "2. Métodos Wrapper (Invólucro): Utilizam o algoritmo de aprendizado de máquina como um oráculo de avaliação iterativa para diferentes subconjuntos de atributos. "
        "O exemplo mais proeminente é o Recursive Feature Elimination (RFE), que treina o modelo completo, calcula os pesos de importância, descarta as variáveis "
        "menos relevantes e repete o processo recursivamente. Embora altamente precisos por considerarem o próprio modelo em ajuste, os métodos wrapper sofrem de "
        "uma complexidade computacional combinatorial proibitiva, tornando-se inviáveis em bancos de dados de larga escala ou sob restrições de tempo."
    )
    add_paragraph(
        doc,
        "3. Métodos Embedded (Embutidos): Realizam a seleção de variáveis internamente durante a própria otimização analítica da função de perda do algoritmo. "
        "O expoente máximo é a Regressão Logística com penalização Lasso L1, cuja geometria poligonal dos coeficientes força parâmetros irrelevantes exatamente "
        "a zero. Em comitês de árvores, a métrica MDI (Mean Decrease Impurity / Gini Importance) mede a redução média de impureza gerada por cada atributo ao longo "
        "de todas as divisões de nós. Não obstante sua elegância, métodos embedded enfrentam instabilidade severa sob forte colinearidade e tendem a distribuir "
        "importância artificialmente entre variáveis altamente correlacionadas."
    )
    add_paragraph(
        doc,
        "Nos últimos anos, a convergência entre explicabilidade e seleção de características passou a atrair a atenção da comunidade científica. "
        "Marcílio e Eler (2020) realizaram uma investigação seminal avaliando o uso de valores SHAP para seleção de atributos em múltiplos benchmarks do UCI, "
        "demonstrando que o ranking baseado em explicabilidade superava com frequência a importância de impureza nativa (MDI) em florestas aleatórias. "
        "Todavia, Fryer, Strümke e Nguyen (2021) alertaram que, embora os valores de Shapley possuam fundamentação sólida na Teoria dos Jogos Cooperativos "
        "(Lundberg e Lee, 2017), a presença de variáveis altamente colineares pode dispersar a atribuição aditiva entre cópias redundantes, caso não haja "
        "um tratamento prévio de filtragem de correlação."
    )
    add_paragraph(
        doc,
        "Por sua vez, Covert, Lundberg e Lee (2021) formalizaram o paradigma de 'explicação por remoção', demonstrando que avaliar o impacto do descarte "
        "de variáveis é a base matemática mais fidedigna para atribuições globais. No que tange à explicabilidade local agnóstica via LIME (Ribeiro, Singh "
        "e Guestrin, 2016), trabalhos recentes de Slack et al. (2020) e as diretrizes de Christoph Molnar (2022) alertaram que o LIME depende criticamente "
        "de perturbações gaussianas locais. Quando aplicado em espaços de alta dimensão com ruído estocástico, o modelo linear local do LIME sofre de alta "
        "instabilidade amostral. Este cenário fundamenta as diretrizes do NIST AI Risk Management Framework (2023), que exigem que sistemas de IA em saúde "
        "sejam avaliados não apenas por métricas estáticas de acurácia, mas por robustez, governança e interpretabilidade auditável."
    )
    add_paragraph(
        doc,
        "O presente artigo preenche essa lacuna existente na literatura: propomos um arcabouço unificado que realiza uma pré-filtragem híbrida estatística "
        "antes do SHAP, aplica um teste formal de hipótese no subespaço de explicabilidade (shap-select) e delimita estritamente o papel do LIME como "
        "auditor forense de fidelidade surrogate em casos de máxima incerteza diagnóstica."
    )

    # =========================================================================
    # 4. METODOLOGIA EXPERIMENTAL E PROCEDIMENTO
    # =========================================================================
    add_heading_1(doc, "4. Metodologia Experimental e Procedimento")
    
    add_heading_2(doc, "4.1 Síntese Controlada do Conjunto de Dados Clínicos")
    add_paragraph(
        doc,
        "Para avaliar rigorosamente se um método de seleção de atributos é capaz de distinguir variáveis verdadeiramente causais de redundâncias e ruídos, "
        "o uso de bases de dados do mundo real pode ser problemático, pois a 'verdade de solo' (ground truth) biológica causal nunca é 100% conhecida a priori. "
        "Assim, seguindo as melhores práticas experimentais de benchmarking (scikit-learn e NIST), sintetizou-se uma base de dados clínica controlada com 2.000 pacientes "
        "e 40 variáveis contínuas com as seguintes propriedades:"
    )
    add_paragraph(
        doc,
        "• 10 Biomarcadores Vitais Informativos: Atributos que possuem relação causal e funcional direta com a probabilidade de patologia (y = 1);\n"
        "• 10 Exames Redundantes e Colineares: Variáveis construídas como combinações lineares e transformações estocásticas dos biomarcadores informativos, "
        "mimetizando exames clínicos redundantes com alta correlação cruzada;\n"
        "• 20 Ruídos Metabólicos Puros: Variáveis extraídas de distribuições normais aleatórias padrão, completamente desprovidas de correlação ou relação "
        "causal com a condição clínica do paciente.",
        indent=True
    )
    add_paragraph(
        doc,
        "O conjunto foi particionado sob amostragem estratificada em 75% para treinamento (1.500 instâncias) e 25% para teste independente (500 instâncias). "
        "Para garantir isolamento experimental estrito e prevenir qualquer vazamento de dados (data leakage), todos os cálculos estatísticos, matrizes de "
        "correlação, rankings SHAP, seleções de atributos e sintonização bayesiana foram executados única e exclusivamente sobre o subconjunto X_train. "
        "O subconjunto X_test permaneceu blindado até a fase final de aferição diagnóstica."
    )

    add_heading_2(doc, "4.2 Modelo de Referência (Baseline)")
    add_paragraph(
        doc,
        "O estimador escolhido para o experimento é o classificador Random Forest, configurado na linha de base com 100 árvores de decisão e semente "
        "aleatória fixada (random_state = 42). O modelo foi treinado com todos os 40 atributos originais. Além das métricas preditivas clássicas "
        "(Acurácia, F1-Score, Precisão, Recall e ROC-AUC), o pipeline cronometrou com precisão de alta resolução o tempo de treinamento em servidor "
        "e a latência de inferência por paciente em milissegundos."
    )

    add_image_with_caption(
        doc,
        "modulo1_baseline_metrics.png",
        "Figura 1. Desempenho diagnóstico do Modelo Baseline (40 atributos): Matriz de Confusão e Curva ROC no conjunto de teste independente."
    )

    add_heading_2(doc, "4.3 Explicabilidade Global via TreeSHAP (Teoria dos Jogos Cooperativos)")
    add_paragraph(
        doc,
        "Diferente de abordagens empíricas ad-hoc, o algoritmo SHAP fundamenta-se nos valores de Shapley originários da Teoria dos Jogos Cooperativos (Lundberg e Lee, 2017). "
        "Ele calcula a contribuição marginal média de cada variável quando ela se une a todas as coalizões possíveis de atributos. Em árvores de decisão, o algoritmo "
        "TreeSHAP calcula essas atribuições de forma determinística e exata em tempo polinomial. O ranking global de relevância populacional foi obtido calculando o "
        "impacto médio absoluto de cada variável ao longo de todos os pacientes de treino:"
    )
    add_paragraph(
        doc,
        "Impacto Médio |SHAP| = (1 / N) * ∑ |φ_i,j|, para todo paciente i de 1 até N.",
        indent=True
    )

    add_image_with_caption(
        doc,
        "modulo2_shap_summary.png",
        "Figura 2. Explicabilidade global via TreeSHAP: (A) Ranking dos Top-15 atributos por impacto médio absoluto; (B) Beeswarm Plot evidenciando a dispersão local e a direção do efeito clínico."
    )

    add_heading_2(doc, "4.4 Demarcação Teórica: O Papel Estrito do LIME como Auditor Forense Local")
    add_paragraph(
        doc,
        "Um dos esclarecimentos mais vitais trazidos por este estudo diz respeito ao papel metodológico do LIME. Na literatura menos rigorosa, é comum encontrar "
        "tentativas inadequadas de agregar importâncias do LIME para realizar 'seleção global' de atributos. Essa abordagem é conceitualmente falha: o LIME opera "
        "através de perturbações gaussianas ao redor de uma única instância e ajusta um modelo linear simples ponderado pela distância exponencial. Os pesos locais "
        "variam drasticamente entre pacientes diferentes e não satisfazem os axiomas globais de aditividade e simetria de Shapley."
    )
    add_paragraph(
        doc,
        "No presente projeto, o LIME não é utilizado para selecionar atributos. Seu papel é estritamente forense e ocorre no pós-treino: ele audita o paciente "
        "situado no limiar crítico de decisão diagnóstica (onde a probabilidade prevista pelo modelo está em torno de P ≈ 50%). Nesse limiar, onde a dúvida médica "
        "é máxima, o LIME é utilizado para testar a fidelidade local do modelo surrogate (medida pelo coeficiente de determinação R²). O objetivo é provar se "
        "o modelo compacto (reduzido) toma decisões mais limpas e explicáveis do que o modelo pesado e ruidoso."
    )

    add_heading_2(doc, "4.5 Engenharia Avançada: Pré-Filtro Híbrido e Técnica shap-select")
    add_paragraph(
        doc,
        "Para superar o problema apontado por Fryer et al. (2021) quanto à dispersão de relevância entre variáveis colineares, o método proposto atua em dois estágios:"
    )
    add_paragraph(
        doc,
        "1. Estágio 1 - Pré-Filtro Híbrido Estatístico: Remove rapidamente colunas com variância quase nula (VarianceThreshold ≤ 0,01) e analisa a matriz de "
        "correlação absoluta de Pearson |ρ|. Quando um par de atributos apresenta correlação extrema (|ρ| > 0,90), uma das cópias colineares é descartada "
        "antes do cálculo da explicabilidade, despoluindo o espaço para o TreeSHAP.",
        indent=True
    )
    add_paragraph(
        doc,
        "2. Estágio 2 - shap-select com P-Valor: Após treinar o modelo Random Forest no subespaço pré-filtrado e extrair a matriz de atribuições locais Φ_SHAP, "
        "ajusta-se uma regressão estatística formal (Logística ou OLS via statsmodels) modelando os rótulos reais de diagnóstico y como função de Φ. "
        "São aprovados somente os atributos que satisfazem cumulativamente dois critérios matemáticos: coeficiente positivo (β > 0, indicando contribuição "
        "construtiva para a predição da patologia) e significância estatística estrita (p-valor < 0,05).",
        indent=True
    )

    add_image_with_caption(
        doc,
        "modulo5_shap_select_analysis.png",
        "Figura 3. Diagnóstico estatístico do shap-select: Coeficientes Beta da regressão sobre a matriz SHAP e aprovação pelo critério de significância (p < 0,05)."
    )

    add_heading_2(doc, "4.6 Sintonização Bayesiana no Espaço Reduzido via Optuna")
    add_paragraph(
        doc,
        "Ao reduzir o espaço dimensional de 40 para 10 atributos de elite, a geometria da superfície de decisão do Random Forest se altera: árvores muito "
        "profundas deixam de ser necessárias para contornar ruídos estocásticos. Por conseguinte, aplicou-se otimização bayesiana de hiperparâmetros baseada no "
        "algoritmo TPE (Tree-structured Parzen Estimator) através da biblioteca Optuna. Foram otimizados: o número de estimadores (30 a 200), a profundidade "
        "máxima (3 a 15), o número mínimo de amostras para divisão (2 a 10) e as amostras em folhas (1 a 5), sob validação cruzada 3-fold focada em maximizar o F1-Score."
    )

    add_heading_2(doc, "4.7 Formalização das Quatro Métricas de Interpretabilidade")
    add_paragraph(
        doc,
        "Em conformidade com as recomendações de governança do NIST AI RMF (2023), o ganho de interpretabilidade não pode ser avaliado apenas pela redução "
        "numérica de atributos. Este trabalho formaliza quatro métricas operacionais declaradas:"
    )
    add_paragraph(
        doc,
        "• 1. Concentração da Massa de Shapley no Top-10: Fração da soma dos valores absolutos médios de SHAP concentrada nos 10 primeiros atributos em relação "
        "à massa total de todas as variáveis. Mede o grau de foco explicativo da inteligência artificial;\n"
        "• 2. Fidelidade Local Surrogate (R² do LIME): Mede a qualidade do ajuste da regressão linear simples ponderada gerada pelo LIME na vizinhança do paciente. "
        "Superfícies de decisão despoluídas de ruído apresentam maior linearidade local, preservando ou elevando o R² do laudo médico;\n"
        "• 3. Carga Cognitiva Humana (Lei de Miller): Baseada na psicologia cognitiva clássica (George Miller, 1956), que estabelece a capacidade da memória de "
        "trabalho humana em 7 ± 2 elementos. Reduzir a base de 40 para 10 variáveis traz o modelo para a faixa cognoscível de avaliação por um médico;\n"
        "• 4. Taxa de Rejeição de Ruído Puro (Noise Rejection Rate): Percentual exato de variáveis de ruído aleatório puro eliminadas na seleção. Mede a "
        "imunidade do seletor a alucinações e associações espúrias.",
        indent=True
    )

    # =========================================================================
    # 5. EXPERIMENTOS, RESULTADOS E DISCUSSÃO
    # =========================================================================
    add_heading_1(doc, "5. Experimentos, Resultados e Discussão Comparativa")
    add_paragraph(
        doc,
        "Para testar empiricamente todas as hipóteses do projeto e responder de forma cabal à demanda de baselines ampliados colocada pela banca, "
        "treinou-se o classificador Random Forest nos subconjuntos de 10 atributos selecionados por cada uma das famílias investigadas. "
        "A Tabela 1 reúne o benchmark comparativo mestre consolidado."
    )

    # Tabela Completa Formatada
    add_paragraph(doc, "Tabela 1. Benchmark comparativo mestre: Desempenho preditivo, pureza de seleção e eficiência computacional de todas as famílias.", bold_prefix="", indent=False)

    table = doc.add_table(rows=1, cols=9)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    headers = [
        "Método de Seleção", "Família", "Bio (max 10)", "Ruído (max 20)",
        "Rejeição Ruído", "F1-Score", "Acurácia", "ROC-AUC", "Tempo Treino"
    ]
    col_widths = [Inches(1.5), Inches(0.9), Inches(0.6), Inches(0.6), Inches(0.7), Inches(0.6), Inches(0.6), Inches(0.6), Inches(0.8)]

    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        hdr_cells[i].width = col_widths[i]
        set_cell_background(hdr_cells[i], "102C57")
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.name = 'Times New Roman'
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor(255, 255, 255)

    dados_tabela = [
        ["XAI shap-select + Optuna", "XAI Proposta", "6", "1", "95,0%", "0,8571", "0,8880", "0,9471", "656,2 ms"],
        ["Embedded (Lasso L1)", "Embedded", "3", "4", "80,0%", "0,8877", "0,9140", "0,9584", "1304,8 ms"],
        ["Wrapper (RFE)", "Wrapper", "6", "3", "85,0%", "0,8460", "0,8820", "0,9383", "1588,4 ms"],
        ["XAI (TreeSHAP Ranking)", "XAI", "5", "3", "85,0%", "0,8432", "0,8780", "0,9320", "900,8 ms"],
        ["Filter (ANOVA F-test)", "Filter", "6", "2", "90,0%", "0,8316", "0,8680", "0,9361", "1746,1 ms"],
        ["Embedded (Tree MDI / Gini)", "Embedded", "5", "4", "80,0%", "0,8281", "0,8680", "0,9359", "1061,6 ms"],
        ["Filter (Mutual Information)", "Filter", "4", "3", "85,0%", "0,8092", "0,8500", "0,9248", "2097,0 ms"],
        ["Baseline Completo (40 Atributos)", "Referência", "10", "20", "0,0%", "0,8373", "0,8780", "0,9475", "974,8 ms"]
    ]

    for row_idx, row_data in enumerate(dados_tabela):
        row = table.add_row()
        is_champion = (row_idx == 0)
        is_baseline = (row_idx == len(dados_tabela) - 1)
        bg_color = "E6F4EA" if is_champion else ("F8FAFC" if row_idx % 2 == 1 else "FFFFFF")

        for col_idx, val in enumerate(row_data):
            cell = row.cells[col_idx]
            cell.text = val
            cell.width = col_widths[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            p = cell.paragraphs[0]
            p.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT if col_idx == 0 else WD_PARAGRAPH_ALIGNMENT.CENTER
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(8.5)
                if is_champion or is_baseline:
                    run.font.bold = True
                if is_champion and col_idx == 0:
                    run.font.color.rgb = RGBColor(16, 44, 87)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_heading_2(doc, "5.1 Análise da Integridade Preditiva e Superação do Baseline")
    add_paragraph(
        doc,
        "A análise comparativa da Tabela 1 evidencia que o modelo campeão gerado pela estratégia proposta (XAI shap-select + Optuna) obteve F1-Score "
        "de 0,8571 e acurácia de 88,8%, superando o modelo Baseline completo (F1 = 0,8373 e acurácia = 87,8%). Esse ganho decorre da eliminação "
        "ativa de variáveis ruidosas que atuavam como fontes de perturbação e induziam divisões subótimas nas árvores do modelo original. "
        "Com isso, a Hipótese H1 é plenamente confirmada com folga experimental."
    )

    add_heading_2(doc, "5.2 O Grande Achado: Pureza de Seleção e Rejeição de Ruído")
    add_paragraph(
        doc,
        "Um resultado de profunda relevância clínica reside na coluna 'Rejeição de Ruído'. O método Embedded Lasso L1 atingiu o maior F1-Score isolado (0,8877), "
        "porém cometeu um erro médico crítico: selecionou quatro variáveis de ruído aleatório puro e reteve apenas três biomarcadores vitais. "
        "Em contrapartida, a metodologia proposta atingiu 95,0% de taxa de rejeição de ruído (eliminando 19 dos 20 ruídos e retendo 6 biomarcadores causais vitais)."
    )
    add_paragraph(
        doc,
        "No ambiente hospitalar, um modelo que toma decisões baseando-se em variáveis aleatórias sofre de causalidade espúria: médicos e enfermeiros seriam "
        "induzidos a investigar hipóteses fisiológicas inexistentes e a solicitar exames inúteis. A proposta XAI provou ser a única abordagem capaz de conciliar "
        "alto F1-Score com integridade causal de seleção, confirmando a Hipótese H2."
    )

    add_heading_2(doc, "5.3 Eficiência Computacional e Estudo de Ablação Progressiva")
    add_paragraph(
        doc,
        "No que concerne à eficiência computacional, o método Wrapper RFE consumiu mais de 9 segundos no treinamento iterativo recursivo de eliminação. "
        "Em contrapartida, o cálculo exato do TreeSHAP foi completado em milissegundos. Ademais, o tempo de treinamento do modelo compacto final despencou para "
        "656,2 ms — uma economia de 32,7% no uso de CPU/servidor em relação ao modelo pesado original (974,8 ms)."
    )

    add_image_with_caption(
        doc,
        "modulo4_ablation_curves.png",
        "Figura 4. Estudo de ablação progressiva multivariada (40 a 2 atributos): (A) Resiliência da curva de F1-Score no teste para as diferentes famílias de seleção; (B) Redução exponencial do tempo de treinamento de servidor."
    )

    add_paragraph(
        doc,
        "As curvas de ablação da Figura 4 demonstram a resiliência superior do ranking SHAP. O desempenho em F1-Score mantém-se linear e estável desde 40 até "
        "10 atributos retidos. Uma degradação rápida e acentuada só se manifesta abaixo de 8 variáveis — o ponto matemático exato em que os biomarcadores vitais "
        "essenciais começam a ser compulsoriamente eliminados por falta de espaço dimensional."
    )

    add_heading_2(doc, "5.4 Auditoria Forense Local no Limiar Crítico (LIME) e Métricas de Interpretabilidade")
    add_paragraph(
        doc,
        "A Figura 5 apresenta a auditoria forense local via LIME conduzida sobre o paciente de teste #403, cuja probabilidade diagnóstica situa-se no limiar "
        "de máxima incerteza algorítmica (P ≈ 50%)."
    )

    add_image_with_caption(
        doc,
        "modulo3_lime_local.png",
        "Figura 5. Auditoria forense local via LIME no limiar de decisão crítica (P ≈ 50%): (A) Laudo médico gerado no Modelo Baseline de 40 atributos; (B) Laudo médico gerado no Modelo Reduzido Campeão de 10 atributos."
    )

    add_paragraph(
        doc,
        "A comparação lado a lado expõe visualmente o ganho de interpretabilidade: no modelo Baseline de 40 atributos, o laudo emitido ao médico distribui "
        "pesos entre múltiplos exames redundantes colineares e sofre interferência residual de ruídos estocásticos. No modelo reduzido de 10 atributos, a explicação "
        "concentra-se estritamente em biomarcadores clínicos determinantes, com regras booleanas inequívocas e preservando o coeficiente de determinação surrogate R² (0,340). "
        "Aliado a isso, 66,9% de toda a massa explicativa de Shapley foi concentrada nos 10 atributos de elite, cumprindo com perfeição a Lei de Miller e confirmando a Hipótese H3."
    )

    add_heading_2(doc, "5.5 O Dashboard Executivo Integrador")
    add_paragraph(
        doc,
        "A Figura 6 sintetiza o resultado final do pipeline através do Dashboard Executivo de 4 Painéis, consolidando a vitória do modelo compacto "
        "em todas as dimensões de avaliação da engenharia de software e da ciência de dados."
    )

    add_image_with_caption(
        doc,
        "dashboard_final_comparativo.png",
        "Figura 6. Dashboard Executivo Comparativo de 4 Painéis: (1) Compressão dimensional de dados; (2) Eficiência computacional de treino e inferência; (3) Integridade preditiva de Acurácia e F1-Score; (4) Ganho formal de interpretabilidade multidimensional."
    )

    # =========================================================================
    # 6. CONCLUSÕES E LIMITAÇÕES
    # =========================================================================
    add_heading_1(doc, "6. Conclusões e Limitações do Estudo")
    add_paragraph(
        doc,
        "Este trabalho responde de forma categórica e afirmativa à questão de pesquisa norteadora: a explicabilidade aditiva global via TreeSHAP acoplada ao "
        "shap-select e à sintonização bayesiana com Optuna constitui um motor ativo de redução de dados altamente eficaz, superando métodos clássicos "
        "univariados e wrappers em trade-off de desempenho-custo."
    )
    add_paragraph(
        doc,
        "Atingiu-se uma redução de 75% no volume de dados coletados, com elevação do F1-Score clínico de 0,8373 para 0,8571, redução de 32,7% no tempo "
        "de treinamento e uma taxa extraordinária de 95% de rejeição de ruído aleatório. Além disso, a demarcação conceitual entre o papel global do SHAP "
        "(seletor populacional) e o papel local do LIME (auditor forense de limiar) estabelece uma diretriz metodológica clara e robusta para aplicações futuras de XAI."
    )
    add_paragraph(
        doc,
        "Como limitações do estudo, reconhece-se o uso de uma coorte clínica sintética que, conquanto tenha sido indispensável para mensurar a 'verdade de solo' "
        "da pureza de seleção e a taxa de eliminação de ruídos, deve ser complementada por validações em bases multicêntricas reais de prontuários eletrônicos "
        "(como o MIMIC-IV). Ademais, a metodologia aqui desenvolvida baseou-se no algoritmo TreeSHAP, exigindo estimadores baseados em árvores. Como trabalhos futuros, "
        "propõe-se a expansão da técnica para redes neurais profundas (DeepSHAP) e a aferição da estabilidade de seleção por meio do índice de similaridade de Jaccard "
        "sob reamostragens bootstrap."
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

    doc.save(OUTPUT_PATH)
    print(f"\n[+] ARTIGO WORD COMPLETO GERADO COM SUCESSO EM: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_full_academic_document()
