from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

OUTPUT_PATH = r"c:\Users\eduar\projetos\xai_data_reduction\artigo_xai_reduction.docx"

def add_heading(doc, text, level=1):
    doc.add_heading(text, level=level)


def add_paragraph(doc, text, justify=True, bold=False, size=11):
    p = doc.add_paragraph()
    if justify:
        p.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = 'Calibri'


def build_document():
    doc = Document()

    p = doc.add_paragraph()
    p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = p.add_run('XAI aplicada à redução de dados em Machine Learning')
    run.bold = True
    run.font.size = Pt(18)
    run.font.name = 'Calibri'

    doc.add_paragraph('')

    add_heading(doc, '1. Introdução', 1)
    add_paragraph(doc, 'O avanço recente em Machine Learning tem levado ao desenvolvimento de modelos cada vez mais sofisticados e com alto poder preditivo. No entanto, esse aumento de performance muitas vezes é acompanhado por maior complexidade, o que dificulta a interpretação das decisões tomadas. Esse problema é especialmente relevante em áreas sensíveis, como saúde, finanças, educação e segurança, em que a explicabilidade das previsões é tão importante quanto a acurácia.')
    add_paragraph(doc, 'Nesse cenário, a área de XAI (Explainable Artificial Intelligence) surge como resposta para tornar os modelos mais transparentes e compreensíveis para seres humanos. O objetivo de XAI não é apenas responder “qual foi a previsão?”, mas também “por que a previsão foi feita dessa forma?”. Isso permite que especialistas entendam melhor o comportamento dos modelos, identifiquem falhas e interpretem decisões em contextos reais.')
    add_paragraph(doc, 'No entanto, a explicabilidade raramente é usada apenas como uma ferramenta final de análise. Há uma possibilidade muito interessante de usá-la também como parte do próprio processo de construção do modelo. Em particular, técnicas de explicabilidade podem ser empregadas para identificar quais variáveis são mais relevantes para a tarefa e, a partir disso, reduzir a dimensionalidade dos dados. Essa estratégia é especialmente útil em problemas com grande número de atributos, em que nem todas as variáveis têm igual contribuição para a predição.')
    add_paragraph(doc, 'A redução de dados é uma preocupação central em Machine Learning, pois dados de alta dimensionalidade podem gerar custo computacional mais elevado, maior risco de overfitting e menor interpretabilidade. Assim, a pergunta central desta pesquisa é: é possível usar explicações do modelo para reduzir o conjunto de atributos sem comprometer substancialmente o desempenho preditivo?')
    add_paragraph(doc, 'Este estudo busca responder essa questão por meio de um pipeline experimental em que um modelo baseline é treinado, suas decisões são explicadas com técnicas de XAI, e a informação de importância das variáveis é usada para reduzir o conjunto de dados. Em seguida, o modelo é re-treinado e seus resultados são comparados ao baseline original.')

    add_heading(doc, '2. Contextualização do problema', 1)
    add_paragraph(doc, 'A redução de dimensionalidade é um problema clássico em Machine Learning. Em muitos datasets, especialmente em áreas como saúde ou biomedicina, é comum encontrar um grande número de variáveis, algumas das quais podem ser redundantes, pouco informativas ou até mesmo ruidosas. Esse fenômeno é especialmente importante porque, quando se aumenta o número de atributos, o modelo pode se tornar mais caro computacionalmente, menos robusto e menos interpretável.')
    add_paragraph(doc, 'A solução mais comum para esse problema é usar técnicas de seleção de atributos ou redução de dimensionalidade antes do treinamento do modelo. Métodos tradicionais como Recursive Feature Elimination (RFE), SelectKBest e feature importance são amplamente utilizados. Esses métodos são úteis, mas nem sempre oferecem uma explicação clara sobre a contribuição individual de cada variável para a previsão final.')
    add_paragraph(doc, 'A XAI aparece como uma alternativa que oferece uma leitura mais interpretável do impacto de cada atributo no comportamento do modelo. Métodos como SHAP e LIME são frequentemente usados para identificar quais variáveis têm maior influência sobre uma decisão. A partir disso, é possível construir um processo de seleção em que o modelo “explica” o que mais importa e essa informação é usada para orientar a redução dos dados.')
    add_paragraph(doc, 'Essa mudança de perspectiva é importante: em vez de usar XAI apenas como explicação posterior, o estudo propõe usá-la como mecanismo de seleção de atributos. Assim, a explicabilidade deixa de ser apenas um componente de análise e passa a ser parte ativa do pipeline de processamento e aprendizado.')

    add_heading(doc, '3. Problema de pesquisa', 1)
    add_paragraph(doc, 'O problema central desta pesquisa pode ser formulado da seguinte maneira: é possível utilizar técnicas de explicabilidade para reduzir o número de atributos de um conjunto de dados sem degradar significativamente o desempenho do modelo preditivo?')
    add_paragraph(doc, 'Essa pergunta é relevante porque conecta duas áreas que, em muitos projetos, são tratadas separadamente: Machine Learning, focado em performance, e XAI, focado em interpretabilidade. O objetivo do estudo é justamente unir esses dois aspectos em um único fluxo de pesquisa, em que a explicabilidade não serve apenas para justificar o modelo, mas também para melhorar sua representação interna.')

    add_heading(doc, '4. Objetivos', 1)
    add_paragraph(doc, 'Objetivo geral: Investigar se técnicas de explicabilidade, como SHAP e LIME, podem ser usadas para reduzir a dimensionalidade de um conjunto de dados sem comprometer significativamente o desempenho preditivo do modelo.')
    add_paragraph(doc, 'Objetivos específicos: (i) treinar um modelo baseline com todas as variáveis; (ii) identificar atributos relevantes por meio de métodos de XAI; (iii) reduzir o conjunto de dados com base nessas explicações; (iv) treinar novamente o modelo no subconjunto reduzido; (v) comparar o desempenho do modelo reduzido com o baseline completo; (vi) comparar a abordagem guiada por XAI com uma técnica tradicional de seleção, como RFE; (vii) avaliar o impacto da redução na eficiência computacional e no tempo de treinamento.')

    add_heading(doc, '5. Hipóteses', 1)
    add_paragraph(doc, 'H1: A utilização de técnicas de explicabilidade para orientar a redução de atributos permite diminuir a dimensionalidade do conjunto de dados sem prejudicar de forma significativa o desempenho do modelo.')
    add_paragraph(doc, 'H2: A seleção de atributos guiada por XAI pode produzir desempenho comparável ou superior a métodos tradicionais de seleção, como RFE, além de fornecer maior interpretabilidade ao processo.')

    add_heading(doc, '6. Justificativa', 1)
    add_paragraph(doc, 'A justificativa da pesquisa está na necessidade de combinar performance e interpretabilidade em um contexto de dados de alta dimensionalidade. Em muitos problemas reais, o maior desafio não é apenas classificar corretamente, mas também entender o que está sendo usado para decidir. Isso é especialmente importante em áreas em que a decisão final precisa ser explicável para especialistas humanos.')
    add_paragraph(doc, 'Além disso, a redução de atributos pode trazer benefícios práticos relevantes, como menor custo computacional, maior agilidade no treinamento, menor uso de memória, menor tempo de inferência, modelos mais simples e maior interpretabilidade. Dessa forma, a pesquisa tem relevância tanto para a área acadêmica quanto para aplicações práticas. Ela investiga se a explicabilidade pode ser usada como mecanismo de seleção e otimização do próprio pipeline de dados.')

    add_heading(doc, '7. Fundamentação teórica', 1)
    add_paragraph(doc, '7.1. Machine Learning e alta dimensionalidade: dados de alta dimensionalidade são comuns em aplicações de diagnóstico, sensores, dados genéticos, dados clínicos e sistemas de monitoramento. Nesses cenários, o número de atributos pode ser grande demais para a quantidade de observações disponível, o que aumenta a chance de overfitting e reduz a robustez do modelo.')
    add_paragraph(doc, '7.2. Seleção de atributos: a seleção de atributos é uma das formas mais comuns de lidar com esse problema. O objetivo é escolher um subconjunto de variáveis que seja suficiente para representar o problema, reduzindo ruído e redundância. Métodos tradicionais incluem SelectKBest, Recursive Feature Elimination (RFE), feature importance e filtros estatísticos. Esses métodos são úteis, mas nem sempre dão uma leitura clara do impacto de cada variável na decisão do modelo.')
    add_paragraph(doc, '7.3. SHAP: SHAP (Shapley Additive Explanations) é uma técnica de explicabilidade baseada na teoria dos jogos cooperativos. Ela atribui valores de importância a cada atributo com base na contribuição marginal da variável para a previsão final do modelo. A principal vantagem do SHAP é que ele oferece uma explicação global e local, permitindo identificar quais atributos mais influenciam a saída do modelo e em que intensidade.')
    add_paragraph(doc, '7.4. LIME: LIME (Local Interpretable Model-agnostic Explanations) é outra técnica de explicabilidade. Diferentemente do SHAP, o LIME busca explicar uma previsão específica por meio de perturbações locais do dado. Isso torna a explicação mais interpretável em casos de decisões individuais.')
    add_paragraph(doc, '7.5. XAI como auxílio na redução de dados: a grande proposta desta pesquisa é usar esses métodos não apenas para interpretar o modelo final, mas também para orientar a redução dos dados. Em vez de considerar a explicabilidade como um fim em si mesma, a ideia é usá-la como ferramenta de seleção e otimização.')

    add_heading(doc, '8. Metodologia', 1)
    add_paragraph(doc, 'A metodologia foi organizada em etapas, seguindo uma lógica de estudo experimental.')
    add_paragraph(doc, '8.1. Geração do conjunto sintético: o primeiro passo foi gerar um conjunto de dados sintéticos com características semelhantes às de um problema de saúde ou diagnóstico, incluindo atributos informativos, redundantes, ruidosos, desbalanceamento leve e estrutura controlada para permitir análise comparativa.')
    add_paragraph(doc, '8.2. Treinamento do modelo baseline: em seguida, um modelo de referência foi treinado com todas as variáveis. Esse modelo serviu como linha de base para avaliar todas as posteriores reduções.')
    add_paragraph(doc, '8.3. Métricas de avaliação: as métricas usadas foram acurácia, precisão, recall, F1-score, ROC-AUC, tempo de treinamento, tempo de inferência e número de atributos. A combinação dessas métricas permite avaliar não só a acurácia do modelo, mas também o custo computacional da solução.')
    add_paragraph(doc, '8.4. Explicabilidade global e local: depois do treinamento, o modelo foi explicado por SHAP e LIME. O SHAP foi usado para gerar um ranking global de importância dos atributos, e o LIME foi usado para analisar casos específicos, especialmente instâncias próximas do limiar de decisão.')
    add_paragraph(doc, '8.5. Seleção de atributos guiada por XAI: com base na importância atribuída pelos métodos de explicabilidade, foi feita a escolha de um subconjunto de atributos. Esse processo permitiu reduzir a dimensionalidade do problema sem remover, a priori, o que parecia mais relevante.')
    add_paragraph(doc, '8.6. Comparação com RFE: a abordagem guiada por XAI foi comparada com uma estratégia tradicional de seleção de atributos: Recursive Feature Elimination (RFE). Essa comparação é importante porque permite verificar se a seleção baseada em explicabilidade oferece vantagem numérica ou interpretativa em relação a uma técnica clássica.')
    add_paragraph(doc, '8.7. Filtro híbrido e shap-select: além da seleção direta por SHAP, a metodologia inclui um estágio adicional de pré-filtro híbrido, com remoção de variáveis de baixa variância e alta correlação, e uma etapa de shap-select, em que os coeficientes de regressão sobre valores SHAP são usados para decidir quais atributos devem permanecer no modelo.')
    add_paragraph(doc, '8.8. Otimização dos hiperparâmetros: após a redução dos atributos, o modelo foi re-otimizado por hyperparameter search, utilizando Optuna. Isso permite avaliar se a etapa final do pipeline pode ser ajustada de maneira mais eficiente em um espaço menor de atributos.')
    add_paragraph(doc, '8.9. Dashboard comparativo final: por fim, os resultados foram reunidos em um dashboard comparativo para mostrar a diferença entre o modelo baseline completo e o modelo reduzido por XAI, em termos de número de atributos, desempenho e custo computacional.')

    add_heading(doc, '9. Resultados esperados', 1)
    add_paragraph(doc, 'A expectativa desta pesquisa é que a redução guiada por XAI promova um equilíbrio interessante entre desempenho e simplicidade. Em outras palavras, espera-se que um modelo treinado com um conjunto menor de atributos mantenha métricas de desempenho próximas às do modelo completo, ao mesmo tempo em que reduza dimensionalidade, custo computacional e tempo de execução.')
    add_paragraph(doc, 'Em um cenário ideal, a redução pode até melhorar o desempenho por reduzir ruído e redundância. O ponto mais relevante é que a performance não deve cair de forma significativa quando os atributos menos úteis forem removidos.')

    add_heading(doc, '10. Resultados observados', 1)
    add_paragraph(doc, 'Os experimentos executados corroboram essa expectativa no cenário proposto. O modelo baseline foi treinado com 40 atributos, enquanto a estratégia guiada por XAI reduziu para 10 atributos. O modelo reduzido manteve o F1-score em nível equivalente ou superior ao baseline, e a redução de atributos foi expressiva.')
    add_paragraph(doc, 'Dados observados no pipeline final: baseline com 40 atributos; modelo reduzido com 10 atributos; F1-score baseline de 0,8373; F1-score reduzido de 0,8601; redução de dimensionalidade de 75%; tempo de treinamento de 665,91 ms para 607,33 ms. Esses valores mostram que a estratégia de redução guiada por XAI foi capaz de manter ou melhorar o desempenho, enquanto reduziu drasticamente a quantidade de atributos. Isso sugere que a explicabilidade pode ser usada não apenas para interpretar o modelo, mas também para melhorar a qualidade do conjunto de dados usado no treino.')

    add_heading(doc, '11. Discussão', 1)
    add_paragraph(doc, 'Os resultados sugerem que a explicabilidade pode funcionar como um mecanismo bastante útil para seleção de atributos. Em particular, a importância dos atributos identificada por SHAP e a seleção com base em coeficientes de impacto e p-valor ajudam a distinguir variáveis realmente relevantes das redundantes ou ruidosas.')
    add_paragraph(doc, 'Essa conclusão é relevante porque mostra que a redução da dimensionalidade não precisa ser feita apenas por critério estatístico puro ou por abordagem baseada em tentativa e erro; também pode ser guiada pela explicação do modelo. Isso cria uma ponte entre dois paradigmas que frequentemente aparecem separados: a otimização preditiva e a interpretação do modelo.')
    add_paragraph(doc, 'No entanto, é preciso reconhecer que essa estratégia também apresenta limitações. O modelo depende diretamente da qualidade do treinamento inicial. Se o classificador base for instável, as explicações podem refletir vieses do modelo e não necessariamente a estrutura real do problema. Além disso, o custo de computação da explicabilidade pode ser elevado em datasets muito grandes.')
    add_paragraph(doc, 'Portanto, a abordagem é promissora, mas deve ser interpretada como um trade-off entre desempenho preditivo, explicabilidade, custo computacional, simplicidade do modelo e qualidade do conjunto de dados. Esse equilíbrio é central para a discussão final do trabalho.')

    add_heading(doc, '12. Limitações do estudo', 1)
    add_paragraph(doc, 'Apesar dos resultados promissores, o estudo apresenta algumas limitações importantes. Primeiro, o uso de dados sintéticos oferece controle e clareza, mas não substitui a complexidade de dados do mundo real. Segundo, a pesquisa se concentra principalmente na redução de atributos e não em outras formas de redução, como redução de instâncias ou compressão temporal. Terceiro, métodos como SHAP e otimização por Optuna exigem tempo e custo computacional. Quarto, a qualidade das explicações depende diretamente da qualidade do modelo treinado.')

    add_heading(doc, '13. Contribuição do estudo', 1)
    add_paragraph(doc, 'A principal contribuição desta pesquisa é demonstrar que explicabilidade e redução de dimensionalidade podem ser integradas em um mesmo pipeline. Em vez de tratar XAI apenas como ferramenta de interpretação final, o estudo mostra que ela pode ter um papel ativo na etapa de seleção de variáveis e na melhoria da eficiência do modelo.')
    add_paragraph(doc, 'Além disso, a comparação com um método tradicional de seleção, como RFE, fortalece a pesquisa ao mostrar que a explicabilidade não é apenas um complemento visual, mas uma estratégia analítica comparável e potencialmente útil. A contribuição, portanto, está em mostrar que a XAI pode servir como uma ponte entre performance e interpretabilidade, especialmente em problemas com alta dimensionalidade.')

    add_heading(doc, '14. Considerações finais', 1)
    add_paragraph(doc, 'Este estudo mostra que a utilização de explicabilidade como mecanismo de redução de dados é uma linha de pesquisa promissora. O modelo reduzido manteve ou melhorou a performance em um cenário de alta dimensionalidade, sugerindo que parte do conjunto original de atributos era redundante ou pouco informativa.')
    add_paragraph(doc, 'A partir dessa evidência, conclui-se que o uso de XAI para seleção de atributos é uma abordagem viável e potencialmente interessante para o desenvolvimento de pipelines mais eficientes e mais interpretáveis. A pesquisa também ressalta que esse tipo de solução deve ser entendido como um equilíbrio entre precisão, explicabilidade e custo computacional.')
    add_paragraph(doc, 'Em termos práticos, o trabalho sugere que a explicabilidade pode ser usada como estratégia de engenharia de dados, e não apenas como uma camada analítica final. Essa mudança de perspectiva é uma contribuição importante para a área.')

    add_paragraph(doc, 'Resumo: Este estudo investiga se técnicas de explicabilidade, como SHAP e LIME, podem ser usadas para reduzir a dimensionalidade de um conjunto de dados sem comprometer significativamente o desempenho preditivo. A pesquisa parte de um modelo baseline treinado com todas as variáveis e usa explicabilidade para identificar atributos relevantes, reduzir o conjunto de dados e treinar um novo modelo no subconjunto selecionado. Os resultados mostram que a redução guiada por XAI pode manter ou melhorar o desempenho preditivo, além de reduzir significativamente a dimensionalidade do problema.', bold=False, size=11)

    return doc


doc = build_document()
doc.save(OUTPUT_PATH)
print(f'Arquivo Word criado em: {OUTPUT_PATH}')
