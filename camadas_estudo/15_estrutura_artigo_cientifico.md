# Camada 15: A Estrutura de um Artigo Científico — Da Teoria ao Padrão Acadêmico

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 15  
**Contexto Técnico:** [gerar_artigo_word.py](file:///c:/Users/eduar/projetos/xai_data_reduction/gerar_artigo_word.py) e [docs/artigo_xai_reduction.docx](file:///c:/Users/eduar/projetos/xai_data_reduction/docs/artigo_xai_reduction.docx)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> Compreender a arquitetura canônica de um **Artigo Científico em Computação e Inteligência Artificial**. Dominar a estrutura internacional **IMRaD (*Introduction, Methods, Results, and Discussion*)**, aprender a diferença cirúrgica entre **"Resultados"** (a frieza dos fatos) e **"Discussão"** (o calor da interpretação causal), entender como formular e testar **Hipóteses Científicas Formais ($H_1$ e $H_2$)**, e descobrir como transformar linhas de código em um artigo acadêmico publicável em periódicos de alto impacto (Qualis A / Q1).

---

## Sumário da Aula

- [Subcamada 15.1: A Analogia da Catedral e a Investigação Policial](#subcamada-151-a-analogia-da-catedral-e-a-investigação-policial)
- [Subcamada 15.2: O Formato Canônico IMRaD (O Funil Científico)](#subcamada-152-o-formato-canônico-imrad-o-funil-científico)
- [Subcamada 15.3: A Fronteira Proibida: Resultados vs. Discussão](#subcamada-153-a-fronteira-proibida-resultados-vs-discussão)
- [Subcamada 15.4: As Duas Hipóteses Científicas Formais do Nosso Projeto (H1 e H2)](#subcamada-154-as-duas-hipóteses-científicas-formais-do-nosso-projeto-h1-e-h2)
- [Subcamada 15.5: Laboratório Lúdico no Colab (Toy Example: Validador Estatístico de Hipóteses)](#subcamada-155-laboratório-lúdico-no-colab-toy-example-validador-estatístico-de-hipóteses)
- [Subcamada 15.6: O Momento Sério da Nossa Aplicação (Do Código Python ao Artigo Word Oficial)](#subcamada-156-o-momento-sério-da-nossa-aplicação-do-código-python-ao-artigo-word-oficial)
- [Subcamada 15.7: Checkpoint de Autonomia & Fixação Ativa](#subcamada-157-checkpoint-de-autonomia--fixação-ativa)

---

## Subcamada 15.1: A Analogia da Catedral e a Investigação Policial

Você não consegue construir uma catedral gótica apenas empilhando pedras aleatoriamente no chão. Se não houver uma planta rigorosa, as paredes desabam:
- A catedral precisa de **alicerces profundos** (Fundamentação Teórica).
- **Pilares mestres de sustentação** (Metodologia Experimental).
- Um **altar central visível** (Resultados Numéricos).
- Uma **cúpula com vitrais coloridos** que dão sentido e beleza a toda a obra (Discussão e Conclusão).

```
                      O FUNIL CIENTÍFICO DO ARTIGO (IMRaD)
                      
          \   1. INTRODUÇÃO (AMPLA)   /       -> O problema do mundo real:
           \                         /           hospitais gastam milhões em exames!
            \  2. METODOLOGIA       /         -> O afunilamento: o que nós fizemos?
             \  (PRECISA & RESTRITA)/            Dataset sintético, SHAP e Optuna.
              | 3. RESULTADOS       |         -> O gargalo: os números frios!
             /   (FATOS BRUTOS)      \           Tabelas de F1, latência e tempos.
            /  4. DISCUSSÃO CRÍTICA   \       -> A reabertura: o que os números significam?
           /   (CONECTA COM O MUNDO)   \         Eliminamos o ruído e fechamos o gap.
          /    5. CONCLUSÃO (IMPACTO)   \     -> A nova verdade científica para o futuro!
```

Um artigo acadêmico funciona exatamente como o relatório de um detetive de elite apresentado ao juiz:
1. Explica o crime investigado (Introdução).
2. Apresenta os métodos periciais de coleta de DNA (Metodologia).
3. Mostra o laudo laboratorial neutro (Resultados).
4. Explica por que o laudo comprova a culpa do réu e refuta os álibis da defesa (Discussão).
5. Solicita o veredito final (Conclusão).

---

## Subcamada 15.2: O Formato Canônico IMRaD (O Funil Científico)

O acrônimo **IMRaD** rege mais de 90% de toda a literatura científica mundial nas áreas de Exatas, Saúde e Computação:

| Seção IMRaD | Pergunta que Responde | Tempo Verbal | O Que Fazer? |
| :--- | :--- | :--- | :--- |
| **I — Introdução (*Introduction*)** | *"Qual é o problema no mundo e por que ele ainda não foi resolvido?"* | Presente | Contextualiza a coleta massiva de dados e o mal da dimensionalidade. Apresenta o objetivo e as hipóteses. |
| **M — Métodos (*Methods*)** | *"Como você executou o experimento para que qualquer cientista possa repetir?"* | Passado | Descreve as 40 variáveis, o Random Forest, o TreeSHAP, o Pré-Filtro, o shap-select e o Optuna. |
| **R — Resultados (*Results*)** | *"O que os seus olhos e computadores viram e mediram?"* | Passado | Apresenta tabelas, gráficos de ablação, matriz de confusão e tempos de CPU sem dar opiniões. |
| **D — Discussão (*Discussion*)** | *"O que esses números significam e por que eles confirmam suas hipóteses?"* | Presente/Passado | Explica o porquê dos ganhos, compara com o RFE tradicional, assume limitações e sugere o futuro. |

---

## Subcamada 15.3: A Fronteira Proibida: Resultados vs. Discussão

O erro que mais reprova artigos em revistas internacionais é a **confusão entre fato e interpretação**:

```
    ❌ ERRADO (MISTURADO NA SEÇÃO DE RESULTADOS):
    "O modelo atingiu F1 de 0.86 porque a eliminação dos ruídos foi maravilhosa para o hospital."
    (O revisor vai te repreender: "Porque foi maravilhosa" é uma opinião! Não cabe em Resultados!)
    
    ✅ CORRETO NA SEÇÃO DE RESULTADOS:
    "O modelo reduzido alcançou F1-Score de 0.8610 no conjunto de teste, ante 0.8373 do Baseline."
    (Fato neutro, frio e inquestionável.)
    
    ✅ CORRETO NA SEÇÃO DE DISCUSSÃO:
    "O incremento de 2.37 pontos percentuais no F1-Score do modelo reduzido corrobora a tese de
    que os 20 atributos de ruído induziam ramificações espúrias no Baseline, conforme alertado
    pelo Fenômeno de Hughes (1968)."
    (Aqui sim cabe a teoria, o porquê e a conexão com a literatura!)
```

---

## Subcamada 15.4: As Duas Hipóteses Científicas Formais do Nosso Projeto (H1 e H2)

Toda a nossa pesquisa foi desenhada para testar e validar duas **Hipóteses Formais**:

1. **Hipótese 1 ($H_1$ — Preservação da Qualidade Diagnóstica):**  
   *"É viável eliminar pelo menos 50% dos atributos de um dataset clínico com alta dimensionalidade utilizando Inteligência Artificial Explicável (XAI), mantendo o $F_1$-Score clínico dentro de uma margem de equivalência de $\pm 0.05$ em relação ao Baseline."*
2. **Hipótese 2 ($H_2$ — Eficiência Operacional e Computacional):**  
   *"O modelo enxuto treinado no subconjunto selecionado apresentará uma redução de pelo menos 40% no tempo de treinamento e na latência de inferência em comparação ao modelo de força bruta."*

Se ao final do experimento os números confirmarem $H_1$ e $H_2$, **o artigo está validado cientificamente!**

---

## Subcamada 15.5: Laboratório Lúdico no Colab (Toy Example: Validador Estatístico de Hipóteses)

Copie e execute no [Google Colab](https://colab.research.google.com) para rodar o testador automático de hipóteses:

```python
# =============================================================================
# LABORATÓRIO DIDÁTICO: VALIDADOR AUTOMÁTICO DE HIPÓTESES CIENTÍFICAS
# Objetivo: Testar programaticamente se H1 e H2 foram confirmadas
# =============================================================================
import pandas as pd

# Resultados do Experimento
dados_baseline = {"atributos": 40, "f1": 0.8373, "tempo_treino": 601.7, "latencia": 30.8}
dados_campeao  = {"atributos": 10, "f1": 0.8610, "tempo_treino": 185.2, "latencia": 12.4}

# 1. Teste da Hipótese H1 (Manutenção Preditiva com Poda >= 50%)
reducao_atributos = (1 - dados_campeao["atributos"] / dados_baseline["atributos"]) * 100
delta_f1 = dados_campeao["f1"] - dados_baseline["f1"]
h1_confirmada = (reducao_atributos >= 50.0) and (abs(delta_f1) <= 0.05 or delta_f1 > 0)

# 2. Teste da Hipótese H2 (Eficiência Operacional >= 40%)
reducao_treino = (1 - dados_campeao["tempo_treino"] / dados_baseline["tempo_treino"]) * 100
reducao_latencia = (1 - dados_campeao["latencia"] / dados_baseline["latencia"]) * 100
h2_confirmada = (reducao_treino >= 40.0) and (reducao_latencia >= 40.0)

# 3. Emissão do Relatório Acadêmico
print("=" * 65)
print("PARECER CIENTÍFICO FORMAL DO COMITÊ EXPERIMENTAL:")
print("=" * 65)
print(f"HIPÓTESE 1 (Manutenção Diagnóstica):")
print(f"  • Redução de Atributos : {reducao_atributos:.1f}% (Meta: >= 50%)")
print(f"  • Delta F1-Score       : {delta_f1:+.4f} (Meta: tolerância +/- 0.05)")
print(f"  • Veredito             : {'✅ CONFIRMADA COM SUCESSO!' if h1_confirmada else '❌ REJEITADA'}")
print("-" * 65)
print(f"HIPÓTESE 2 (Eficiência Computacional):")
print(f"  • Redução Tempo Treino : {reducao_treino:.1f}% (Meta: >= 40%)")
print(f"  • Redução Latência     : {reducao_latencia:.1f}% (Meta: >= 40%)")
print(f"  • Veredito             : {'✅ CONFIRMADA COM SUCESSO!' if h2_confirmada else '❌ REJEITADA'}")
print("=" * 65)
```

---

## Subcamada 15.6: O Momento Sério da Nossa Aplicação (Do Código Python ao Artigo Word Oficial)

No nosso ecossistema técnico, nós criamos uma ponte automatizada espetacular: o script [gerar_artigo_word.py](file:///c:/Users/eduar/projetos/xai_data_reduction/gerar_artigo_word.py).  
Ele pega todos os dados calculados pelo [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py) e compila automaticamente o artigo científico completo no formato `.docx` ([artigo_xai_reduction.docx](file:///c:/Users/eduar/projetos/xai_data_reduction/docs/artigo_xai_reduction.docx)), formatado no padrão acadêmico internacional!

```
    [ pipeline_completo.py ]
               │
               ▼
    Gera tabelas de KPIs e gráficos em assets/
               │
               ▼
    [ gerar_artigo_word.py ]
               │
               ▼
    [ docs/artigo_xai_reduction.docx ]
    • 14 Seções estruturadas no padrão IMRaD
    • Tabelas formatadas em padrão APA
    • Inserção automática de figuras de alta resolução
    • Formatação ABNT/IEEE pronta para submissão!
```

---

### 15.6.1 Quadro de Validação Científica das Hipóteses

| Hipótese Formal | Meta Quantitativa | Resultado Alcançado no Projeto | Veredito Acadêmico |
| :--- | :--- | :--- | :--- |
| **Hipótese $H_1$ (Qualidade)** | Cortar $\ge 50\%$ das variáveis mantendo o $F_1$ em $\pm 0.05$. | **Cortou 75.0% dos atributos** e o $F_1$ **aumentou em $+0.0237$** ($0.8610$). | **CONFIRMADA E SUPERADA!** |
| **Hipótese $H_2$ (Eficiência)** | Acelerar o treinamento e a latência em $\ge 40\%$. | **Tempo caiu 69.2%** e **Latência caiu 59.7%**. | **CONFIRMADA E SUPERADA!** |

---

## Subcamada 15.7: Checkpoint de Autonomia & Fixação Ativa

Responda para fixar a estrutura acadêmica:

1. **Por que um revisor de artigo científico rejeitaria um manuscrito se você escrevesse especulações e hipóteses na seção de "Resultados"?**
2. **Explique a metáfora do funil para o formato IMRaD: por que a Introdução e a Discussão são largas, enquanto a Metodologia e os Resultados são estreitos?**
3. **Qual é o papel das hipóteses científicas ($H_1$ e $H_2$) em uma pesquisa de computação? Como elas impedem que o pesquisador "mude o alvo depois que a flecha foi disparada"?**
4. **Desafio no Colab:** Na Subcamada 15.5, altere os dados do campeão para simular um cenário onde o modelo enxuto cortou 80% das variáveis, mas o F1 caiu para `0.70`. A Hipótese $H_1$ continuou aprovada ou o testador acusou rejeição?
