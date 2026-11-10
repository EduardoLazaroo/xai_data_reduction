# XAI Data Reduction: Seleção de Características e Redução de Dados Guiada por Explicabilidade

Projeto de pesquisa aplicada sobre seleção de atributos em um benchmark sintético controlado (`make_classification`), com controle exato de *ground truth* (10 informativos, 10 redundantes por combinação linear e 20 ruídos estocásticos). Os rótulos e nomes das colunas são ilustrativos para rastreamento experimental; os resultados não validam uso médico nem inferência causal.

- **Autor:** Eduardo Lázaro Roesler de Oliveira
- **Instituição:** Centro Universitário Eurípides de Marília (UNIVEM)
- **Programa:** Ciência da Computação / Pós-Graduação
- **Data de Entrega da Documentação:** 10/11
- **Entregáveis Centrais (100% Prontos):**
  - **a. Apresentação do projeto (PDF):** [`apresentacao_projeto_xai.pdf`](apresentacao_projeto_xai.pdf) (11 slides widescreen 16:9, cronometrados para 12 min de exposição + 3 min de perguntas).
  - **b. Implementação realizada:** [`pipeline_completo.py`](pipeline_completo.py), com execução reproduzível e artefatos experimentais em [`assets/`](assets/).
  - **c. Artigo científico IEEE (PDF):** [`artigo_cientifico_ieee.pdf`](artigo_cientifico_ieee.pdf), formatado no padrão IEEE de duas colunas (até 3 páginas).
  - **Relatório técnico complementar:** [`relatorio_tecnico_projeto.pdf`](relatorio_tecnico_projeto.pdf).

---

## 🎯 Questão Central de Pesquisa

> *"Em um benchmark sintético controlado, como uma seleção de atributos informada por pré-filtragem VIF e TreeSHAP Out-of-Fold (`shap-select`) se compara a métodos filter, wrapper e embedded em recuperação do subespaço informativo, rejeição de ruído, desempenho preditivo e custo computacional, e o que a auditoria LIME na fronteira de decisão permite concluir sobre fidelidade local?"*

### Hipóteses Científicas Formais
- **H1 (desempenho preditivo):** eliminar ao menos 50% dos atributos mantendo ou superando o F1-Score do Baseline (queda máxima tolerada de 0,05) no Holdout e na Validação Cruzada de 5 Folds.
- **H2 (pureza de seleção e custo):** recuperar mais atributos informativos verdadeiros e rejeitar 100% do ruído sintético frente aos métodos comparados, avaliando o trade-off de tempo contra o RFE.
- **H3 (auditoria local LIME):** na observação crítica de fronteira ($P = 50\%$) e na coorte das 20 instâncias mais incertas ($N = 20$), eliminar a contaminação por atributos redundantes/ruído nas regras locais sem degradar o ajuste $R^2$ do LIME.

---

## 📦 Estrutura dos Entregáveis do Projeto

```
xai_data_reduction/
│
├── apresentacao_projeto_xai.pdf        # APRESENTAÇÃO OFICIAL (PDF Widescreen 16:9, 11 Slides, 15 min)
├── artigo_cientifico_ieee.pdf          # ARTIGO CIENTÍFICO OFICIAL (Padrão IEEE, até 3 Páginas)
├── artigo_cientifico_ieee.docx         # Fonte editável do Artigo IEEE
├── relatorio_tecnico_projeto.pdf       # Relatório técnico complementar
├── relatorio_tecnico_projeto.docx      # Fonte editável do Relatório Técnico
│
├── pipeline_completo.py                # Pipeline científico (Pré-Filtro VIF + TreeSHAP OOF + Optuna + LIME)
├── gerar_apresentacao_pdf.py           # Gerador da apresentação executiva em PDF
├── gerar_artigo_ieee.py                # Gerador e validador do artigo IEEE de até 3 páginas
├── gerar_relatorio_tecnico.py          # Gerador do relatório técnico complementar
├── requirements.txt                    # Dependências Python completas do projeto
│
├── assets/                             # Diagnósticos e gráficos experimentais em alta resolução (300 DPI)
│   ├── modulo1_baseline_metrics.png        # Matriz de Confusão e Curva ROC do Baseline (40 atributos)
│   ├── modulo2_shap_summary.png            # Bar Plot e Beeswarm Global TreeSHAP
│   ├── modulo3_lime_local.png              # Auditoria Local LIME na observação limítrofe e coorte N=20
│   ├── modulo4_ablation_curves.png         # Curvas de ablação progressiva multivariada (40 a 2 atributos)
│   ├── modulo5_shap_select_analysis.png    # Seleção inferencial shap-select Out-of-Fold (p < 0,05)
│   ├── dashboard_final_comparativo.png     # Dashboard Executivo Integrador de 4 Painéis
│   └── tabela_comparativa_completa.csv     # Dados consolidados do benchmark (Holdout + 5-Fold CV)
│
└── docs/                               # Cópia sincronizada dos PDFs de submissão
    ├── apresentacao_projeto_xai.pdf
    ├── artigo_cientifico_ieee.pdf
    └── relatorio_tecnico_projeto.pdf
```

---

## 🔬 Benchmark Comparativo Consolidado (Holdout 75/25 e Validação Cruzada 5-Fold)

Execução com $N = 2.000$ observações sintéticas e $D = 40$ atributos (10 informativos, 10 redundantes por combinação linear e 20 ruídos puros):

| Método / Abordagem | Família | Inform. | Redund. | Ruído | Rej. Ruído | F1 Holdout | F1 CV (5-Fold) | Acurácia | ROC-AUC | Seleção (s) | Treino (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| **XAI Proposto (shap-select OOF + Optuna)** | **XAI** | **10/10** | **0/10** | **0/20** | **100,0%** | **0,8872** | **0,8864 ± 0,012** | **0,9100** | **0,9734** | 79,30 | 762,2 |
| **XAI Proposto (shap-select OOF Puro)** | **XAI** | **10/10** | **0/10** | **0/20** | **100,0%** | **0,8861** | **0,8873 ± 0,012** | **0,9080** | **0,9686** | 11,04 | 396,0 |
| Wrapper (RFE) | Wrapper | 7/10 | 3/10 | 0/20 | 100,0% | 0,8759 | 0,8533 ± 0,020 | 0,8980 | 0,9580 | 4,55 | 414,3 |
| Embedded (Lasso L1) | Embedded | 6/10 | 3/10 | 1/20 | 95,0% | 0,8689 | 0,8621 ± 0,014 | 0,8920 | 0,9626 | 0,05 | 427,8 |
| Embedded (Tree MDI) | Embedded | 7/10 | 3/10 | 0/20 | 100,0% | 0,8558 | 0,8409 ± 0,025 | 0,8800 | 0,9493 | 0,01 | 355,7 |
| Filter (ANOVA F) | Filter | 7/10 | 3/10 | 0/20 | 100,0% | 0,8550 | 0,8186 ± 0,021 | 0,8840 | 0,9510 | 0,005 | 361,2 |
| XAI (TreeSHAP s/ VIF) | XAI | 7/10 | 3/10 | 0/20 | 100,0% | 0,8465 | 0,8288 ± 0,023 | 0,8760 | 0,9489 | 13,56 | 351,0 |
| Filter (Mutual Info) | Filter | 5/10 | 4/10 | 1/20 | 95,0% | 0,8365 | 0,8229 ± 0,021 | 0,8640 | 0,9378 | 0,41 | 358,5 |
| **Baseline Completo (40 Atributos)** | **Referência** | **10/10** | **10/10** | **20/20** | **0,0%** | **0,8856** | **0,8649 ± 0,016** | **0,9060** | **0,9655** | 0,00 | 558,5 |

### Principais Conclusões Científicas:
1. **H1 Confirmada com Superação do Baseline (Mitigação do Fenômeno de Hughes):** A remoção de 75% das variáveis (40 → 10) elevou o F1-Score de `0,8856` para `0,8861` (mesmo estimador padrão, *ceteris paribus*) e `0,8872` (com Optuna) no Holdout, além de saltar de `0,8649` para `0,8873` na validação cruzada de 5 folds.
2. **H2 Confirmada em Recuperação Exata do Subespaço (10/10) e Trade-off de Custo:** O pré-filtro de colinearidade multivariada ($\text{VIF} \le 10$) combinado ao `shap-select` *Out-of-Fold* ($p < 0{,}05$) foi o único método a recuperar **10/10 atributos informativos**, **0/10 redundantes** e **0/20 ruídos** sem corte Top-K manual. Em contrapartida, o tempo de seleção offline (`11,04 s` puro; `79,30 s` com Optuna) é maior que o RFE (`4,55 s`).
3. **H3 Confirmada na Observação Crítica e na Coorte de Fronteira ($N = 20$):** Na observação mais incerta (#295, $P = 50{,}0\%$), o $R^2$ local do LIME subiu de `0,3066` para `0,3851`. Na coorte das 20 instâncias de fronteira, o $R^2$ médio passou de `0,2478` para `0,2555` e a presença de variáveis redundantes/ruído entre as 10 principais regras locais caiu de `3,80/10` para **`0,00/10`**.

---

## 🚀 Como Executar e Reproduzir

```bash
.\.venv\Scripts\python.exe pipeline_completo.py
.\.venv\Scripts\python.exe gerar_artigo_ieee.py
.\.venv\Scripts\python.exe gerar_apresentacao_pdf.py
```
