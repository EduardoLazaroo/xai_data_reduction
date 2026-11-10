# XAI Data Reduction: Seleção de Características e Redução de Dados Guiada por Explicabilidade

Projeto de pesquisa aplicada sobre seleção de atributos em um benchmark sintético controlado. Os rótulos clínicos e nomes das colunas são ilustrativos; os resultados não validam uso médico nem inferência causal.

- **Autor:** Eduardo Lázaro Roesler de Oliveira
- **Instituição:** Centro Universitário Eurípides de Marília (UNIVEM)
- **Programa:** Ciência da Computação / Pós-Graduação
- **Data de Entrega da Documentação:** 10/11
- **Entregáveis Centrais:**
  - **Artigo científico IEEE (PDF):** [`artigo_cientifico_ieee.pdf`](artigo_cientifico_ieee.pdf), até 3 páginas.
  - **Implementação realizada:** [`pipeline_completo.py`](pipeline_completo.py), com execução reproduzível e artefatos experimentais.
  - **Apresentação (PDF):** já preparada pelo autor e fora do escopo deste repositório; nenhum slide será gerado aqui.
  - **Relatório técnico:** material complementar, não substitui os entregáveis exigidos.

Os arquivos `gerar_artigo_word.py`, `artigo_xai_reduction.docx` e `docs/relatorio_tecnico_completo.pdf` são versões legadas e não devem ser usados na entrega. Use apenas o artigo IEEE listado acima e o relatório técnico complementar identificado pelo nome `relatorio_tecnico_projeto`.

---

## 🎯 Questão Central de Pesquisa

> *"Em um benchmark sintético controlado, como uma seleção informada por TreeSHAP e shap-select se compara a métodos filter, wrapper e embedded em desempenho preditivo, rejeição de ruído e custo de seleção, e o que uma auditoria LIME de uma observação permite concluir sobre fidelidade local?"*

### Hipóteses Científicas Formais
- **H1 (desempenho):** no holdout sintético, reduzir ao menos 50% dos atributos sem queda de F1 superior a 0,05.
- **H2 (seleção):** rejeitar mais ruído que os métodos comparados e ter menor tempo de seleção que RFE; a execução atual não atende ao critério de custo.
- **H3 (auditoria local):** o R² do LIME não diminui para a observação selecionada; é uma medida local de uma instância, não de explicabilidade geral.

---

## 📦 Estrutura dos Entregáveis do Projeto

```
xai_data_reduction/
│
├── artigo_cientifico_ieee.pdf          # ARTIGO CIENTÍFICO OFICIAL (IEEE, 3 Páginas Estritas)
├── artigo_cientifico_ieee.docx         # Fonte editável do Artigo IEEE
├── relatorio_tecnico_projeto.pdf       # Relatório técnico complementar
├── relatorio_tecnico_projeto.docx      # Fonte editável do Relatório Técnico
│
├── pipeline_completo.py                # Pipeline científico automatizado (ML + XAI + Optuna)
├── gerar_artigo_ieee.py                # Gerador e validador do artigo IEEE de até 3 páginas
├── gerar_relatorio_tecnico.py          # Gerador e validador do relatório técnico expandido
├── requirements.txt                    # Dependências Python completas do projeto
│
├── assets/                             # Diagnósticos e gráficos experimentais em alta resolução
│   ├── modulo1_baseline_metrics.png        # Matriz de Confusão e Curva ROC do Baseline
│   ├── modulo2_shap_summary.png            # Bar Plot e Beeswarm Global TreeSHAP
│   ├── modulo3_lime_local.png              # Auditoria Forense LIME no limiar P ≈ 50%
│   ├── modulo4_ablation_curves.png         # Curvas de ablação progressiva multivariada
│   ├── modulo5_shap_select_analysis.png    # Diagnósticos estatísticos com p-valor
│   ├── dashboard_final_comparativo.png     # Dashboard Executivo Integrador de 4 Painéis
│   └── tabela_comparativa_completa.csv     # Dados brutos do benchmark de todas as famílias
│
└── docs/                               # Cópia sincronizada de submissão e documentação institucional
    ├── artigo_cientifico_ieee.pdf
    ├── relatorio_tecnico_projeto.pdf
    ├── proposta_projeto_xai.docx
    └── roteiro_de_estudo.txt
```

---

## 🔬 Benchmark Comparativo Consolidado

Uma execução com 2.000 observações sintéticas, 40 atributos (10 informativos, 10 redundantes e 20 de ruído) e divisão estratificada 75/25. Os métodos por ranking usam K = 10; o baseline usa as 40 colunas. As contagens de atributos verdadeiros decorrem apenas da construção sintética:

| Método | Família | Informativos | Ruído selecionado | Rejeição de ruído | F1 | Acurácia | ROC-AUC | Seleção (s) | Treino (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| **XAI (shap-select + Optuna)** | XAI | 7/10 | 0/20 | 100,0% | 0,8395 | 0,8700 | 0,9442 | 14,1670 | 377,3 |
| Embedded (Lasso L1) | Embedded | 6/10 | 1/20 | 95,0% | 0,8689 | 0,8920 | 0,9626 | 0,0333 | 427,8 |
| Wrapper (RFE) | Wrapper | 7/10 | 0/20 | 100,0% | 0,8759 | 0,8980 | 0,9580 | 3,0661 | 414,3 |
| XAI (TreeSHAP) | XAI | 7/10 | 0/20 | 100,0% | 0,8465 | 0,8760 | 0,9489 | 8,9278 | 351,0 |
| Filter (ANOVA F) | Filter | 7/10 | 0/20 | 100,0% | 0,8550 | 0,8840 | 0,9510 | 0,0094 | 361,2 |
| Embedded (Tree MDI) | Embedded | 7/10 | 0/20 | 100,0% | 0,8558 | 0,8800 | 0,9493 | 0,0084 | 355,7 |
| Filter (Mutual Information) | Filter | 5/10 | 1/20 | 95,0% | 0,8365 | 0,8640 | 0,9378 | 0,3144 | 358,5 |
| **Baseline (40 atributos)** | Referência | 10/10 | 20/20 | 0,0% | 0,8856 | 0,9060 | 0,9655 | 0,0000 | 558,5 |

Os tempos são medições desta execução e dependem do hardware e do ambiente; a tabela completa e os valores atualizados estão no CSV.

### Principais Conclusões Científicas:
1. **H1, neste holdout:** 75% dos atributos foram removidos; o F1 caiu 0,0461, dentro por pouco do limite predefinido de 0,05. Isso é resultado descritivo de uma única partição.
2. **H2 não foi atendida em custo:** a proposta rejeitou todo o ruído sintético, empatando com RFE, ANOVA e Tree MDI; a seleção da proposta levou mais tempo que RFE nesta execução.
3. **Custo do modelo final:** o treino foi menor, mas a latência de inferência aumentou ligeiramente (34,49 ms contra 32,14 ms) nesta medição; os tempos dependem do hardware e não demonstram escalabilidade.
4. **LIME local:** o R² foi 0,2029 no baseline e 0,2679 no reduzido para uma única observação, indicando ajuste local limitado e sem conclusão geral.
5. **Cognição:** 10 atributos excedem o limite superior 9 da heurística de Miller; não houve estudo com usuários.

---

## 🚀 Como Executar e Reproduzir

### 1. Preparação do Ambiente
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Executar o Pipeline Experimental
Executa baseline, TreeSHAP, benchmark, ablação, seleção, Optuna e auditoria LIME, atualizando os gráficos e o CSV. Os números de execução e tempos são escritos em `assets/tabela_comparativa_completa.csv`:
```bash
python pipeline_completo.py
```

### 3. Compilar o Artigo Científico no Padrão IEEE (Até 3 Páginas)
Depois de executar o pipeline, gera o documento em duas colunas IEEE a partir do CSV e valida a contagem de páginas:
```bash
python gerar_artigo_ieee.py
```

### 4. Compilar o Relatório Técnico Complementar
Gera o relatório técnico complementar com metodologia, resultados e figuras; a contagem de páginas é informada pelo script:
```bash
python gerar_relatorio_tecnico.py
```

---

## 📋 Conformidade com os Critérios de Avaliação

- [x] **Artigo científico:** PDF IEEE de até 3 páginas, orientado pela questão de pesquisa e com método, resultados, limitações e referências.
- [x] **Implementação:** pipeline e dependências documentadas; execute-o para atualizar os resultados.
- [x] **Apresentação:** PDF já preparado pelo autor; não será gerado nem alterado neste projeto.
- [ ] **Revisão final de entrega:** conferir o PDF final do artigo e incluir a apresentação já pronta no pacote de entrega até 10/11.
