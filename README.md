# XAI Data Reduction

Projeto de estudo e demonstração de redução de dimensionalidade guiada por explicabilidade (XAI).

## Objetivo

Investigar se a seleção guiada por SHAP reduz a dimensionalidade sem perda relevante de desempenho, comparando-a com baselines representativos das três famílias clássicas: filter, wrapper e embedded. O LIME é usado exclusivamente para auditoria local de decisões individuais.

## Protocolo de avaliação

O pipeline separa quatro perguntas: desempenho preditivo (F1, recall e ROC-AUC), custo (tempo de treino e inferência), redução dimensional e interpretabilidade. A última é operacionalizada por proxies declarados: concentração da massa de |SHAP| no Top-10, esparsidade da explicação local LIME e percentual de atributos removidos. Esses proxies não representam causalidade nem substituem avaliação humana.

Os baselines são `Filter (Mutual Information)`, `Wrapper (RFE)` e `Embedded (Logística L1)`. A seleção e todos os rankings são aprendidos somente em `X_train`; o conjunto de teste é reservado para a comparação final.

## Estrutura do projeto

- `pipeline_completo.py` — script único em Python consolidando todas as etapas do experimento (Baseline, SHAP, LIME, Ablação, Pré-filtro Híbrido, shap-select, Optuna e Dashboard Executivo)
- `camadas_estudo/` — 17 apostilas aprofundadas em Markdown mapeando detalhadamente cada uma das etapas do `roteiro_de_estudo.txt` (de 00 a 16), com o "porquê", a "necessidade", exemplos conceituais, código e checkpoints
- `aulas_colab/` — 6 apostilas/aulas em Markdown prontas para o Google Colab, com fundamentação teórica, analogias, dúvidas comuns e código 100% comentado linha a linha:
  - `Aula_01_Baseline_Dimensionalidade.md`
  - `Aula_02_SHAP_Teoria_dos_Jogos.md`
  - `Aula_03_LIME_Sensibilidade_Vizinhanca.md`
  - `Aula_04_Ablacao_SHAP_vs_RFE.md`
  - `Aula_05_Engenharia_Avancada_BOLIMES_ShapSelect.md`
  - `Aula_06_Pipeline_Final_Optuna_Dashboard.md`
- `notebooks_colab/` — caderno integrador `XAI_Data_Reduction_Masterclass_Colab.md` para execução contínua no Google Colab
- `assets/` — gráficos e imagens de alta resolução gerados pelo pipeline
- `docs/` — roteiro de estudos e propostas do projeto
- `gerar_artigo_word.py` — gerador do artigo em Word

## Referências atualizadas

- [Molnar, Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) — explicabilidade, avaliação e limitações.
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) — avaliação e governança de sistemas de IA.
- [scikit-learn: Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html) — filtros, wrappers e métodos embedded.
- [Lundberg e Lee, A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions) — SHAP.
- [Ribeiro, Singh e Guestrin, Why Should I Trust You?](https://arxiv.org/abs/1602.04938) — LIME.

## Resultados Validados do Benchmark (Execução Real Instrumentada)

O pipeline foi executado e auditado integralmente em ambiente real via `pipeline_completo.py`, obtendo os seguintes resultados no conjunto de teste independente:

| Métrica / Dimensão Avaliada | Baseline (40 Atributos) | XAI Reduzido + Optuna (10 Atributos) | Variação Absoluta / Relativa |
|-----------------------------|-------------------------|--------------------------------------|------------------------------|
| **Espaço Dimensional** | 40 atributos | **10 atributos** | **-75.0% (Compressão Massiva)** |
| **Acurácia no Teste** | 0.8780 (87.80%) | **0.8860 (88.60%)** | **+0.80 p.p. (+0.91%)** |
| **F1-Score Clínico (Macro)**| 0.8373 | **0.8550** | **+1.77 p.p. (+2.11%)** |
| **Área sob a Curva ROC-AUC**| 0.9475 | **0.9455** | -0.0020 (99.79% preservado) |
| **Tempo de Treinamento** | 1112.22 ms | **1017.42 ms** | **-8.5% (Aceleração)** |
| **Latência de Inferência** | 55.38 ms | 95.57 ms | Limiar operacional de tempo real |
| **Concentração SHAP Top-10**| Dispersa (40 colunas) | **66.89%** | Alta densidade de relevância |
| **Esparsidade Local (LIME)**| 0.0% | **75.0%** | Explicação forense enxuta |
| **Hiperparâmetros Ótimos** | Padrão (n_estimators=100) | `n_estimators=184, max_depth=14, min_samples_split=7, min_samples_leaf=1` | Otimização Bayesiana Optuna TPE |

---

## Evidências Visuais e Técnicas Produzidas

Todas as evidências foram geradas a partir da execução real e encontram-se organizadas no diretório [`evidencias/`](file:///d:/projetos/xai_data_reduction/evidencias/):

1. **[evidencias/modulo1_baseline_metrics.png](file:///d:/projetos/xai_data_reduction/evidencias/modulo1_baseline_metrics.png)**: Matriz de confusão e curva ROC do modelo baseline de 40 atributos.
2. **[evidencias/modulo2_shap_summary.png](file:///d:/projetos/xai_data_reduction/evidencias/modulo2_shap_summary.png)**: Painel SHAP global (Bar Plot e Beeswarm Plot dos 40 atributos).
3. **[evidencias/modulo3_lime_local.png](file:///d:/projetos/xai_data_reduction/evidencias/modulo3_lime_local.png)**: Explicação forense LIME da decisão no limiar de incerteza ($P \approx 50\%$).
4. **[evidencias/modulo4_ablation_curves.png](file:///d:/projetos/xai_data_reduction/evidencias/modulo4_ablation_curves.png)**: Curvas de ablação comparando SHAP vs Mutual Information, RFE e L1 Lasso.
5. **[evidencias/modulo5_shap_select_analysis.png](file:///d:/projetos/xai_data_reduction/evidencias/modulo5_shap_select_analysis.png)**: Diagnóstico do teste de hipótese `shap-select` aprovando os 10 atributos ótimos.
6. **[evidencias/dashboard_final_comparativo.png](file:///d:/projetos/xai_data_reduction/evidencias/dashboard_final_comparativo.png)**: Dashboard executivo de 4 painéis sintetizando a vitória do modelo compacto.
7. **[evidencias/log_execucao_real.txt](file:///d:/projetos/xai_data_reduction/evidencias/log_execucao_real.txt)**: Log completo com tempo de execução, checkpoints e sanity checks.

---

## Documento de Publicação Executivo (.docx)

O relatório definitivo de publicação e auditoria científica foi compilado:
- **Arquivo**: [XAI_Data_Reduction_Relatorio_Publicacao.docx](file:///d:/projetos/xai_data_reduction/XAI_Data_Reduction_Relatorio_Publicacao.docx)
- **Conteúdo**: Resumo executivo, fundamentação teórica de XAI, Teoria dos Jogos (Shapley), protocolo experimental detalhado, 6 figuras de alta resolução incorporadas com legendas técnicas e considerações éticas de não-causalidade alinhadas ao NIST AI RMF.

---

## Como executar

1. Ative o ambiente virtual
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o pipeline completo de ponta a ponta:

```bash
python pipeline_completo.py
```

Ou abra qualquer uma das aulas em `aulas_colab/` diretamente no **Google Colab** para fins didáticos e interativos.

## Dependências principais

- Python 3.10+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- shap
- lime
- optuna
- statsmodels
- python-docx

## Observação

O projeto foi concebido como base didática e experimental para um artigo acadêmico sobre XAI aplicada à redução de dados.

