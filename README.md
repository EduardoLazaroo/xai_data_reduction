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

## Como executar

1. Crie e ative o ambiente virtual
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
