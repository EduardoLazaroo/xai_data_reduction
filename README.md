# XAI Data Reduction

Projeto de estudo e demonstração de redução de dimensionalidade guiada por explicabilidade (XAI).

## Objetivo

Explorar se técnicas como SHAP e LIME podem ser usadas para identificar atributos relevantes, reduzir a dimensionalidade do conjunto de dados e manter ou melhorar o desempenho preditivo do modelo.

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
