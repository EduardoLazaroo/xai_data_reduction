# XAI Data Reduction

Projeto de estudo e demonstração de redução de dimensionalidade guiada por explicabilidade (XAI).

## Objetivo

Explorar se técnicas como SHAP e LIME podem ser usadas para identificar atributos relevantes, reduzir a dimensionalidade do conjunto de dados e manter ou melhorar o desempenho preditivo do modelo.

## Estrutura do projeto

- `modulo1_baseline.py` — baseline com Random Forest em todos os atributos
- `modulo2_shap.py` — análise global via SHAP
- `modulo3_lime.py` — explicação local via LIME
- `modulo4_ablation_comparison.py` — ablação comparando SHAP e RFE
- `modulo5_advanced_xai.py` — pré-filtro híbrido e seleção shap-select
- `modulo6_pipeline_final.py` — pipeline completo com otimização via Optuna
- `assets/` — imagens e artefatos finais gerados
- `docs/` — material textual e documentação do projeto
- `gerar_artigo_word.py` — gerador do artigo em Word

## Como executar

1. Crie o ambiente virtual
2. Instale as dependências
3. Execute os módulos em ordem:

```bash
python modulo1_baseline.py
python modulo2_shap.py
python modulo3_lime.py
python modulo4_ablation_comparison.py
python modulo5_advanced_xai.py
python modulo6_pipeline_final.py
```

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
