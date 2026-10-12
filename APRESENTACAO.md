# Apresentacao profissional

## Identidade e valor

XAI Data Reduction e um estudo experimental e didatico de reducao de dimensionalidade guiada por explicabilidade. O pipeline compara selecao por SHAP, Mutual Information, RFE e regressao L1; usa LIME para auditoria local, Random Forest para modelagem e Optuna para otimizacao. O conjunto de dados de saude e sintetico.

## Texto para publicacao

> Neste estudo, investiguei como explicabilidade pode orientar a reducao de atributos sem transformar a selecao de variaveis em uma caixa-preta. O pipeline compara SHAP, Mutual Information, RFE e regressao L1, adiciona auditoria local com LIME e mostra os efeitos da reducao em metricas e visualizacoes. O desafio foi conciliar desempenho, interpretabilidade e uma narrativa didatica que pudesse ser executada e discutida em sala ou no Google Colab. Como os dados sao sinteticos, o projeto deve ser entendido como laboratorio metodologico, nao como validacao clinica.

## Evidencias visuais recomendadas

1. Dashboard de baseline e modelo reduzido.
2. SHAP summary plot.
3. Explicacao local LIME.
4. Curva de ablation por numero de atributos.

## Prompt para imagem complementar

```text
Infografico cientifico de reducao de dimensionalidade explicavel: conjunto de atributos de saude sinteticos passa por SHAP, Mutual Information, RFE e L1; atributos selecionados alimentam modelo Random Forest; dashboard compara desempenho e interpretabilidade com um grafico SHAP e explicacao local, estilo didatico de ciencia de dados, fundo claro, sem logotipos.
```
