# Apresentação Profissional: XAI Data Reduction

## Identidade e Proposta de Valor

**XAI Data Reduction** é uma pesquisa experimental e laboratório metodológico pioneiro em **Explicabilidade de Inteligência Artificial (XAI)** e **Engenharia de Recursos (Feature Engineering)**. O projeto investiga quantitativamente se a atribuição global de relevância baseada em Teoria dos Jogos (**TreeSHAP**) pode guiar a redução de dimensionalidade sem perda de capacidade preditiva, superando baselines clássicos das famílias *filter* (Mutual Information), *wrapper* (RFE) e *embedded* (Regressão Logística L1 Lasso).

O pipeline divide o problema em duas frentes complementares:
- **Seleção Global por Explicabilidade**: Uso do TreeSHAP refinado estatisticamente pelo algoritmo `shap-select` com pré-filtro de variância e colinearidade.
- **Auditoria Local Forense**: Emprego do **LIME** restrito a instâncias no limiar crítico de probabilidade ($P \approx 50\%$) para avaliar a esparsidade da explicação em decisões clínicas ambíguas.
- **Otimização Bayesiana (Optuna TPE)**: Calibração de hiperparâmetros no subespaço compacto resultante.

---

## Texto Sugerido para Publicação (LinkedIn / Portfólio / Artigo)

> *É possível eliminar 75% dos atributos de um modelo preditivo guiando a seleção exclusivamente por explicabilidade (XAI) e ainda aumentar a acurácia e o F1-Score?*
>
> Desenvolvi e executei o estudo experimental **XAI Data Reduction**, demonstrando na prática que técnicas de explicabilidade não servem apenas para justificar previsões no final, mas podem ser o próprio motor da engenharia de dados.
>
> 📊 **Resultados Chave Validados em Execução Real:**
> - **Compressão Dimensional Massiva**: Redução de **40 para apenas 10 variáveis clínicas** (**75,0% dos dados eliminados**).
> - **Ganho Preditivo**: O F1-Score no conjunto de teste independente saltou de **0,8373 para 0,8550 (+1,77 p.p.)** e a Acurácia subiu de **87,80% para 88,60%**, provando que a remoção de atributos ruidosos melhora a fronteira de decisão.
> - **Preservação Discriminativa**: O ROC-AUC manteve-se em **0,9455** (preservando 99,79% do baseline de 0,9475).
> - **Eficiência Computacional**: Tempo de treinamento acelerado em **8,5%** (1112,22 ms para 1017,42 ms).
> - **Auditoria Forense (LIME)**: No limiar de incerteza clínica ($P \approx 50\%$), o LIME confirmou **75,0% de esparsidade**, isolando os fatores que realmente governavam a dúvida diagnóstica.
>
> O repositório conta com pipeline 100% reproduzível, estudo de ablação progressiva, 17 cadernos teóricos e 6 aulas interativas para Google Colab!

---

## Evidências Reais Produzidas e Validadas

Todas as figuras foram produzidas via execução real do pipeline e encontram-se organizadas no diretório [`evidencias/`](file:///d:/projetos/xai_data_reduction/evidencias/):

1. **[evidencias/modulo1_baseline_metrics.png](file:///d:/projetos/xai_data_reduction/evidencias/modulo1_baseline_metrics.png)**: Matriz de confusão e curva ROC do modelo baseline de 40 atributos.
2. **[evidencias/modulo2_shap_summary.png](file:///d:/projetos/xai_data_reduction/evidencias/modulo2_shap_summary.png)**: Painel SHAP global (Bar Plot e Beeswarm Plot dos 40 atributos).
3. **[evidencias/modulo3_lime_local.png](file:///d:/projetos/xai_data_reduction/evidencias/modulo3_lime_local.png)**: Explicação forense LIME da decisão limiar ($P \approx 50\%$).
4. **[evidencias/modulo4_ablation_curves.png](file:///d:/projetos/xai_data_reduction/evidencias/modulo4_ablation_curves.png)**: Curvas de ablação comparando SHAP vs Mutual Information, RFE e L1 Lasso.
5. **[evidencias/modulo5_shap_select_analysis.png](file:///d:/projetos/xai_data_reduction/evidencias/modulo5_shap_select_analysis.png)**: Diagnóstico do teste de hipótese `shap-select` aprovando os 10 atributos ótimos.
6. **[evidencias/dashboard_final_comparativo.png](file:///d:/projetos/xai_data_reduction/evidencias/dashboard_final_comparativo.png)**: Dashboard executivo de 4 painéis sintetizando a vitória do modelo compacto.
7. **[evidencias/log_execucao_real.txt](file:///d:/projetos/xai_data_reduction/evidencias/log_execucao_real.txt)**: Log completo com tempo de execução, checkpoints e sanity checks.

---

## Documento Executivo Word (.docx)

O relatório editorial completo foi compilado e está pronto para apresentação:
- **Arquivo**: [XAI_Data_Reduction_Relatorio_Publicacao.docx](file:///d:/projetos/xai_data_reduction/XAI_Data_Reduction_Relatorio_Publicacao.docx)
- **Estrutura**: Resumo executivo, fundamentação teórica de XAI e Teoria dos Jogos, protocolo experimental, 6 figuras de alta resolução incorporadas com legendas técnicas, tabela comparativa multidimensional e considerações éticas de não-causalidade (NIST AI RMF e Molnar).

