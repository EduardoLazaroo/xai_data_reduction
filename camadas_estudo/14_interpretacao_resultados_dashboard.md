# Camada 14: Interpretacao de Resultados e Dashboard Executivo

**Trilha:** XAI Aplicada à Redução de Dados em Machine Learning
**Aplicacao:** classificacao binaria em benchmark sintetico; os rotulos de saude sao ilustrativos.
**Código de referência:** [pipeline_completo.py](../pipeline_completo.py), funções com dashboard e comparativo de KPIs

> **Objetivo da aula:** Analisar a sintese executiva e cientifica dos resultados experimentais, interpretando redução dimensional, tempo de treinamento, latencia, F1, concentração Top-10 do SHAP e esparsidade local do LIME. Esses últimos são proxies operacionais de interpretabilidade, não medidas causais nem substitutos de avaliação humana.

## Campo Didático: Um Dashboard e um Argumento

Leia cada painel em duas camadas: primeiro descreva literalmente o que o grafico mostra; depois interprete apenas o que foi medido. O roteiro e **dimensao, treino, tempo de predicao e desempenho no holdout**. O benchmark nao mede erro clinico ou custo operacional.

```text
grafico -> numero -> comparacao baseline/proposta -> limite da conclusao
```

Observe se a reducao de atributos preserva F1 e recall, e se o ganho de tempo e maior que a variacao de medicao. O erro comum e usar escala visual para sugerir uma melhora que os numeros nao sustentam. A ponte para a Camada 15 e transformar essa leitura em metodologia e hipoteses publicaveis.

---

### Roteiro de dominio

Leia o dashboard em ordem: **o que mudou, quanto mudou e quais etapas ficaram mais lentas ou mais rapidas**. Para cada painel escreva uma frase factual e uma interpretativa separadas. Nao extrapole uma unica particao sintetica para risco clinico.

### Duvidas que esta aula responde

- **Um dashboard substitui a analise estatistica?** Nao. Ele organiza evidencias; nao cria significancia.
- **Uma porcentagem de economia e universal?** Nao. Depende de hardware, volume, custos e frequencia de uso.
- **Por que mostrar baseline e campeao juntos?** Para dar escala e evitar interpretar um numero isolado.
- **O que fazer quando os indicadores discordam?** Declarar o trade-off e voltar a metrica prioritaria do dominio.

### Regra de explicacao Feynman

Explique o dashboard como o painel de um carro: ele mostra velocidade, combustivel e alertas simultaneamente. Uma unica luz verde nao autoriza dizer que a viagem inteira esta segura.

### Ler o dashboard em quatro perguntas

Cada painel responde a uma pergunta diferente:

```text
reducao dimensional  -> quantos atributos deixaram de ser usados?
tempo de treino       -> quanto custa reconstruir o modelo?
latencia              -> quanto levou a predicao do lote de teste?
F1/recall             -> quais foram as metricas no holdout?
```

Um resultado deve ser interpretado como um conjunto: a proposta reduziu 75% dos atributos (recuperando 10/10 informativos e 0/20 ruídos), elevou o F1 tanto sem Optuna (0,8861, com redução de 40,2% no tempo de treino) quanto com Optuna (0,8872 no holdout e 0,8864 ± 0,0122 na validação cruzada de 5 folds). Nao foram medidos volume de producao ou custos de exames.

### Como separar observacao de interpretacao

Observacao: “a proposta usou 10 atributos e obteve F1 de 0,8861 (sem Optuna) e 0,8872 (com Optuna), contra 0,8856 do baseline”. Interpretacao: “nesta particao e na validacao cruzada de 5 folds, a reducao de 75% recuperou os 10 atributos informativos sintéticos e superou o baseline em F1”. Conclusao forte demais: “os outros 30 exames sao clinicamente inuteis”. O dashboard deve ajudar a enxergar essa diferenca.

### Duvidas frequentes

- **Uma barra maior significa importancia causal?** Nao; significa o valor da metrica representada.
- **Tempo menor em uma rodada prova economia permanente?** Nao; repita em varias execucoes e considere o custo total.
- **F1 igual significa risco clinico igual?** Nao necessariamente; a composicao entre precision, recall, FP e FN pode mudar.
- **Como evitar uma escala enganosa?** Exiba unidades, baseline, numero de casos e intervalo ou variacao quando disponivel.

### Ponte para a comunicacao cientifica

O painel resume a evidencia; a Camada 15 ensina como relatar essa evidencia em IMRaD, separando resultado observado de discussao.

## Cultura, História e Referências

Dashboard executivo tem uma historia propria: a visualizacao estatistica busca reduzir carga cognitiva sem reduzir a verdade. Edward Tufte popularizou principios de clareza, comparacao e economia visual em [The Visual Display of Quantitative Information](https://www.edwardtufte.com/book/the-visual-display-of-quantitative-information/). O [dashboard_final_comparativo.png](../assets/dashboard_final_comparativo.png) deve ser usado para perguntar se a forma da imagem ajuda a conferir os numeros.

Uma boa visualizacao nao “vende” o modelo; ela torna diferencas visiveis e contestaveis. Sempre acompanhe cada painel de unidade, baseline, denominador e contexto. Evite cortar eixos para fabricar dramaticidade ou usar cor sem legenda.

**Pergunta cultural:** quem e o leitor do dashboard? Um pesquisador precisa de protocolo e limites; diferentes leitores podem ter perguntas distintas. A figura deve identificar o que foi medido.

## Recursos de Mídia (Visual e Áudio)

- **Visual local:** [Dashboard final comparativo](../assets/dashboard_final_comparativo.png), com dimensão, treino, latência e F1.
- **Imagem incorporada:**

![Dashboard executivo comparativo](../assets/dashboard_final_comparativo.png)
- **Referencia visual:** principios de [The Visual Display of Quantitative Information](https://www.edwardtufte.com/book/the-visual-display-of-quantitative-information/).
- **Áudio sugerido:** um painel para pesquisador, gestor e equipe clínica, mostrando que cada público pergunta algo diferente.
- **Imagem mental:** quatro paineis formando uma decisao, nao quatro graficos isolados.

## 📊 Elementos de Comunidade e Status

- **Status:** `Resultado comunicado` quando observação, interpretação e recomendação estiverem separadas.
- **Debate:** “Qual painel mudaria sua decisao de implantacao e por que?”
- **Papel rotativo:** designer, estatistico, gestor e profissional de saude.

## 💡 Engajamento e Conhecimento

- **Atividade:** uma leitura de 60 segundos do dashboard é feita sem adjetivos.
- **Produto da aula:** legenda executiva com baseline, unidade, denominador, ganho e limite.
- **Conexao profissional:** transformar numeros em decisao sem esconder FN, custo ou variacao.

## Mapa da aula

1. [Subcamada 14.1: O conceito na vida real](#subcamada-141-o-conceito-na-vida-real)
2. [Subcamada 14.2: Desenhando o conceito](#subcamada-142-desenhando-o-conceito)
3. [Subcamada 14.3: Desmistificando a teoria e a notacao formal](#subcamada-143-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 14.4: Laboratorio ludico no Colab](#subcamada-144-laboratorio-ludico-no-colab)
5. [Subcamada 14.5: O momento serio da nossa aplicacao](#subcamada-145-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 14.6: Checkpoint de autonomia e fixacao ativa](#subcamada-146-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 14.1: O conceito na vida real

### A analogia dos quatro mostradores na cabine de voo

Em uma aeronave comercial durante o procedimento de descida e pouso, o comandante nao analisa linhas de codigo dos computadores de bordo; ele monitora um painel com quatro instrumentos essenciais:
1. **Peso total da aeronave:** aeronaves mais leves demandam pistas menores e consomem menos combustivel por milha nautica.
2. **Temperatura e potencia dos motores:** reflete a sobrecarga termica do sistema durante o voo.
3. **Tempo de resposta dos atuadores mecanicos:** a agilidade em milissegundos com que as asas respondem aos comandos do manche em situacoes de turbulencia severa.
4. **Horizonte artificial e altitude:** a estabilidade de navegacao que garante a integridade fisica de passageiros e tripulacao.

Na apresentacao dos resultados, compare dimensao, desempenho e tempos medidos sob o mesmo protocolo. Os dados sao sinteticos: a reducao de atributos nao mede exames coletados, custo hospitalar ou eficacia diagnostica.

### O debate metodologico: custo medido e escopo das conclusoes

Uma objecao frequente apresentada por avaliadores diz respeito ao esforco computacional dispendido nas fases de explicabilidade: *"O calculo analitico de valores SHAP, a avaliacao da curva de ablacao e a busca bayesiana com Optuna nao consomem tempo relevante de CPU?"*

Os tempos de selecao, treino e predicao medem etapas diferentes e devem ser reportados separadamente. Esta execucao nao mediu implantacao, custos de producao ou efeitos em pacientes.

| Metrica | Baseline (40 atributos) | Proposta Puro (10 atrib.) | Proposta + Optuna (10 atrib.) |
| :--- | :--- | :--- | :--- |
| Atributos | 40 | 10 | 10 |
| Informativos retidos (sinteticos) | 10/10 | 10/10 | 10/10 |
| Redundantes / Ruidos retidos | 10/10 red., 20/20 ruidos | 0/10 red., 0/20 ruidos | 0/10 red., 0/20 ruidos |
| F1 no holdout | 0,8856 | 0,8861 | 0,8872 |
| F1 CV (5-Fold) | 0,8649 ± 0,0159 | 0,8873 ± 0,0213 | 0,8864 ± 0,0122 |
| Acuracia no holdout | 0,9060 | 0,9080 | 0,9100 |
| ROC-AUC no holdout | 0,9655 | 0,9686 | 0,9734 |
| Treino (ms) | 878,8 | 525,7 | 1075,5 |
| Predicao do lote de teste (ms) | 48,70 | 40,39 | 78,36 |
| Selecao de atributos (s) | 0,00 | 11,04 | 79,30 |

Nesta execucao, a reducao de atributos foi de 75% (recuperando 10/10 atributos informativos sem corte Top-K forçado) e houve ganho de F1 tanto no modelo puro sem Optuna (+0,0005 no holdout; +0,0224 na CV 5-Fold) quanto no modelo com Optuna (+0,0016 no holdout; +0,0215 na CV 5-Fold), superando com folga o criterio de H1. No modelo puro (*ceteris paribus*), o tempo de treino caiu 40,2% (878,8 ms para 525,7 ms); a selecao OOF levou 11,04 s contra 4,55 s do RFE.

---

## Subcamada 14.2: Desenhando o conceito

O diagrama abaixo sintetiza a disposicao dos quatro quadrantes do dashboard executivo gerado pelo pipeline:

```text
==================================================================================
           PAINEL EXECUTIVO: BASELINE (40) VS. XAI PURO (10) VS. + OPTUNA (10)
==================================================================================
 QUADRANTE 1: ATRIBUTOS SINTETICOS          QUADRANTE 2: TEMPO DE TREINO (ms)
   40 | [==========] 40 (Baseline)           1100 | [==========] 1075.5 (+Optuna)
   20 | [===] 10 (XAI Puro / +Optuna)         880 | [========] 878.8 (Baseline)
    0 +--------------------------             525 | [=====] 525.7 (XAI Puro -40.2%)
----------------------------------------------------------------------------------
 QUADRANTE 3: PREDICAO DO HOLDOUT (ms)     QUADRANTE 4: F1 NO HOLDOUT
   80 | [==========] 78.36 (+Optuna)          1.0 | [========] 0.8856 (Baseline)
   50 | [======] 48.70 (Baseline)             0.9 | [========] 0.8861 (XAI Puro)
   40 | [=====] 40.39 (XAI Puro)              0.8 | [========] 0.8872 (+Optuna)
==================================================================================
```

A comparacao nao estima retorno operacional nem desempenho futuro:

```text
[ ETAPA DE SELECAO ]                       [ MODELO NO HOLDOUT E 5-FOLD CV ]
Tempo medido nesta execucao:               Resultados observados:
• shap-select OOF Puro: 11.04 s            • F1 Holdout: 0.8861 (Puro) / 0.8872 (+Optuna)
• shap-select OOF + Optuna: 79.30 s        • F1 CV 5-Fold: 0.8873 (Puro) / 0.8864 (+Optuna)
• RFE: 4.55 s                              • Baseline (40): F1 Holdout 0.8856 / CV 0.8649
      │                                          │
      ▼                                          ▼
[ CUSTO DE SELECAO OFFLINE ]               [ BENCHMARK SINTETICO CONTROLADO ]
```

| Quadrante | Metrica Apresentada | Leitura Tecnica | Significado Institucional |
| :--- | :--- | :--- | :--- |
| **Q1 (Atributos)** | $M_{\text{inicial}} \to M_{\text{final}}$ | 40 para 10 atributos (10/10 informativos) | Reducao dimensional de 75% com 0 ruídos |
| **Q2 (Treinamento)** | Tempo de ajuste (ms) | 878,8 ms para 525,7 ms (Puro) | Reducao de 40,2% sob o mesmo estimador padrao |
| **Q3 (Predicao)** | Tempo para prever o lote de teste (ms) | 48,70 ms para 40,39 ms (Puro) | Aceleracao no estimador padrao; maior no Optuna por ter mais arvores |
| **Q4 (Desempenho)** | $F_1$-Score e Acuracia | F1 0,8856 para 0,8861 (Puro) e 0,8872 (+Optuna) | Ganho preditivo no Holdout e na CV de 5 Folds |

---

## Subcamada 14.3: Desmistificando a teoria e a notacao formal

### Metricas de eficiencia e reducao dimensional

A taxa de reducao volumetrica no espaco de entrada e expressa como:

$$\Delta_p = \left( 1 - \frac{p_{\text{reduzido}}}{p_{\text{bruto}}} \right) \times 100\%$$

Em nosso experimento oficial: $\Delta_p = (1 - 10/40) \times 100\% = 75.0\%$.

### Aceleracao computacional (*Speedup*)

A razao de ganho de velocidade (speedup) no treinamento e na inferencia e formalizada por:

$$S_{\text{train}} = \frac{T_{\text{train}}(\text{Baseline})}{T_{\text{train}}(\text{Proposta})}, \quad S_{\text{infer}} = \frac{\tau_{\text{infer}}(\text{Baseline})}{\tau_{\text{infer}}(\text{Proposta})}$$

Nesta execucao, $S_{\text{train}} \approx 1.48$, enquanto $S_{\text{infer}} \approx 0.93$ para o lote de teste; portanto, o treino foi mais rapido e a predicao do lote nao.

### Gap de generalizacao (Monitoramento de Overfitting)

A reducao do gap entre o desempenho de treino e teste formaliza o combate ao sobreajuste:

$$\text{Gap}_{\text{overfit}} = \text{Acuracia}_{\text{treino}} - \text{Acuracia}_{\text{teste}}$$

O pipeline nao exporta uma metrica de gap de generalizacao nesta tabela. Nao se deve inferir reducao de overfitting apenas a partir da quantidade de atributos ou da diferenca de F1 no holdout.

| Simbolo | Significado Formal | Leitura no Projeto |
| :--- | :--- | :--- |
| $\Delta_p$ | Taxa percentual de reducao de dimensionalidade | Eliminacao de 75% dos atributos sinteticos (10/10 informativos retidos) |
| $S_{\text{train}}$ | Fator de aceleracao no tempo de treinamento (modelo puro) | Aproximadamente 1,67 (redução de 40,2%: 878,8 ms → 525,7 ms) |
| $S_{\text{infer}}$ | Fator entre tempos de predicao do lote (modelo puro) | Aproximadamente 1,21 (48,70 ms → 40,39 ms) |
| $\text{Gap}_{\text{overfit}}$ | Diferenca entre desempenho de treino e teste | Nao calculada nem exportada pelo pipeline |
| $F_1$ | Media harmonica entre precisao e recall | 0,8856 no baseline, 0,8861 no XAI Puro e 0,8872 no XAI + Optuna |

### A ordem correta evita distorcoes

Todas as metricas apresentadas nos quadrantes de treinamento e desempenho devem ser apuradas rigorosamente sob o mesmo particionamento inicial de dados. Modificar a divisao de treino/teste entre os modelos invalidaria as comparacoes de speedup e $F_1$.

---

## Subcamada 14.4: Laboratorio ludico no Colab

Execute o bloco abaixo no Google Colab para renderizar os quatro quadrantes do dashboard executivo:

```python
# =============================================================================
# CAMADA 14: LABORATORIO LUDICO DO DASHBOARD EXECUTIVO
# Demonstracao: Plotagem Comparativa dos 4 Quadrantes Estrategicos
# =============================================================================
import matplotlib.pyplot as plt
import pandas as pd

# Os valores sao lidos da execucao mais recente do pipeline.
resultados = pd.read_csv("assets/tabela_comparativa_completa.csv")
baseline = resultados.loc[resultados["Metodo"] == "Baseline Completo (40 Atributos)"].iloc[0]
proposta_pura = resultados.loc[
    resultados["Metodo"] == "XAI Proposto (shap-select OOF Puro)"
].iloc[0]
proposta = resultados.loc[
    resultados["Metodo"] == "XAI Proposto (shap-select OOF + Optuna)"
].iloc[0]
modelos = ["Baseline (40)", "XAI Puro (10)", "XAI + Optuna (10)"]
paleta = ["#7f8c8d", "#1f77b4", "#27ae60"]
reducao = (1 - proposta["N_Atributos"] / baseline["N_Atributos"]) * 100

# 2 linhas x 2 colunas formam os quatro quadrantes do dashboard.
fig, eixos = plt.subplots(2, 2, figsize=(12, 7.5))

# Quadrante 1: atributos sinteticos selecionados.
eixos[0, 0].bar(modelos, [baseline["N_Atributos"], proposta_pura["N_Atributos"], proposta["N_Atributos"]], color=paleta, width=0.45, edgecolor="black")
eixos[0, 0].set_title("1. Atributos Sinteticos", fontsize=10, fontweight="bold")
eixos[0, 0].set_ylabel("Quantidade de Atributos", fontsize=9)
eixos[0, 0].text(2, proposta["N_Atributos"] + 2, f"-{reducao:.1f}%", ha="center", fontsize=10, fontweight="bold")
eixos[0, 0].grid(axis="y", linestyle=":", alpha=0.6)

# Quadrante 2: milissegundos gastos para ajustar a floresta.
eixos[0, 1].bar(modelos, [baseline["Tempo_Treino_ms"], proposta_pura["Tempo_Treino_ms"], proposta["Tempo_Treino_ms"]], color=paleta, width=0.45, edgecolor="black")
eixos[0, 1].set_title("2. Tempo de Treinamento (ms)", fontsize=10, fontweight="bold")
eixos[0, 1].set_ylabel("Milissegundos", fontsize=9)
eixos[0, 1].grid(axis="y", linestyle=":", alpha=0.6)

# Quadrante 3: tempo de predicao do lote de teste.
eixos[1, 0].bar(modelos, [baseline["Latencia_Inf_ms"], proposta_pura["Latencia_Inf_ms"], proposta["Latencia_Inf_ms"]], color=paleta, width=0.45, edgecolor="black")
eixos[1, 0].set_title("3. Predicao do Lote de Teste (ms)", fontsize=10, fontweight="bold")
eixos[1, 0].set_ylabel("Milissegundos", fontsize=9)
eixos[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

# Quadrante 4: F1 no holdout sintetico.
eixos[1, 1].bar(modelos, [baseline["F1_Score"], proposta_pura["F1_Score"], proposta["F1_Score"]], color=paleta, width=0.45, edgecolor="black")
eixos[1, 1].set_title("4. F1 no Holdout Sintetico", fontsize=10, fontweight="bold")
eixos[1, 1].set_ylabel("F1-Score no Teste Cego", fontsize=9)
eixos[1, 1].set_ylim(0.80, 0.92)
eixos[1, 1].grid(axis="y", linestyle=":", alpha=0.6)

plt.suptitle("DASHBOARD EXECUTIVO COMPARATIVO: XAI APLICADA A REDUCAO DE DADOS", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
```

> **O que voce deve notar no grafico gerado:**
> 1. A proposta usa 75% menos atributos (recuperando 10/10 informativos) e, na comparação *ceteris paribus* (XAI Puro), reduz o tempo de treino em 40,2% e eleva o F1 para 0,8861.
> 2. Com Optuna, o F1 sobe para 0,8872 (e ROC-AUC para 0,9734); estes resultados descrevem o benchmark sintético e não demonstram segurança ou benefício clínico.

**Mini-experimento:** use o CSV para comparar a selecao da proposta com RFE. O que a diferenca de pureza (10/10 vs. 7/10 informativos) e de tempo de selecao (11,04 s vs. 4,55 s) permite concluir sobre H2?

---

## Subcamada 14.5: O momento serio da nossa aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

No fechamento do projeto de reducao de dimensionalidade, emitimos o quadro consolidado oficial de KPIs cruzando todas as dimensoes operacionais, estatisticas e preditivas.

```python
# =============================================================================
# APLICACAO AO BENCHMARK: QUADRO COMPARATIVO DA EXECUCAO ATUAL
# Os valores sao lidos do CSV gerado pelo pipeline.
# =============================================================================
import pandas as pd

resultados = pd.read_csv("assets/tabela_comparativa_completa.csv")
baseline = resultados.loc[resultados["Metodo"] == "Baseline Completo (40 Atributos)"].iloc[0]
proposta_pura = resultados.loc[
    resultados["Metodo"] == "XAI Proposto (shap-select OOF Puro)"
].iloc[0]
proposta = resultados.loc[
    resultados["Metodo"] == "XAI Proposto (shap-select OOF + Optuna)"
].iloc[0]
quadro_resultados = pd.DataFrame([
    {"Metrica": "Atributos sinteticos", "Baseline (40)": baseline["N_Atributos"], "XAI Puro (10)": proposta_pura["N_Atributos"], "XAI + Optuna (10)": proposta["N_Atributos"]},
    {"Metrica": "Informativos retidos (/10)", "Baseline (40)": baseline["Biomarcadores_Retidos (max 10)"], "XAI Puro (10)": proposta_pura["Biomarcadores_Retidos (max 10)"], "XAI + Optuna (10)": proposta["Biomarcadores_Retidos (max 10)"]},
    {"Metrica": "F1 no holdout", "Baseline (40)": baseline["F1_Score"], "XAI Puro (10)": proposta_pura["F1_Score"], "XAI + Optuna (10)": proposta["F1_Score"]},
    {"Metrica": "F1 CV (5-Fold Media)", "Baseline (40)": baseline["F1_CV_Media"], "XAI Puro (10)": proposta_pura["F1_CV_Media"], "XAI + Optuna (10)": proposta["F1_CV_Media"]},
    {"Metrica": "Acuracia no holdout", "Baseline (40)": baseline["Acuracia"], "XAI Puro (10)": proposta_pura["Acuracia"], "XAI + Optuna (10)": proposta["Acuracia"]},
    {"Metrica": "ROC-AUC no holdout", "Baseline (40)": baseline["ROC_AUC"], "XAI Puro (10)": proposta_pura["ROC_AUC"], "XAI + Optuna (10)": proposta["ROC_AUC"]},
    {"Metrica": "Treino (ms)", "Baseline (40)": baseline["Tempo_Treino_ms"], "XAI Puro (10)": proposta_pura["Tempo_Treino_ms"], "XAI + Optuna (10)": proposta["Tempo_Treino_ms"]},
    {"Metrica": "Predicao do lote de teste (ms)", "Baseline (40)": baseline["Latencia_Inf_ms"], "XAI Puro (10)": proposta_pura["Latencia_Inf_ms"], "XAI + Optuna (10)": proposta["Latencia_Inf_ms"]},
    {"Metrica": "Selecao (s)", "Baseline (40)": baseline["Tempo_Selecao_s"], "XAI Puro (10)": proposta_pura["Tempo_Selecao_s"], "XAI + Optuna (10)": proposta["Tempo_Selecao_s"]},
])

print("=" * 88)
print("QUADRO OFICIAL CONSOLIDADO DE KPIS: BASELINE VS. XAI PURO VS. XAI + OPTUNA")
print("=" * 88)
print(quadro_resultados.to_string(index=False))
print("=" * 88)
```

### Tabela oficial de KPIs

> Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Como e calculado | Pergunta operacional |
| :--- | :--- | :--- |
| **Reducao de atributos (%)** | $(1 - p_{\text{proposta}} / p_{\text{baseline}}) \times 100$ | Qual foi a reducao neste benchmark? |
| **F1 no holdout** | Media harmonica de precisao e recall | A queda ficou dentro do criterio predefinido de H1? |
| **Tempo de selecao** | Segundos medidos no pipeline | Como se compara a selecao com RFE? |
| **Fidelidade LIME** | $R^2$ do substituto em uma observacao | Qual foi o ajuste local e qual o limite dessa observacao? |

### Interpretacao do benchmark sintetico

Os KPIs descrevem somente esta execucao:

1. **Desempenho:** o F1 subiu de 0,8856 para 0,8861 (sem Optuna) e 0,8872 (com Optuna), além de superar na validação cruzada de 5 folds (0,8864 ± 0,0122 vs. 0,8649 ± 0,0159).
2. **Custo medido:** o treino no modelo puro foi 40,2% mais rápido (525,7 ms vs. 878,8 ms), enquanto a seleção OOF levou mais tempo offline que o RFE (11,04 s vs. 4,55 s).
3. **Limites:** nao foram medidos custos de exames, uso clinico, escalabilidade ou desempenho em dados externos.

---

## Subcamada 14.6: Checkpoint de autonomia e fixacao ativa

Explique sem consultar o texto e depois confira sua resposta:

1. **Como comparar a queda de F1 observada com o limite predefinido de H1?**
2. **Qual e a resposta metodologica para questionamentos sobre o custo computacional consumido nas fases de explicabilidade e busca bayesiana?**
3. **O que permite concluir a rejeicao de ruido sintetico neste benchmark?**
4. **Como interpretar o tempo de predicao medido para o lote de teste?**
5. **Quais afirmacoes exigiriam repeticoes, dados externos ou avaliacao humana?**
6. **Como os quatro quadrantes do dashboard resumem os resultados sem sugerir uso clinico?**

### Mini-desafio pratico

Use o CSV atualizado para comparar o tempo de selecao da proposta com RFE e discuta o que essa medicao permite concluir sobre H2.

**Pergunta reflexiva:** que evidencias adicionais seriam necessarias para extrapolar estes resultados para dados reais?
