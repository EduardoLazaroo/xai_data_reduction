# Camada 14: Interpretacao de Resultados e Dashboard Executivo

**Trilha:** XAI Aplicada a Reducao de Dados em Machine Learning  
**Aplicacao:** classificacao binaria de saude ('0 = Saudavel', '1 = Patologia')  
**Codigo de referencia:** [pipeline_completo.py](../pipeline_completo.py), funcoes com dashboard e comparativo de KPIs

> **Objetivo da aula:** Analisar a sintese executiva e cientifica dos resultados experimentais, interpretando os quatro quadrantes do dashboard comparativo (reducao dimensional, tempo de treinamento, latencia de inferencia e retencao de F1-score) e formalizando o argumento de retorno de investimento entre o custo analitico de P&D e os ganhos operacionais permanentes em producao.

## Campo Didatico: Um Dashboard e um Argumento

Leia cada painel em duas camadas: primeiro descreva literalmente o que o grafico mostra; depois explique o que isso significa para a operacao. O roteiro e **dimensao, treino, latencia, qualidade, erro clinico e custo**. Nunca pule diretamente do grafico para a recomendacao.

```text
grafico -> numero -> comparacao baseline/campeao -> impacto operacional -> limite da conclusao
```

Observe se a reducao de atributos preserva F1 e recall, e se o ganho de tempo e maior que a variacao de medicao. O erro comum e usar escala visual para sugerir uma melhora que os numeros nao sustentam. A ponte para a Camada 15 e transformar essa leitura em metodologia e hipoteses publicaveis.

---

### Roteiro de dominio

Leia o dashboard em ordem: **o que mudou, quanto mudou, com que variabilidade, a que custo e com qual risco clinico**. Para cada painel escreva uma frase factual e uma frase interpretativa separadas. Depois procure contradicoes: menos atributos pode vir acompanhado de mais `FN`, e menor tempo nao compensa uma queda clinica inaceitavel.

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
latencia              -> quanto custa avaliar um paciente?
F1/recall             -> a qualidade diagnostica foi preservada?
```

Um resultado so e operacionalmente bom quando as respostas formam um conjunto coerente. Reduzir de 40 para 10 atributos e perder recall pode ser inaceitavel, mesmo que o treino fique rapido. Manter F1 e reduzir latencia pode ser valioso, mas a conclusao depende do volume de pacientes e do custo real dos exames.

### Como separar observacao de interpretacao

Observacao: “o campeao usou 10 atributos e teve F1 de 0,86”. Interpretacao: “isso sugere que parte das colunas nao era necessaria para a tarefa nesta amostra”. Conclusao forte demais: “os outros 30 exames sao clinicamente inuteis”. O dashboard deve ajudar a enxergar essa diferenca.

### Duvidas frequentes

- **Uma barra maior significa importancia causal?** Nao; significa o valor da metrica representada.
- **Tempo menor em uma rodada prova economia permanente?** Nao; repita em varias execucoes e considere o custo total.
- **F1 igual significa risco clinico igual?** Nao necessariamente; a composicao entre precision, recall, FP e FN pode mudar.
- **Como evitar uma escala enganosa?** Exiba unidades, baseline, numero de casos e intervalo ou variacao quando disponivel.

### Ponte para a comunicacao cientifica

O painel resume a evidencia; a Camada 15 ensina como relatar essa evidencia em IMRaD, separando resultado observado de discussao.

## Cultura, Historia e Referencias

Dashboard executivo tem uma historia propria: a visualizacao estatistica busca reduzir carga cognitiva sem reduzir a verdade. Edward Tufte popularizou principios de clareza, comparacao e economia visual em [The Visual Display of Quantitative Information](https://www.edwardtufte.com/book/the-visual-display-of-quantitative-information/). O [dashboard_final_comparativo.png](../assets/dashboard_final_comparativo.png) deve ser usado para perguntar se a forma da imagem ajuda a conferir os numeros.

Uma boa visualizacao nao “vende” o modelo; ela torna diferencas visiveis e contestaveis. Sempre acompanhe cada painel de unidade, baseline, denominador e contexto. Evite cortar eixos para fabricar dramaticidade ou usar cor sem legenda.

**Pergunta cultural:** quem e o leitor do dashboard? Um pesquisador quer incerteza e protocolo; um gestor quer decisao e custo; uma equipe clinica quer risco por tipo de erro. A mesma figura precisa declarar para quem foi desenhada.

## Recursos de Mídia (Visual e Áudio)

- **Visual local:** [Dashboard final comparativo](../assets/dashboard_final_comparativo.png), com dimensao, treino, latencia e F1.
- **Referencia visual:** principios de [The Visual Display of Quantitative Information](https://www.edwardtufte.com/book/the-visual-display-of-quantitative-information/).
- **Audio de abertura:** descrever um painel para pesquisador, gestor e equipe clinica, mostrando que cada publico pergunta algo diferente.
- **Imagem mental:** quatro paineis formando uma decisao, nao quatro graficos isolados.

## 📊 Elementos de Comunidade e Status

- **Status:** `Resultado comunicado` quando observacao, interpretacao e recomendacao estiverem separadas.
- **Debate:** “Qual painel mudaria sua decisao de implantacao e por que?”
- **Papel rotativo:** designer, estatistico, gestor e profissional de saude.

## 💡 Engajamento e Conhecimento

- **Atividade:** cada grupo faz uma leitura de 60 segundos do dashboard sem adjetivos.
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

Na apresentacao de uma pesquisa em Inteligencia Artificial para comites medicos, gestores hospitalares ou bancas examinadoras, o raciocinio e idêntico: a complexidade matematica dos algoritmos deve convergir para quatro dimensoes executivas comparaveis:
- Volume dimensional de entrada (quantos exames laboratoriais precisam ser coletados).
- Custo computacional de re-treinamento (tempo de processamento em servidores).
- Latencia de inferencia por paciente (tempo para emissao do diagnostico no ponto de atendimento).
- Eficacia diagnostica ($F_1$-score e acuracia sobre pacientes do teste cego).

### O debate metodologico: custo de P&D contra economia em producao

Uma objecao frequente apresentada por avaliadores diz respeito ao esforco computacional dispendido nas fases de explicabilidade: *"O calculo analitico de valores SHAP, a avaliacao da curva de ablacao e a busca bayesiana com Optuna nao consomem tempo relevante de CPU?"*

A resposta metodologica apoia-se na diferenciacao entre fases do ciclo de vida:
- **Fase de Pesquisa e Desenvolvimento (P&D):** a extracao analitica e a busca bayesiana ocorrem em ambiente laboratorial controlado. Trata-se de um custo computacional pago uma unica vez para descobrir quais sao as variaveis causais verdadeiras e qual e a configuracao hiperparametrica otima.
- **Fase de Producao Hospitalar:** uma vez consolidado, o modelo enxuto (10 atributos) e implantado na rotina clinica, onde processara milhares de pacientes ao longo de anos. A cada ciclo semanal de re-treinamento e a cada prontuario submetido a inferencia, a instituicao acumula ganhos permanentes de tempo e infraestrutura.

**A grande sacada:** o custo pontual de compreensao analitica no desenvolvimento financia a reducao perene de despesas operacionais e laboratoriais em ambiente de producao.

| Dimensao | Modelo Baseline (Forca Bruta) | Modelo Reduzido Campeao (XAI + Optuna) | Ganho Operacional |
| :--- | :--- | :--- | :--- |
| **Coleta de Atributos** | 40 exames laboratoriais completos | 10 biomarcadores de elite validados | Reducao de 75% na carga de coleta |
| **Tempo de Treinamento** | ~600 ms por floresta | ~185 ms por floresta | Speedup de ~3.2x em retreinos periodicos |
| **Latencia de Inferencia** | ~30.8 microssegundos por paciente | ~12.4 microssegundos por paciente | Reducao de ~60% na latencia individual |
| **Qualidade ($F_1$-Score)** | 0.8373 no teste cego | 0.8610 no teste cego | Preservacao e leve incremento preditivo |

---

## Subcamada 14.2: Desenhando o conceito

O diagrama abaixo sintetiza a disposicao dos quatro quadrantes do dashboard executivo gerado pelo pipeline:

```text
==================================================================================
                 PAINEL EXECUTIVO: CONFRONTO BASELINE VS. MODELO XAI
==================================================================================
 QUADRANTE 1: DIMENSAO DOS DADOS           QUADRANTE 2: CUSTO DE SERVIDOR
   40 | [==========] 40 exames               800 | [==========] 600 ms
   30 |                                      600 | 
   20 |                                      400 | 
   10 | [===] 10 exames (-75.0%)             200 | [===] 185 ms (-69.2%)
    0 +--------------------------              0 +--------------------------
        Baseline     Campeao XAI                   Baseline     Campeao XAI
----------------------------------------------------------------------------------
 QUADRANTE 3: LATENCIA DE INFERENCIA       QUADRANTE 4: QUALIDADE DIAGNOSTICA
   40 | [==========] 30.8 us                 1.0 | [=======] 0.8373  [=======] 0.8610
   30 |                                      0.8 |
   20 |                                      0.6 |
   10 | [===] 12.4 us (-59.7%)               0.4 |
    0 +--------------------------            0.0 +--------------------------
        Baseline     Campeao XAI                   Baseline (40)  Campeao (10)
==================================================================================
```

A dinamica temporal entre investimento computacional e retorno operacional e mapeada a seguir:

```text
[ FASE DE P&D (LABORATORIO) ]              [ FASE DE PRODUCAO (HOSPITAL) ]
Custo unico:                               Economia continua:
• Calculo SHAP TreeExplainer               • 75% menos reagentes de laboratorio
• Filtragem shap-select                    • Retreinos 3x mais ageis no cluster
• Optuna Bayesian Tuning                   • Inferencia ultra-rapida (12 us/paciente)
      │                                          │
      ▼                                          ▼
[ INVESTIMENTO PONTUAL: ~15-20 s ]         [ ECONOMIA PERMANENTE ESCALAVEL ]
```

| Quadrante | Metrica Apresentada | Leitura Tecnica | Significado Institucional |
| :--- | :--- | :--- | :--- |
| **Q1 (Atributos)** | $M_{\text{inicial}} \to M_{\text{final}}$ | Descarte de 30 colunas de ruido e redundancia | Menos desconforto ao paciente e menor custo de coleta |
| **Q2 (Treinamento)** | Latencia de ajuste da floresta (ms) | Menos amostras e menor profundidade de divisao | Menor consumo de energia e servidores na nuvem |
| **Q3 (Inferencia)** | Tempo de execucao de `.predict()` ($\mu$s) | Menos comparacoes condicionais por arvore | Viabilidade para Edge AI e triagens de emergencia |
| **Q4 (Desempenho)** | $F_1$-Score e Acuracia no teste | Ausencia de colapso preditivo; fechamento de gap | Seguranca clinica absoluta no diagnostico emitido |

---

## Subcamada 14.3: Desmistificando a teoria e a notacao formal

### Metricas de eficiencia e reducao dimensional

A taxa de reducao volumetrica no espaco de entrada e expressa como:

$$\Delta_p = \left( 1 - \frac{p_{\text{reduzido}}}{p_{\text{bruto}}} \right) \times 100\%$$

Em nosso experimento oficial: $\Delta_p = (1 - 10/40) \times 100\% = 75.0\%$.

### Aceleracao computacional (*Speedup*)

A razao de ganho de velocidade (speedup) no treinamento e na inferencia e formalizada por:

$$S_{\text{train}} = \frac{T_{\text{train}}(\text{Baseline})}{T_{\text{train}}(\text{Campeao})}, \quad S_{\text{infer}} = \frac{\tau_{\text{infer}}(\text{Baseline})}{\tau_{\text{infer}}(\text{Campeao})}$$

Quando $S > 1.0$, o sistema apresenta ganho relativo de eficiencia proporcional ao inverso da complexidade dimensional restante.

### Gap de generalizacao (Monitoramento de Overfitting)

A reducao do gap entre o desempenho de treino e teste formaliza o combate ao sobreajuste:

$$\text{Gap}_{\text{overfit}} = \text{Acuracia}_{\text{treino}} - \text{Acuracia}_{\text{teste}}$$

No modelo com 40 atributos, $\text{Gap}_{\text{bruto}} \approx 12.2\%$; no modelo compacto de 10 atributos regularizado via Optuna, $\text{Gap}_{\text{enxuto}} \approx 6.1\%$. O estreitamento dessa diferenca comprova que o modelo deixou de memorizar combinacoes aleatorias presentes nas colunas de ruido.

| Simbolo | Significado Formal | Leitura no Projeto |
| :--- | :--- | :--- |
| $\Delta_p$ | Taxa percentual de reducao de dimensionalidade | Eliminacao de 75% dos exames coletados |
| $S_{\text{train}}$ | Fator de aceleracao no tempo de treinamento | Re-treino ~3.2 vezes mais rapido no servidor |
| $S_{\text{infer}}$ | Fator de aceleracao na inferencia individual | Resposta ~2.5 vezes mais rapida por paciente |
| $\text{Gap}_{\text{overfit}}$ | Diferenca entre acuracia de treino e teste cego | Queda de 12.2% para 6.1% (reducao pela metade) |
| $F_1$ | Media harmonica entre precisao e sensibilidade | Preservado em patamar superior a 0.86 |

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

modelos = ["Baseline (40 Atrib.)", "Campeao XAI (10 Atrib.)"]
paleta = ["#7f8c8d", "#2980b9"]

fig, eixos = plt.subplots(2, 2, figsize=(11, 7.5))

# Quadrante 1: Dimensao de Atributos
eixos[0, 0].bar(modelos, [40, 10], color=paleta, width=0.45, edgecolor="black")
eixos[0, 0].set_title("1. Atributos Coletados (Dimensionalidade)", fontsize=10, fontweight="bold")
eixos[0, 0].set_ylabel("Quantidade de Exames", fontsize=9)
eixos[0, 0].text(1, 12, "-75.0%", ha="center", fontsize=10, fontweight="bold", color="#c0392b")
eixos[0, 0].grid(axis="y", linestyle=":", alpha=0.6)

# Quadrante 2: Tempo de Treinamento
eixos[0, 1].bar(modelos, [600, 185], color=paleta, width=0.45, edgecolor="black")
eixos[0, 1].set_title("2. Tempo de Ajuste em Servidor (ms)", fontsize=10, fontweight="bold")
eixos[0, 1].set_ylabel("Milissegundos", fontsize=9)
eixos[0, 1].text(1, 210, "-69.2%", ha="center", fontsize=10, fontweight="bold", color="#c0392b")
eixos[0, 1].grid(axis="y", linestyle=":", alpha=0.6)

# Quadrante 3: Latencia por Paciente
eixos[1, 0].bar(modelos, [30.8, 12.4], color=paleta, width=0.45, edgecolor="black")
eixos[1, 0].set_title("3. Latencia de Inferencia (Microssegundos)", fontsize=10, fontweight="bold")
eixos[1, 0].set_ylabel("Microssegundos (us)", fontsize=9)
eixos[1, 0].text(1, 14.5, "-59.7%", ha="center", fontsize=10, fontweight="bold", color="#c0392b")
eixos[1, 0].grid(axis="y", linestyle=":", alpha=0.6)

# Quadrante 4: Rendimento Clinico (F1-Score)
eixos[1, 1].bar(modelos, [0.8373, 0.8610], color=paleta, width=0.45, edgecolor="black")
eixos[1, 1].set_title("4. Retencao Diagnostica (F1-Score)", fontsize=10, fontweight="bold")
eixos[1, 1].set_ylabel("F1-Score no Teste Cego", fontsize=9)
eixos[1, 1].set_ylim(0.70, 0.92)
eixos[1, 1].text(1, 0.872, "+2.37 pts", ha="center", fontsize=10, fontweight="bold", color="#27ae60")
eixos[1, 1].grid(axis="y", linestyle=":", alpha=0.6)

plt.suptitle("DASHBOARD EXECUTIVO COMPARATIVO: XAI APLICADA A REDUCAO DE DADOS", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
```

> **O que voce deve notar no grafico gerado:**
> 1. Os graficos de recursos operacionais (atributos, tempo e latencia) exibem quedas expressivas superiores a 50%.
> 2. O grafico de qualidade diagnostica mantem-se estavel com leve acrescimo, confirmando que a remocao de colunas nao degradou o poder resolutivo.

**Mini-experimento:** altere os valores de latencia para simular um cenario onde o modelo enxuto operasse com tempo identico ao original. O projeto ainda assim justificaria sua adocao com base apenas na reducao dos 30 exames de laboratorio?

---

## Subcamada 14.5: O momento serio da nossa aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

No fechamento do projeto de reducao de dimensionalidade, emitimos o quadro consolidado oficial de KPIs cruzando todas as dimensoes operacionais, estatisticas e clinicas.

```python
# =============================================================================
# APLICACAO REAL: EMISSAO OFICIAL DO QUADRO COMPARATIVO DE RESULTADOS
# Base oficial: 2.000 pacientes, avaliacao cega de teste
# =============================================================================
import pandas as pd

quadro_resultados = pd.DataFrame({
    "Dimensao Avaliada": [
        "Numero de Atributos de Entrada",
        "Acuracia Global no Teste Cego",
        "F1-Score Diagnostico (Harmonico)",
        "Sensibilidade Clinica (Recall)",
        "Especificidade do Modelo",
        "Area sob a Curva ROC (AUC)",
        "Gap de Overfitting (Treino - Teste)",
        "Tempo de Treinamento da Floresta",
        "Latencia Media de Inferencia",
        "Custo Estimado de Exames / Paciente"
    ],
    "Baseline Bruto (40 Vars)": [
        "40 atributos",
        "87.80%",
        "0.8373",
        "78.11%",
        "94.33%",
        "0.9475",
        "12.20% (Sobreajuste acentuado)",
        "~600 ms",
        "~30.8 us",
        "R$ 1.200,00"
    ],
    "Modelo Campeao (10 Vars)": [
        "10 atributos (-75.0%)",
        "89.40% (+1.60%)",
        "0.8610 (+2.37 pts)",
        "82.50% (+4.39 pts)",
        "94.00% (-0.33%)",
        "0.9510 (+0.0035)",
        "6.10% (Gap reduzido a metade)",
        "~185 ms (-69.2%)",
        "~12.4 us (-59.7%)",
        "R$ 300,00 (-75.0%)"
    ]
})

print("=" * 82)
print("QUADRO OFICIAL CONSOLIDADO DE KPIS: BASELINE VS. MODELO REDUZIDO CAMPEAO")
print("=" * 82)
print(quadro_resultados.to_string(index=False))
print("=" * 82)
```

### Tabela oficial de KPIs

> Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Como e calculado | Pergunta operacional |
| :--- | :--- | :--- |
| **Taxa de Poda Estrutural (%)** | $(1 - p_{\text{reduzido}} / p_{\text{bruto}}) \times 100$ | Quantas variaveis foram retiradas permanentemente do fluxo de coleta? |
| **Sensibilidade Clinica (Recall)** | $\text{VP} / (\text{VP} + \text{FN})$ no teste cego | A remocao de exames evitou o aumento de Falsos Negativos? |
| **Contencao de Overfitting ($\Delta \text{Gap}$)** | $\text{Gap}_{\text{bruto}} - \text{Gap}_{\text{enxuto}}$ | A reducao de dimensionalidade tornou as arvores menos dependentes de ruido? |
| **Reducao de Custo Laboratorial (%)** | Proporcao direta do custo financeiro dos exames eliminados | Qual e o impacto economico direto da pesquisa na operacao hospitalar? |

### Interpretacao clinica e de negocio

A analise integrada dos KPIs fundamenta a viabilidade pratica da pesquisa:

1. **Prevalencia da seguranca diagnostica:** o ganho de 4.39 pontos percentuais na Sensibilidade Clinica comprova que o modelo enxuto detecta pacientes doentes com maior eficacia do que a versao original de 40 atributos, reduzindo Falsos Negativos graves.
2. **Estabilidade de custos laboratoriais:** ao concentrar o diagnostico em 10 biomarcadores consolidados, o servico de saude economiza insumos e padroniza processos de triagem sem incorrer em riscos regulatorios.
3. **Escalabilidade em ambientes embarcados:** a latencia reduzida para 12 microssegundos viabiliza a integracao do modelo em equipamentos medicos portateis com capacidade limitada de processamento local.

---

## Subcamada 14.6: Checkpoint de autonomia e fixacao ativa

Explique sem consultar o texto e depois confira sua resposta:

1. **Como se explica tecnicamente o fato de um modelo com 10 variaveis atingir $F_1$-score superior ao modelo com 40 variaveis?**
2. **Qual e a resposta metodologica para questionamentos sobre o custo computacional consumido nas fases de explicabilidade e busca bayesiana?**
3. **Por que a reducao do gap de generalizacao de 12.2% para 6.1% e considerada uma vitoria cientifica relevante?**
4. **Qual e o impacto clinico de observar que a sensibilidade (recall) aumentou de 78.1% para 82.5% apos a reducao de atributos?**
5. **Em termos de infraestrutura de computacao, qual e o beneficio de reduzir a latencia media de inferencia de 30 para 12 microssegundos?**
6. **Como os quatro quadrantes do dashboard executivo resumem o trade-off entre custo computacional, dimensionalidade e qualidade diagnostica?**

### Mini-desafio pratico

Construa uma projecao de impacto orcamentario em diferentes escalas de atendimento hospitalar e complete a tabela comparativa:

| Volume Anual de Pacientes | Custo Baseline (40 Exames) | Custo Proposto (10 Exames) | Economia Financeira Liquida |
| :--- | :--- | :--- | :--- |
| 10.000 pacientes | | | |
| 50.000 pacientes | | | |
| 200.000 pacientes | | | |

*Premissa de calculo:* custo medio estimado de R$ 30,00 por exame individual.

**Pergunta reflexiva:** o ganho financeiro obtido e acompanhado de reducao ou de elevacao do risco diagnostico de negligencia medica?
