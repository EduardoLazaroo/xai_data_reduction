# Camada 03: Métricas de Avaliação — O Rigor Diagnóstico Além da Acurácia

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 3  
**Contexto Técnico:** [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py) (`treinar_e_avaliar_modelo` e `plotar_graficos_baseline`)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> Dominar de forma profunda e visual as métricas de diagnóstico em Machine Learning. Entender por que a **Acurácia é uma ilusão perigosa** em dados reais de saúde, o cabo de guerra entre **Precisão** e **Recall**, o segredo por trás da média harmônica do **$F_1$-Score**, e como ler a **Matriz de Confusão** e a **Curva ROC-AUC** como um verdadeiro especialista.

---

## Sumário da Aula

- [Subcamada 3.1: A Analogia do Alarme de Incêndio e a Matriz da Verdade](#subcamada-31-a-analogia-do-alarme-de-incêndio-e-a-matriz-da-verdade)
- [Subcamada 3.2: O Paradoxo da Acurácia — A Fraude do Médico Preguiçoso](#subcamada-32-o-paradoxo-da-acurácia--a-fraude-do-médico-preguiçoso)
- [Subcamada 3.3: O Cabo de Guerra: Precisão vs. Recall (A Metáfora do Pescador)](#subcamada-33-o-cabo-de-guerra-precisão-vs-recall-a-metáfora-do-pescador)
- [Subcamada 3.4: Por Que o F1-Score Usa a Média Harmônica?](#subcamada-34-por-que-o-f1-score-usa-a-média-harmônica)
- [Subcamada 3.5: A Curva ROC e o ROC-AUC Explicados Visualmente](#subcamada-35-a-curva-roc-e-o-roc-auc-explicados-visualmente)
- [Subcamada 3.6: Laboratório Lúdico no Colab (Toy Example: Desmascarando a Falsa Acurácia)](#subcamada-36-laboratório-lúdico-no-colab-toy-example-desmascarando-a-falsa-acurácia)
- [Subcamada 3.7: O Momento Sério da Nossa Aplicação (Auditoria Completa de KPIs no Dataset Clínico)](#subcamada-37-o-momento-sério-da-nossa-aplicação-auditoria-completa-de-kpis-no-dataset-clínico)
- [Subcamada 3.8: Checkpoint de Autonomia & Fixação Ativa](#subcamada-38-checkpoint-de-autonomia--fixação-ativa)

---

## Subcamada 3.1: A Analogia do Alarme de Incêndio e a Matriz da Verdade

Imagine que você instalou um sensor eletrônico de fumaça na cozinha do seu apartamento. Existem apenas quatro coisas possíveis que podem acontecer no mundo real:

```
                            SITUAÇÃO REAL DO APARTAMENTO
                         PEGANDO FOGO 🔥             SEM FOGO (TUDO BEM) 🟢
                   ┌─────────────────────────┬─────────────────────────┐
    ALARME         │  VERDADEIRO POSITIVO    │     FALSO POSITIVO      │
    TOCOU 🚨       │  (VP - Salvou vidas!)   │     (FP - Alarme Falso) │
                   ├─────────────────────────┼─────────────────────────┤
    ALARME FICOU   │     FALSO NEGATIVO      │  VERDADEIRO NEGATIVO    │
    SILENCIOSO 🔕  │  (FN - TRAGÉDIA TOTAL!) │  (VN - Noite Tranquila) │
                   └─────────────────────────┴─────────────────────────┘
```

### O Peso Humano de Cada Quadrante em Medicina:
1. **Verdadeiro Positivo (VP):** O paciente tem a patologia e o algoritmo detectou. Ele inicia o tratamento a tempo e sobrevive.
2. **Verdadeiro Negativo (VN):** O paciente está saudável e o algoritmo confirmou. Ele volta para casa aliviado.
3. **Falso Positivo (FP) — Erro Tipo I:** O paciente é saudável, mas o modelo disse que ele está com a patologia.  
   *Consequência:* Pânico psicológico, repetição de exames laboratoriais caros e biópsias invasivas desnecessárias.
4. **Falso Negativo (FN) — Erro Tipo II:** O paciente está gravemente doente, mas o modelo disse que ele está saudável!  
   *Consequência:* **O pior cenário em medicina.** O paciente vai embora do hospital com uma dipirona acreditando que está bem, o tumor se espalha e ele pode ir a óbito.

---

## Subcamada 3.2: O Paradoxo da Acurácia — A Fraude do Médico Preguiçoso

A fórmula da acurácia é a mais famosa e ingênua de todas:
$$\text{Acurácia} = \frac{\text{Total de Acertos}}{\text{Total de Pacientes}} = \frac{VP + VN}{VP + VN + FP + FN}$$

Por que ela pode ser uma **mentira deslavada** em problemas médicos?

### A História do "Médico Preguiçoso":
Imagine um hospital que faz triagem para uma doença cardíaca rara que afeta **1 a cada 100 pessoas** (1% doentes, 99% saudáveis).  
Um médico charlatão decide não trabalhar: para todos os 10.000 pacientes que passam pela porta dele no ano, ele carimba na ficha: **"Saudável (0)"** sem sequer levantar os olhos do celular!

- Quantos pacientes ele acertou? **9.900 pessoas!**
- Qual é a acurácia dele? $\frac{9.900}{10.000} = \mathbf{99.0\%}$!
- Se você olhasse apenas para a Acurácia, acharia esse médico um gênio com 99% de precisão!
- **A tragédia:** Ele errou todos os 100 pacientes cardíacos. Todos os 100 doentes foram para casa e sofreram infarto. Ele matou 100 pessoas com 99% de acurácia!

> [!CAUTION]
> 💡 **Regra de Ouro:** Nunca apresente um projeto de Inteligência Artificial para saúde ou negócios confiando exclusivamente na Acurácia. Dados reais quase nunca são perfeitamente 50/50.

---

## Subcamada 3.3: O Cabo de Guerra: Precisão vs. Recall (A Metáfora do Pescador)

Quando tentamos diagnosticar pacientes doentes, existe um cabo de guerra natural entre duas virtudes: **Precisão** e **Recall**. Pense em dois tipos de pescadores em um barco:

```
    PESCADOR COM ARPÃO DE PRECISÃO                     PESCADOR COM REDE DE RECALL
    
         "Só atiro se tiver certeza!"                     "Pego tudo o que se mexer!"
         
             🐟   🎯                                       [=== REDE GIGANTE ===]
             🐟                                            🐟  🐟  🥾  🧴  🐟  🌿
             🐟
    Acerta 100% dos tiros (Precisão Alta).             Captura todos os peixes (Recall Alto).
    Mas deixa 9 peixes escaparem (Recall Baixo).       Mas a rede vem cheia de lixo (Precisão Baixa).
```

### As Fórmulas Traduzidas para Português:
1. **Precisão (*Precision*):**  
   $$\text{Precisão} = \frac{VP}{VP + FP}$$  
   *Pergunta:* *"De todas as vezes que o modelo gritou 'ESTÁ DOENTE!', quantas vezes ele estava certo?"*  
   - Alta precisão significa poucos alarmes falsos ($FP \approx 0$).
2. **Recall / Sensibilidade (*True Positive Rate*):**  
   $$\text{Recall} = \frac{VP}{VP + FN}$$  
   *Pergunta:* *"De todos os pacientes que realmente tinham a doença no hospital, quantos o modelo conseguiu encontrar?"*  
   - Alto recall significa quase nenhum doente ignorado ($FN \approx 0$).

---

## Subcamada 3.4: Por Que o F1-Score Usa a Média Harmônica?

O **$F_1$-Score** combina a Precisão e o Recall em uma única nota:

$$F_1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}$$

### Por Que Não Usar a Média Normal (Aritmética)?
Imagine que um modelo tenha:
- **Precisão = 100%** (quando fala, nunca erra).
- **Recall = 0%** (mas é tão covarde que nunca deu diagnóstico positivo para ninguém!).

Se fizéssemos a média normal:
$$\text{Média Aritmética} = \frac{100\% + 0\%}{2} = 50\%$$
O modelo ganharia nota 50% ("mediano"), mesmo sendo totalmente inútil!

A **Média Harmônica** funciona como a velocidade média de uma viagem: se você vai a 100 km/h na ida, mas o carro quebra e você fica a 0 km/h na volta, você nunca chega ao destino!  
A média harmônica pune impiedosamente o desequilíbrio:
$$F_1 = 2 \cdot \frac{1.0 \cdot 0.0}{1.0 + 0.0} = \mathbf{0.00\%}$$
Ela só atinge valores altos se o modelo tiver **simultaneamente** alta Precisão e alto Recall.

---

## Subcamada 3.5: A Curva ROC e o ROC-AUC Explicados Visualmente

Pense no modelo de Machine Learning não apenas emitindo um "Sim" ou "Não", mas como um médico que diz: *"Este paciente tem **62% de chance** de estar doente"*.  
Por padrão, se a chance for maior que $50\%$ ($0.5$), o sistema carimba "Doente".

Mas e se mudarmos esse corte (*Threshold*)?
- Se você abaixar a régua para $20\%$, o modelo vira o Pescador de Rede: o Recall sobe, mas os alarmes falsos disparam.
- Se você subir a régua para $90\%$, o modelo vira o Pescador de Arpão: a Precisão sobe, mas vários doentes escapam.

```
       Taxa de Verdadeiros Positivos (Recall)
       1.0 ┌─────────────────────*───────────┐  <- AUC = 1.0 (Classificador Perfeito!)
           │                   *             │
           │                *                │
           │              *  [Área = 0.94]   │  <- Modelo Baseline do Nosso Projeto
           │            *                    │
           │          *                      │
           │        *                        │  <- Linha Tracejada (AUC = 0.50):
           │      *                          │     Equivale a jogar uma moeda!
       0.0 └─────────────────────────────────┘
          0.0                               1.0
             Taxa de Falsos Positivos (1 - Especificidade)
```

- **ROC-AUC = 0.50:** O modelo é tão bom quanto jogar uma moeda no ar (chute puro).
- **ROC-AUC = 0.90 - 0.99:** Excelente separação diagnóstica.
- **ROC-AUC = 1.00:** Perfeição absoluta (nenhum erro em nenhum limiar).

---

## Subcamada 3.6: Laboratório Lúdico no Colab (Toy Example: Desmascarando a Falsa Acurácia)

Copie e rode no [Google Colab](https://colab.research.google.com) para ver o Paradoxo da Acurácia acontecer na sua tela:

```python
# =============================================================================
# LABORATÓRIO DIDÁTICO: O PARADOXO DA ACURÁCIA
# Objetivo: Provar que um modelo que chuta sempre Saudável tem alta acurácia mas F1 nulo
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 1. Criamos uma população de 1.000 pacientes com doença rara (95% Saudáveis, 5% Doentes)
np.random.seed(42)
X_raro = np.random.randn(1000, 5)
y_raro = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])

# 2. Modelo A: O "Médico Preguiçoso" (Chuta sempre a classe majoritária 0)
modelo_preguicoso = DummyClassifier(strategy="most_frequent")
modelo_preguicoso.fit(X_raro, y_raro)
pred_preguicoso = modelo_preguicoso.predict(X_raro)

# 3. Modelo B: Uma Random Forest que realmente tenta aprender
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_raro, y_raro)
pred_rf = modelo_rf.predict(X_raro)

print("=" * 65)
print(f"MODELO PREGUIÇOSO (Chuta sempre Saudável):")
print(f"  • Acurácia : {accuracy_score(y_raro, pred_preguicoso)*100:.1f}% (Parece espetacular!)")
print(f"  • F1-Score : {f1_score(y_raro, pred_preguicoso):.4f} (DESASTRE TOTAL: zero utilidade!)")
print("-" * 65)
print(f"MODELO REAL (Random Forest):")
print(f"  • Acurácia : {accuracy_score(y_raro, pred_rf)*100:.1f}%")
print(f"  • F1-Score : {f1_score(y_raro, pred_rf):.4f} (Capaz de salvar vidas!)")
print("=" * 65)

# Plotagem da Matriz de Confusão do Médico Preguiçoso
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_raro, pred_preguicoso), annot=True, fmt="d", cmap="Reds", ax=ax, cbar=False,
            xticklabels=["Saudável", "Doente"], yticklabels=["Saudável", "Doente"])
ax.set_title("Matriz de Confusão do 'Médico Preguiçoso'\n(Errou 100% dos Doentes com 95% de Acurácia!)", fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## Subcamada 3.7: O Momento Sério da Nossa Aplicação (Auditoria Completa de KPIs no Dataset Clínico)

Chegou a hora de auditar o **modelo Baseline real de 40 atributos** do nosso hospital ([pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py)) e extrair a régua completa de KPIs.

```python
# =============================================================================
# O MOMENTO SÉRIO DA NOSSA APLICAÇÃO:
# Auditoria Completa de Métricas Clínicas no Baseline (40 Atributos)
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

# 1. Dataset Clínico Hospitalar (2.000 Pacientes, 40 Atributos)
X_raw, y = make_classification(
    n_samples=2000, n_features=40, n_informative=10, n_redundant=10,
    n_classes=2, weights=[0.6, 0.4], flip_y=0.03, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.25, stratify=y, random_state=42
)

# 2. Treino do Baseline
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 3. Predições e Matriz de Confusão
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
VN, FP = cm[0, 0], cm[0, 1]
FN, VP = cm[1, 0], cm[1, 1]

# 4. Cálculo Manual e Verificação das Métricas
acc = (VP + VN) / (VP + VN + FP + FN)
prec = VP / (VP + FP)
rec = VP / (VP + FN)
f1 = 2 * (prec * rec) / (prec + rec)
auc = roc_auc_score(y_test, y_proba)

print("=" * 65)
print("AUDITORIA MÉDICA DE KPIS - TESTE CEGO (500 PACIENTES)")
print("=" * 65)
print(f"  • Verdadeiros Positivos (VP) : {VP} pacientes doentes detectados com sucesso")
print(f"  • Verdadeiros Negativos (VN) : {VN} pacientes saudáveis confirmados")
print(f"  • Falsos Positivos (FP)      : {FP} alarmes falsos (biópsias extras)")
print(f"  • Falsos Negativos (FN)      : {FN} pacientes doentes liberados por engano!")
print("-" * 65)
print(f"  • Acurácia Global            : {acc*100:.2f}%")
print(f"  • Precisão Diagnóstica       : {prec*100:.2f}%")
print(f"  • Sensibilidade (Recall)     : {rec*100:.2f}%")
print(f"  • F1-Score Consolidado       : {f1:.4f}")
print(f"  • Área sob a Curva (ROC-AUC) : {auc:.4f}")
print("=" * 65)
```

---

### 3.7.1 Tabela de Auditoria e Decisão Médica Hospitalar

| Métrica Diagnóstica | Valor Clínico Obtido | A Pergunta que ela Responde | Decisão de Gestão Hospitalar |
| :--- | :--- | :--- | :--- |
| **Acurácia** | **87.80%** | De 500 pacientes, quantos tiveram laudo correto? | Boa para apresentar à diretoria, mas esconde os 44 doentes ignorados. |
| **Precisão** | **90.23%** | Quando o laudo dá Positivo, qual a chance de ser verdade? | Alta. De 174 alarmes, 157 estavam corretos (poucas biópsias desnecessárias). |
| **Recall (Sensibilidade)**| **78.11%** | De todos os doentes reais (201), quantos foram achados? | **Preocupante!** Mais de 21% dos doentes passaram despercebidos. |
| **F1-Score** | **0.8373** | Qual o equilíbrio harmônico entre precisão e recall? | **Nossa régua de corte.** Nenhum modelo enxuto futuro pode ficar abaixo de 0.837. |
| **ROC-AUC** | **0.9475** | O modelo consegue ranquear os doentes acima dos saudáveis? | Excelente! Prova que mexendo no limiar podemos aumentar o recall. |

---

## Subcamada 3.8: Checkpoint de Autonomia & Fixação Ativa

Responda mentalmente para fixar antes de avançar para a Camada 04:

1. **Se um novo modelo de IA médica tiver Acurácia de 98% e F1-Score de 0.12, você autorizaria seu uso em um hospital? Por quê?**
2. **Em um teste de triagem para um vírus mortal e altamente contagioso, você prefere otimizar o modelo para ter alta Precisão ou alto Recall? Por quê?**
3. **Explique a metáfora do pescador com arpão versus pescador com rede de arrasto.**
4. **Desafio no Colab:** Na Subcamada 3.7, altere o limiar de decisão (`threshold`) de $0.5$ para $0.35$ usando as probabilidades (`y_proba >= 0.35`). O que aconteceu com o número de Falsos Negativos (FN)? O Recall aumentou? O que aconteceu com os alarmes falsos (FP)?
