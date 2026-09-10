# Camada 01: Fundamentos de Aprendizado Supervisionado e a Dinâmica do Overfitting

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 1  
**Contexto Técnico:** [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py) (`gerar_dataset_sintetico_saude` e `train_test_split`)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> Compreender de forma definitiva e cristalina — como quem aprende com um bom professor — o que é o Aprendizado de Máquina Supervisionado, a diferença prática entre prever números contínuos e tomar decisões categóricas, o perigo invisível do Sobreajuste (*Overfitting*) e por que ter atributos demais em uma base de dados pode sabotar qualquer modelo de inteligência artificial.

---

## Sumário da Aula

- [Subcamada 1.1: O Duelo de Paradigmas — Classificação vs. Regressão](#subcamada-11-o-duelo-de-paradigmas--classificação-vs-regressão)
- [Subcamada 1.2: A Anatomia dos Dados — Features ($X$) e Target ($y$)](#subcamada-12-a-anatomia-dos-dados--features-x-e-target-y)
- [Subcamada 1.3: A Separação Sagrada — Treino, Teste e o Fantasma do Data Leakage](#subcamada-13-a-separação-sagrada--treino-teste-e-o-fantasma-do-data-leakage)
- [Subcamada 1.4: A Dinâmica do Overfitting & O Mal da Alta Dimensionalidade](#subcamada-14-a-dinâmica-do-overfitting--o-mal-da-alta-dimensionalidade)
- [Subcamada 1.5: Laboratório Lúdico no Colab (Toy Example com Visualização Gráfica)](#subcamada-15-laboratório-lúdico-no-colab-toy-example-com-visualização-gráfica)
- [Subcamada 1.6: O Momento Sério da Nossa Aplicação (Dataset Clínico Real & Análise de KPIs)](#subcamada-16-o-momento-sério-da-nossa-aplicação-dataset-clínico-real--análise-de-kpis)
- [Subcamada 1.7: Checkpoint de Autonomia & Fixação Ativa](#subcamada-17-checkpoint-de-autonomia--fixação-ativa)

---

## Subcamada 1.1: O Duelo de Paradigmas — Classificação vs. Regressão

Imagine que você está ensinando uma criança a entender o mundo. Se você mostrar a ela uma régua e perguntar: *"Qual é a altura exata desta planta?"*, a resposta será um número medido em centímetros: $18.4\text{ cm}$.  
Agora, se você apontar para um cesto de frutas e perguntar: *"Esta fruta é uma Maçã ou uma Laranja?"*, a resposta não será um número contínuo, mas uma **escolha entre caixas pré-definidas**.

Em Machine Learning Supervisionado, todos os problemas do planeta se dividem fundamentalmente nessas duas grandes famílias.

```
                  ┌───────────────────────────────────────────────┐
                  │       APRENDIZADO DE MÁQUINA SUPERVISIONADO    │
                  └───────────────────────┬───────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     ┌────────────────────────┐                      ┌────────────────────────┐
     │       REGRESSÃO        │                      │     CLASSIFICAÇÃO      │
     │      (Linha / Eixo)    │                      │   (Fronteira / Caixas) │
     └────────────┬───────────┘                      └────────────┬───────────┘
                  │                                               │
        Pergunta: "QUANTO?"                             Pergunta: "QUAL CAIXA?"
        Alvo: Número Contínuo                           Alvo: Categoria Discreta
        Ex: Pressão Arterial (128.5 mmHg)               Ex: Doente (1) vs Saudável (0)
```

### 1.1.1 Analogias do Cotidiano (Para Fixar de Primeira)

| Situação Real | Pergunta de Regressão ("Quanto?") | Pergunta de Classificação ("Qual Rótulo?") |
| :--- | :--- | :--- |
| **Clima e Tempo** | *"Qual será a temperatura exata amanhã ao meio-dia?"* (ex: $28.3^\circ\text{C}$) | *"Devo levar guarda-chuva ou não?"* (Sim / Não) |
| **Mercado Imobiliário**| *"Por quanto este apartamento será vendido?"* (ex: $R\$\,435.000,00$) | *"Este imóvel é um Bom Negócio ou Mau Negócio?"* |
| **Medicina Diagnóstica**| *"Quantos miligramas de glicose há por decilitro de sangue?"* (ex: $112\text{ mg/dL}$) | *"O paciente é Diabético ou Não-Diabético?"* (Classe 1 ou 0) |
| **Transações Bancárias**| *"Qual é o valor financeiro da transferência?"* (ex: $R\$\,1.450,20$) | *"Esta transação é Fraude ou Legítima?"* |

### 1.1.2 Onde Nosso Projeto se Encaixa?
Nosso projeto lida estritamente com **Classificação Binária**.  
Não estamos tentando prever a quantidade de dias que um paciente viverá (isso seria regressão de sobrevida). Nós queremos responder com precisão cirúrgica a uma pergunta de sim ou não:

$$\text{O paciente possui a patologia sob investigação? } \longrightarrow \begin{cases} y = 1 & (\text{SIM, Positivo}) \\ y = 0 & (\text{NÃO, Negativo / Saudável}) \end{cases}$$

---

## Subcamada 1.2: A Anatomia dos Dados — Features ($X$) e Target ($y$)

Pense em um modelo de Machine Learning como um **detetive novato** que precisa investigar um caso.

```
       [ FICHA DE PISTAS DO DETETIVE ]                     [ O VEREDITO FINAL ]
       Matriz de Atributos: X (Features)                    Vetor Alvo: y (Target)
       
       Paciente 1: [ Glicose: 140, Idade: 58, IMC: 31 ] ──► [ Patologia: 1 (Doente) ]
       Paciente 2: [ Glicose:  85, Idade: 24, IMC: 21 ] ──► [ Patologia: 0 (Saudável) ]
       Paciente 3: [ Glicose: 195, Idade: 67, IMC: 29 ] ──► [ Patologia: 1 (Doente) ]
```

1. **Features ($X$ — Letra Maiúscula):**  
   São as pistas, evidências e medições. Chamamos de matriz porque possui **linhas** (cada paciente atendido) e **colunas** (cada exame clínico coletado).
2. **Target ($y$ — Letra Minúscula):**  
   É o veredito comprovado por um exame padrão-ouro (biópsia, laudo patológico). É um vetor unidimensional (uma única coluna) contendo a verdade dos fatos (*Ground Truth*).

O trabalho do modelo é encontrar uma regra matemática que ligue as pistas ($X$) ao veredito final ($y$).

---

## Subcamada 1.3: A Separação Sagrada — Treino, Teste e o Fantasma do Data Leakage

### 1.3.1 A Analogia do Aluno "Decorador de Gabarito"

Imagine um professor de física que, antes do vestibular, entrega uma lista com **100 exercícios resolvidos** para seus alunos estudarem.
- **Aluno A (Entendeu a Física):** Ele estuda os princípios fundamentais, a lei da gravidade e as fórmulas de movimento.
- **Aluno B (Decorador):** Ele não entende nada de física, mas tem uma memória fotográfica invejável. Ele decora que *"no exercício 34, a resposta é 42 m/s"* e que *"no exercício 78, a resposta é a letra C"*.

Se o professor aplicar uma prova contendo **exatamente as mesmas 100 questões da lista**:
- O Aluno B tira nota 10 com louvor!
- Mas o Aluno B aprendeu física? **Não.** No dia do vestibular, diante de uma questão inédita, ele tirará zero!

```
                                  TODO O HOSPITAL (2.000 Pacientes)
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼ (75%)                                                           ▼ (25%)
      DADOS DE TREINO (1.500 Casos)                                     DADOS DE TESTE (500 Casos)
     (A "Lista de Exercícios")                                           (A "Prova Surpresa")
                 │                                                                 │
                 ▼                                                                 ▼
      O algoritmo estuda aqui!                                          O algoritmo NUNCA viu!
      Ajusta pesos, ramos e regras.                                     Mede a capacidade real de
                                                                        salvar vidas no mundo real.
```

> [!CAUTION]
> 🚨 **O Fantasma do Data Leakage (Vazamento de Dados):**  
> Se você normalizar os dados, preencher valores nulos ou selecionar atributos olhando para o dataset inteiro antes de fazer a separação, os dados de teste "vazam" para dentro do treino. O modelo fingirá ter um desempenho espetacular, mas falhará catastroficamente assim que for instalado no computador do hospital!

---

## Subcamada 1.4: A Dinâmica do Overfitting & O Mal da Alta Dimensionalidade

### 1.4.1 O que é Overfitting na Prática?
O **Overfitting (Sobreajuste)** é a doença do Aluno Decorador no Machine Learning: o algoritmo decora até os menores ruídos e imperfeições da amostra de treino, perdendo completamente a capacidade de **generalizar**.

```
    PADRÃO REAL (GENERALIZÁVEL)                     OVERFITTING (MEMORIZAÇÃO DE RUÍDO)
    
       y │        *                                    y │        *
         │      *   *  *                                 │      * / \ *
         │    *       *                                  │    *--/   \--*   <- Fronteira "doida"
         │  *                                            │  * /         \      tentando abraçar
         └────────────── x                               └─/───────────── x    cada ponto isolado!
       Curva suave e elegante.                         Curva cheia de dentes e recortes bizarros.
       Acerta novos pacientes!                         Erra qualquer paciente inédito!
```

---

### 1.4.2 Por Que Datasets com Muitos Exames Aceleram o Overfitting?

Esta é a pergunta de ouro do nosso projeto de pesquisa: **Por que ter 40 colunas em vez de 10 facilita tanto o sobreajuste?**

#### Analogia 1: A Lei das Moedas e as Coincidências Espúrias
Se você jogar uma moeda para o alto 10 vezes, a chance de caírem 10 "caras" seguidas é ínfima ($0.098\%$). Você diria que isso é quase impossível.  
No entanto, imagine que você reúna **10.000 pessoas em um estádio de futebol** e mande todas jogarem moedas simultaneamente.  
Por pura probabilidade estatística, **cerca de 10 pessoas vão tirar 10 "caras" seguidas!**

Essas 10 pessoas têm poderes telecinéticos? Claro que não! Foi pura sorte estatística.  
No nosso dataset médico:
- Temos **20 colunas de ruído aleatório puro** (geradas como chiado estático de rádio).
- O algoritmo de IA não sabe o que é medicina.
- Se uma coluna de ruído qualquer tiver, por mero acaso, números ligeiramente maiores nos pacientes que estavam doentes no treino, a IA assumirá que aquele ruído é uma "descoberta científica revolucionária"!  
- No teste cego, a coincidência não se repete, e o diagnóstico falha.

#### Analogia 2: A Sala de Duas Paredes vs. O Labirinto de 40 Dimensões
- Em **2 dimensões** (ex: Altura e Peso), 2.000 pacientes preenchem o plano de forma densa e agrupada. É fácil traçar uma linha divisória elegante.
- Em **40 dimensões**, o espaço geométrico se expande de forma astronômica. 2.000 pacientes tornam-se grãos de poeira infinitesimais perdidos em um vazio colossal. O modelo cria "tendas" e "reentrâncias" em volta de cada paciente solitário, acreditando que descobriu uma regra universal.

---

## Subcamada 1.5: Laboratório Lúdico no Colab (Toy Example com Visualização Gráfica)

Antes de irmos para os dados sérios do hospital, vamos provar essa intuição com um experimento simples de apenas 120 pontos em 2D.  
Você pode copiar e colar o bloco abaixo diretamente no [Google Colab](https://colab.research.google.com).

```python
# =============================================================================
# LABORATÓRIO DIDÁTICO: O MONSTRO DO OVERFITTING EM AÇÃO
# Objetivo: Ver visualmente como um modelo sem freios cria uma fronteira "doida"
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

# 1. Criamos um dataset "brinquedo" 2D com formato de duas meias-luas e bastante ruído
np.random.seed(42)
X_toy, y_toy = make_moons(n_samples=120, noise=0.35, random_state=42)

# 2. Treinamos dois modelos:
#    Modelo A: Equilibrado (Profundidade controlada = 3 cortes simples)
#    Modelo B: Com Overfitting Severo (Profundidade infinita = decora cada ponto)
clf_suave = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_overfit = DecisionTreeClassifier(max_depth=15, min_samples_split=2, random_state=42)

clf_suave.fit(X_toy, y_toy)
clf_overfit.fit(X_toy, y_toy)

# 3. Função para desenhar a fronteira de decisão de forma limpa
def plotar_fronteira(clf, X, y, titulo, ax):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", edgecolors="k", label="Classe 0 (Saudável)")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="red", edgecolors="k", label="Classe 1 (Doente)")
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)

# 4. Exibição gráfica comparativa
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plotar_fronteira(clf_suave, X_toy, y_toy, "Modelo Saudável (Generaliza Bem)\nFronteira Suave", axes[0])
plotar_fronteira(clf_overfit, X_toy, y_toy, "Modelo com Overfitting (Decorou o Ruído)\nFronteira com 'Ilhas' Artificiais", axes[1])
plt.tight_layout()
plt.show()

print(f"Acurácia Treino - Modelo Saudável : {clf_suave.score(X_toy, y_toy)*100:.1f}%")
print(f"Acurácia Treino - Modelo Overfit  : {clf_overfit.score(X_toy, y_toy)*100:.1f}% (Falsamente perfeito!)")
```

### O Que Você Deve Observar no Gráfico Gerado:
1. No gráfico da **esquerda**, o modelo aceita errar um ou outro ponto ruidoso isolado para manter uma fronteira lógica e contínua.
2. No gráfico da **direita**, o modelo cria **"ilhas" e "tentáculos" azuis dentro do território vermelho** apenas para conseguir 100% de acerto na amostra de treino. Quando um paciente novo cair dentro desse tentáculo, o diagnóstico será um desastre!

---

## Subcamada 1.6: O Momento Sério da Nossa Aplicação (Dataset Clínico Real & Análise de KPIs)

Agora que você dominou o conceito na teoria, compreendeu a analogia das moedas e viu o gráfico do overfitting com seus próprios olhos, **acabou a brincadeira**.  
Vamos para a trincheira real do nosso projeto de pesquisa: **o cenário clínico do hospital com 40 exames coletados por paciente**.

### 1.6.1 O Cenário Técnico da Aplicação Médica
Nosso hospital atende 2.000 pacientes. Para cada um, o sistema colhe:
- **10 Biomarcadores Vitais:** Indicadores biológicos reais que causam a doença (ex: Troponina, Glicemia, D-Dímero).
- **10 Exames Redundantes:** Exames que repetem a mesma informação (ex: Hemoglobina Glicada medindo o mesmo fenômeno da Glicemia).
- **20 Ruídos Metabólicos Puros:** Flutuações aleatórias sem nenhuma relação causal com o desfecho clínico.

Vamos treinar o **Modelo Baseline (Random Forest)** e extrair a régua oficial de **KPIs do Projeto**.

```python
# =============================================================================
# O MOMENTO SÉRIO DA NOSSA APLICAÇÃO:
# Baseline Clínico com 40 Atributos & Extração dos KPIs Oficiais
# =============================================================================
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

print("=" * 70)
print("INICIANDO PROTOCOLO EXPERIMENTAL: MODELO CLÍNICO BASELINE (40 ATRIBUTOS)")
print("=" * 70)

# 1. GERAÇÃO DO DATASET SINTÉTICO HOSPITALAR (Simulando 2.000 Pacientes)
X_raw, y = make_classification(
    n_samples=2000,
    n_features=40,
    n_informative=10,
    n_redundant=10,
    n_repeated=0,
    n_classes=2,
    weights=[0.6, 0.4],  # 60% Saudáveis (0) e 40% Doentes (1)
    flip_y=0.03,          # 3% de ruído humano/laboratorial no diagnóstico
    random_state=42
)

feature_names = (
    [f"biomarcador_{i+1}" for i in range(10)] +
    [f"exame_redundante_{i+1}" for i in range(10)] +
    [f"ruido_metabolico_{i+1}" for i in range(20)]
)
df_clinico = pd.DataFrame(X_raw, columns=feature_names)

# 2. SEPARAÇÃO RIGOROSA: TREINO (75%) E TESTE (25%) COM ESTRATIFICAÇÃO
X_train, X_test, y_train, y_test = train_test_split(
    df_clinico, y, test_size=0.25, stratify=y, random_state=42
)

# 3. TREINAMENTO DO BASELINE COM CRONOMETRIA DE ALTA PRECISÃO
modelo_baseline = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

t0_treino = time.perf_counter()
modelo_baseline.fit(X_train, y_train)
tempo_treino_ms = (time.perf_counter() - t0_treino) * 1000

# 4. INFERÊNCIA NO CONJUNTO DE TESTE COM MEDIÇÃO DE LATÊNCIA
t0_inf = time.perf_counter()
y_pred = modelo_baseline.predict(X_test)
tempo_inf_ms = (time.perf_counter() - t0_inf) * 1000
y_proba = modelo_baseline.predict_proba(X_test)[:, 1]

# 5. CÁLCULO DOS KPIS DE DESEMPENHO E GENERALIZAÇÃO
acc_treino = accuracy_score(y_train, modelo_baseline.predict(X_train))
acc_teste = accuracy_score(y_test, y_pred)
gap_overfitting = (acc_treino - acc_teste) * 100

kpis = {
    "Acurácia Treino": f"{acc_treino*100:.2f}%",
    "Acurácia Teste": f"{acc_teste*100:.2f}%",
    "Gap de Overfitting": f"{gap_overfitting:.2f}%",
    "F1-Score": f"{f1_score(y_test, y_pred):.4f}",
    "Precisão": f"{precision_score(y_test, y_pred):.4f}",
    "Sensibilidade (Recall)": f"{recall_score(y_test, y_pred):.4f}",
    "ROC-AUC": f"{roc_auc_score(y_test, y_proba):.4f}",
    "Tempo de Treinamento": f"{tempo_treino_ms:.1f} ms",
    "Latência de Inferência (500 pacientes)": f"{tempo_inf_ms:.1f} ms",
    "Latência Média por Paciente": f"{(tempo_inf_ms / len(y_test))*1000:.1f} µs"
}

print("\n📊 TABELA OFICIAL DE KPIS DO MODELO BASELINE:")
print("-" * 60)
for k, v in kpis.items():
    print(f"  • {k.ljust(35)} : {v}")
print("-" * 60)

# 6. PLOTAGEM DOS GRÁFICOS CLÍNICOS: MATRIZ DE CONFUSÃO & CURVA ROC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
            xticklabels=["Saudável (0)", "Patologia (1)"],
            yticklabels=["Saudável (0)", "Patologia (1)"])
axes[0].set_title("Matriz de Confusão (Baseline - 40 Atributos)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Diagnóstico Previsto pelo Modelo", fontweight="bold")
axes[0].set_ylabel("Diagnóstico Real do Paciente", fontweight="bold")

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.3f})")
axes[1].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Classificador Aleatório (Chute)")
axes[1].set_title("Curva ROC Diagnóstica", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Taxa de Falsos Positivos (1 - Especificidade)", fontweight="bold")
axes[1].set_ylabel("Taxa de Verdadeiros Positivos (Sensibilidade)", fontweight="bold")
axes[1].legend(loc="lower right")
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()
```

---

### 1.6.2 Interpretação Clínica e de Negócio dos Resultados do KPI

Vamos dissecar o que esses números significam na prática hospitalar:

| Indicador (KPI) | Valor Obtido no Baseline | Significado no Mundo Real | Consequência no Hospital |
| :--- | :--- | :--- | :--- |
| **Acurácia no Treino** | **100.00%** | O modelo memorizou perfeitamente todos os 1.500 casos de treino. | Alerta vermelho! Ele se comportou exatamente como o "aluno decorador". |
| **Acurácia no Teste** | **87.80%** | Em pacientes novos e inéditos, a acurácia cai para $87.8\%$. | Queda expressiva de desempenho. |
| **Gap de Overfitting** | **12.20%** | A diferença entre treino e teste é de mais de 12 pontos percentuais! | **Comprova o sobreajuste.** As 20 colunas de ruído iludiram as árvores de decisão. |
| **F1-Score** | **0.8373** | Média harmônica entre a precisão e o recall da classe doente. | Nossa régua mínima de qualidade. Não aceitaremos nenhum modelo futuro abaixo de $0.83$. |
| **Erros na Matriz (FN)**| **44 Falsos Negativos** | 44 pacientes doentes receberam laudo de "saudável" e foram para casa sem tratamento! | O custo mais alto em medicina: risco de vida para o paciente. |
| **Tempo de Treinamento**| **~600 ms** | Tempo necessário para construir 100 árvores dividindo 40 colunas. | Alto consumo de processamento; inviável para retreinamentos contínuos em grande escala. |
| **Número de Exames** | **40 colunas** | O hospital precisa coletar 40 exames por paciente. | Custo financeiro absurdo para o SUS/plano de saúde e dias de espera para o paciente. |

> [!IMPORTANT]
> 💡 **A Missão do Restante do Nosso Projeto:**  
> O Baseline provou duas coisas:
> 1. Ele consegue diagnosticar razoavelmente bem ($F_1 \approx 0.84$), mas está sofrendo de **12.2% de overfitting** devido às 20 colunas inúteis de ruído.
> 2. O custo para manter 40 exames é desnecessário.
> 
> Nas próximas camadas, usaremos **XAI (SHAP e LIME)** para interrogar o modelo, descobrir os 10 biomarcadores vitais, jogar fora os 20 ruídos e provar que um modelo com apenas 8 a 10 atributos atinge o **mesmo F1-Score com zero ruído, menor latência e custo infinitamente menor**!

---

## Subcamada 1.7: Checkpoint de Autonomia & Fixação Ativa

Não avance para a Camada 02 sem responder com total clareza mental às perguntas abaixo (use o método de Feynman: imagine que você está explicando para um amigo que nunca viu programação na vida):

1. **Por que um modelo com 100% de acurácia no treinamento deve acender um sinal de perigo imediato em vez de ser motivo de comemoração?**
2. **Qual é a diferença fundamental entre prever a pressão arterial de alguém (ex: 135 mmHg) e prever se a pessoa está hipertensa? Qual é regressão e qual é classificação?**
3. **Explique a analogia das moedas no estádio de futebol: como ela prova matematicamente que colocar 20 colunas de ruído aleatório em um dataset cria 'falsas descobertas' no aprendizado de máquina?**
4. **Desafio no Colab:** No código da Subcamada 1.6, mude o parâmetro `n_features=40` para `n_features=10` e remova os ruídos (deixe apenas os 10 biomarcadores úteis). O que aconteceu com o **Gap de Overfitting** entre Treino e Teste? Ele diminuiu?
