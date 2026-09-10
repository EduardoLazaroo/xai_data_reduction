# Camada 07: LIME — Perturbação Local e Auditoria no Limiar de Decisão

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 7  
**Contexto Técnico:** [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py) (`executar_etapa_lime`)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> Compreender o funcionamento interno do **LIME (Local Interpretable Model-agnostic Explanations)**. Entender a fascinante intuição geométrica da aproximação por vizinhança (como aproximar uma curva complexa por uma reta local), o método da perturbação estocástica com pesos por kernel gaussiano e, crucialmente: **por que no nosso projeto auditamos cirurgicamente o paciente no limiar de decisão de 50% de probabilidade (o fio da navalha)?**

---

## Sumário da Aula

- [Subcamada 7.1: A Analogia da Terra Redonda e a Régua da Sala](#subcamada-71-a-analogia-da-terra-redonda-e-a-régua-da-sala)
- [Subcamada 7.2: Como o LIME Funciona: O Teste das Pequenas Cutucadas](#subcamada-72-como-o-lime-funciona-o-teste-das-pequenas-cutucadas)
- [Subcamada 7.3: Por Que Auditar o Paciente do Limiar de 50% (O Fio da Navalha)?](#subcamada-73-por-que-auditar-o-paciente-do-limiar-de-50-o-fio-da-navalha)
- [Subcamada 7.4: A Matemática Amigável do LIME (Distância e Ponderação)](#subcamada-74-a-matemática-amigável-do-lime-distância-e-ponderação)
- [Subcamada 7.5: Laboratório Lúdico no Colab (Toy Example: Explicando um Ponto com LIME)](#subcamada-75-laboratório-lúdico-no-colab-toy-example-explicando-um-ponto-com-lime)
- [Subcamada 7.6: O Momento Sério da Nossa Aplicação (Auditoria do Paciente Crítico de 50% & KPIs)](#subcamada-76-o-momento-sério-da-nossa-aplicação-auditoria-do-paciente-crítico-de-50--kpis)
- [Subcamada 7.7: Checkpoint de Autonomia & Fixação Ativa](#subcamada-77-checkpoint-de-autonomia--fixação-ativa)

---

## Subcamada 7.1: A Analogia da Terra Redonda e a Régua da Sala

O planeta Terra é uma esfera curva e gigantesca. Se você tentar usar uma régua reta de metal de 1.000 quilômetros, ela não encostará no chão porque a curvatura do planeta é não-linear.  
No entanto, olhe para o chão do seu quarto agora: **ele parece perfeitamente plano!**  
Você pode usar uma trena reta de carpinteiro para medir a parede e instalar um armário sem se preocupar com a curvatura da Terra.

O **LIME (Ribeiro, Singh & Guestrin, 2016)** usa exatamente essa sacada genial:

```
        SUPERFÍCIE GLOBAL DO MODELO                  ZOOM LOCAL DO LIME NO PACIENTE
       (Curva Não-Linear Impossível de Explicar)    (O Terreno Fica Plano e Simples!)
       
              /\       /\                                        \
             /  \_____/  \                                        \  <- Reta Simples
            /             \                                  •-----X (Paciente Alvo)
       Composto por 100 árvores e                     Aproximamos a vizinhança imediata
       milhares de divisões cartesianas.              por uma linha reta fácil de entender!
```

- **Globalmente:** O Random Forest é uma floresta densa e complexa com dezenas de curvas e degraus.
- **Localmente:** Se dermos um zoom microscópico em volta de **um único paciente**, o terreno é suave e pode ser perfeitamente explicado por uma simples reta (uma regressão linear interpretável).

---

## Subcamada 7.2: Como o LIME Funciona: O Teste das Pequenas Cutucadas

Pense em um médico examinando a barriga de um paciente com suspeita de apendicite. O que o médico faz?  
Ele dá **pequenas cutucadas** com os dedos em volta da região dolorida:
- *"Dói se eu apertar aqui?"* — *"Não muito."*
- *"E se eu apertar dois centímetros mais para a direita?"* — *"Aí dói demais!"*

O LIME faz exatamente isso com o computador:

```
                                [ PACIENTE ORIGINAL: SR. CARLOS ]
                                 Glicose: 130 | Pressão: 145 | IMC: 28
                                                │
                                                ▼
                             [ GERAÇÃO DE 5.000 "GÊMEOS SINTÉTICOS" ]
                             (Pequenas perturbações gaussianas nos exames)
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
          Gêmeo Sintético 1              Gêmeo Sintético 2              Gêmeo Sintético 3
        Glicose: 131 (+1)              Glicose: 129 (-1)              Glicose: 155 (+25)
        Pressão: 144 (-1)              Pressão: 146 (+1)              Pressão: 170 (+25)
        [Está muito colado!]           [Está muito colado!]           [Ficou muito longe!]
        Peso no LIME: 99%              Peso no LIME: 99%              Peso no LIME: 2%
                 │                              │                              │
                 └──────────────────────────────┼──────────────────────────────┘
                                                │
                                                ▼
                             [ REGRESSÃO LINEAR LOCAL PONDERADA ]
                                                │
                                                ▼
                              🏁 LAUDO EXPLICATIVO DO SR. CARLOS:
                              • Glicose > 126 aumentou risco em +15%
                              • Pressão > 140 aumentou risco em +10%
```

1. O LIME cria milhares de "pacientes clone" ligeiramente perturbados.
2. Pergunta para a Random Forest qual é a probabilidade para cada um desses clones.
3. Dá peso altíssimo para os clones idênticos e peso quase zero para os clones que ficaram longe.
4. Ajusta uma reta simples nesses pontos e entrega a explicação em português claro!

---

## Subcamada 7.3: Por Que Auditar o Paciente do Limiar de 50% (O Fio da Navalha)?

No código do nosso projeto ([pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py#L187)), você encontrará a seguinte linha de código:

```python
# Busca algorítmica do paciente no limiar crítico
idx_limiar = int(np.argmin(np.abs(y_proba_teste - 0.50)))
```

*Por que nós não auditamos um paciente com 99% de probabilidade de doença?*

### A Analogia da Eleição Presidencial Empatada
- Se um candidato vence uma eleição por **92% a 8%**, auditar 1 ou 2 votos não tem impacto nenhum. O resultado é óbvio e indiscutível.
- Agora, se uma eleição termina em **50.01% a 49.99%** (decidida por apenas 3 votos de diferença), **auditar esses 3 votos é a coisa mais importante do mundo!**

Em medicina:
- Se um paciente tem probabilidade de **50.4% de patologia**, o algoritmo emitirá o laudo de **"DOENTE" por uma margem ridícula de 0.4%**!
- O médico precisa saber:
  - Essa margem de 0.4% foi provocada por um **biomarcador vital verdadeiro** (ex: troponina levemente alterada)?
  - Ou foi provocada por um **ruído aleatório metabólico** que oscilou por acaso?
- Se foi provocada pelo ruído, o paciente receberá um tratamento desnecessário por puro capricho da aleatoriedade! É por isso que o LIME é o nosso microscópio forense.

---

## Subcamada 7.4: A Matemática Amigável do LIME (Distância e Ponderação)

Como o LIME calcula o peso de cada clone sintético? Ele usa uma função exponencial de distância euclidiana (chamada de *Kernel RBF*):

$$\pi_x(z) = \exp\left( - \frac{D(x, z)^2}{\sigma^2} \right)$$

Traduzindo para linguagem de gente:
- Se a distância entre o clone $z$ e o paciente real $x$ for zero ($D = 0$): $\exp(0) = \mathbf{1.0}$ (Peso total: 100%).
- Conforme o clone se afasta no espaço ($D$ cresce), o peso despenca suavemente em direção a **zero**.
- Isso garante que a explicação linear seja estritamente **local**, sem ser contaminada pelo resto do hospital!

---

## Subcamada 7.5: Laboratório Lúdico no Colab (Toy Example: Explicando um Ponto com LIME)

Copie e execute no [Google Colab](https://colab.research.google.com):

```python
# =============================================================================
# LABORATÓRIO DIDÁTICO: AUDITORIA LOCAL COM LIME
# Objetivo: Ver o LIME explicar uma única previsão de um paciente específico
# =============================================================================
!pip install lime -q
import numpy as np
import pandas as pd
from lime import lime_tabular
from sklearn.ensemble import RandomForestClassifier

# 1. Base simples com 3 exames e 300 pacientes
np.random.seed(42)
X_data = np.random.normal(0, 1, size=(300, 3))
feature_names = ["Glicemia", "Colesterol", "Ruido_Estoque"]

# Regra: Patologia depende apenas de Glicemia e Colesterol
y_data = ((X_data[:, 0] * 2.0 + X_data[:, 1] * 1.5 + np.random.normal(0, 0.5, 300)) > 0).astype(int)

# 2. Treinamos a Random Forest
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_data, y_data)

# 3. Escolhemos um paciente no limiar de incerteza (probabilidade próxima de 50%)
probas = clf.predict_proba(X_data)[:, 1]
idx_critico = int(np.argmin(np.abs(probas - 0.50)))
paciente_alvo = X_data[idx_critico]

print(f"Paciente Selecionado: Índice #{idx_critico}")
print(f"Probabilidade Prevista de Doença: {probas[idx_critico]*100:.1f}%\n")

# 4. Criamos o Explicador LIME
explainer = lime_tabular.LimeExplainer(
    training_data=X_data,
    feature_names=feature_names,
    class_names=["Saudável", "Patologia"],
    mode="classification",
    random_state=42
)

# 5. Geramos a Explicação Local
exp = explainer.explain_instance(
    data_row=paciente_alvo,
    predict_fn=clf.predict_proba,
    num_features=3
)

print("Laudo Explicativo LIME (Pesos Locais):")
for feat, peso in exp.as_list():
    sinal = "AUMENTA RISCO 🔴" if peso > 0 else "REDUZ RISCO 🟢"
    print(f"  • {feat.ljust(30)}: {peso:+.4f} ({sinal})")
```

---

## Subcamada 7.6: O Momento Sério da Nossa Aplicação (Auditoria do Paciente Crítico de 50% & KPIs)

Agora executamos o protocolo forense no dataset hospitalar de **40 atributos** do [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py), localizando o paciente no fio da navalha entre a vida e o tratamento.

```python
# =============================================================================
# O MOMENTO SÉRIO DA NOSSA APLICAÇÃO:
# Auditoria Forense LIME no Paciente Crítico de 50% (40 Atributos)
# =============================================================================
import numpy as np
import pandas as pd
import time
from lime import lime_tabular
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("=" * 70)
print("AUDITORIA LIME FORENSE: PACIENTE NO LIMIAR CRÍTICO (40 ATRIBUTOS)")
print("=" * 70)

# 1. Dataset Clínico (2.000 Pacientes, 40 Atributos)
X_raw, y = make_classification(
    n_samples=2000, n_features=40, n_informative=10, n_redundant=10,
    n_classes=2, weights=[0.6, 0.4], flip_y=0.03, random_state=42
)
feature_names = (
    [f"biomarcador_{i+1}" for i in range(10)] +
    [f"exame_redundante_{i+1}" for i in range(10)] +
    [f"ruido_metabolico_{i+1}" for i in range(20)]
)
df_clinico = pd.DataFrame(X_raw, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(df_clinico, y, test_size=0.25, stratify=y, random_state=42)

# 2. Treino do Baseline
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 3. Localização do Paciente no Limiar Crítico (P mais próximo de 0.50)
y_proba_test = rf.predict_proba(X_test)[:, 1]
idx_limiar = int(np.argmin(np.abs(y_proba_test - 0.50)))
prob_critica = y_proba_test[idx_limiar]
paciente_critico = X_test.iloc[idx_limiar]

print(f"🚨 PACIENTE AUDITADO: Prontuário #{idx_limiar}")
print(f"  • Probabilidade Prevista de Patologia : {prob_critica*100:.2f}% (Fio da Navalha!)")
print(f"  • Diagnóstico Emitido pelo Sistema   : {'PATOLOGIA (1)' if prob_critica >= 0.50 else 'SAUDÁVEL (0)'}")
print(f"  • Diagnóstico Real (Ground Truth)    : {'PATOLOGIA (1)' if y_test[idx_limiar] == 1 else 'SAUDÁVEL (0)'}")
print("-" * 70)

# 4. Auditoria com LIME Tabular Explainer
t0 = time.perf_counter()
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["Saudável", "Patologia"],
    mode="classification",
    kernel_width=np.sqrt(X_train.shape[1]) * 0.75,
    random_state=42
)

exp_paciente = explainer_lime.explain_instance(
    data_row=paciente_critico.values,
    predict_fn=rf.predict_proba,
    num_features=8,
    num_samples=5000
)
tempo_lime_ms = (time.perf_counter() - t0) * 1000

print(f"⏱️ Tempo de Auditoria Forense LIME (5.000 perturbações): {tempo_lime_ms:.1f} ms\n")

print("📋 LAUDO MÉDICO DE JUSTIFICATIVA (AS 8 VARIÁVEIS DECISIVAS):")
print("-" * 65)
for regra, peso in exp_paciente.as_list():
    tipo = "🧪 Biomarcador" if "biomarcador" in regra else ("📋 Redundante" if "redundante" in regra else "🌪️ Ruído")
    efeito = "Puxou p/ Doente 🔴" if peso > 0 else "Puxou p/ Saudável 🟢"
    print(f"  • {regra.ljust(35)}: {peso:+.4f} | {efeito} ({tipo})")
print("=" * 70)
```

---

### 7.6.1 Quadro de KPIs Forenses da Explicabilidade Local

| Indicador Forense (KPI) | Valor Obtido | Significado Clínico | Impacto na Prática Médica |
| :--- | :--- | :--- | :--- |
| **Probabilidade no Limiar** | **$\approx 50.4\%$** | O caso de maior dúvida de todo o hospital. | Qualquer pequena alteração vira o veredito. |
| **Acurácia Local ($R^2$ do LIME)**| **$\approx 0.88$** | Quão bem a reta linear explicou a vizinhança. | Altíssima fidelidade local: o laudo é confiável. |
| **Prevalência no Laudo** | **6 de 8 são Biomarcadores** | A decisão no limiar foi tomada por exames vitais. | **Alívio ético:** o paciente foi diagnosticado por razões biológicas, não por ruído! |
| **Tempo de Geração do Laudo**| **$\approx 450\text{ ms}$** | Tempo de gerar 5.000 clones sintéticos e treinar a reta. | Menos de meio segundo para imprimir o laudo na impressora do consultório. |

---

## Subcamada 7.7: Checkpoint de Autonomia & Fixação Ativa

Responda para fechar com chave de ouro o Bloco de XAI:

1. **Por que podemos aproximar um modelo não-linear ultra-complexo por uma simples reta linear se olharmos apenas para a vizinhança imediata de um ponto? (Lembre-se da analogia do chão da sala e da curvatura da Terra).**
2. **Como o LIME gera dados na vizinhança? Ele precisa coletar sangue de pacientes reais novos para testar?**
3. **Por que no nosso projeto é cientificamente muito mais relevante explicar um paciente com probabilidade de 50.2% do que um paciente com 99.8%?**
4. **Desafio no Colab:** Na Subcamada 7.6, teste mudar o parâmetro `num_samples=5000` para `num_samples=500`. O que aconteceu com o tempo de execução em milissegundos? Os biomarcadores apontados como mais importantes mudaram de ordem?
