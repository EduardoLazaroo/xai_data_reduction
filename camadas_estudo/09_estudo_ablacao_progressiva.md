# Camada 09: Estudo de Ablacao Progressiva

**Trilha:** XAI Aplicada a Reducao de Dados em Machine Learning  
**Aplicacao:** classificacao binaria de saude ('0 = Saudavel', '1 = Patologia')  
**Codigo de referencia:** [pipeline_completo.py](../pipeline_completo.py), funcao `executar_etapa_ablacao`

> **Objetivo da aula:** Compreender a formulacao metodologica de um estudo de ablacao (ablation study), analisando a resposta empirica de modelos de aprendizado sob remocao progressiva de atributos, a interpretacao das curvas de retencao de F1-score e a determinacao matematica do ponto de inflexao (knee point).

---

## Mapa da aula

1. [Subcamada 09.1: O conceito na vida real](#subcamada-091-o-conceito-na-vida-real)
2. [Subcamada 09.2: Desenhando o conceito](#subcamada-092-desenhando-o-conceito)
3. [Subcamada 09.3: Desmistificando a teoria e a notacao formal](#subcamada-093-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 09.4: Laboratorio ludico no Colab](#subcamada-094-laboratorio-ludico-no-colab)
5. [Subcamada 09.5: O momento serio da nossa aplicacao](#subcamada-095-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 09.6: Checkpoint de autonomia e fixacao ativa](#subcamada-096-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 09.1: O conceito na vida real

### A analogia do teste de estresse estrutural e a torre de blocos

Em engenharia estrutural, antes de liberar um novo projeto de ponte para o trafego publico, os projetistas realizam testes de estresse progressivo. Eles removem deliberadamente cabos auxiliares ou suportes secundarios para responder a duas questoes criticas:
1. Qual e a capacidade de carga residual da estrutura quando operando apenas com seus pilares mestres?
2. Em que limiar exato a retirada de mais um elemento provoca o colapso estrutural?

Em uma torre de blocos de madeira, retirar as pecas periféricas e soltas das extremidades nao abala a firmeza do conjunto; a torre permanece estavel. No entanto, quando se remove uma viga central de sustentacao, toda a estrutura desmorona de maneira instantanea.

No aprendizado de maquina, o estudo de ablacao (*ablation study*) e o procedimento cientifico de remocao metódica de componentes (neste caso, colunas de dados) para isolar sua contribuicao real para a performance do modelo.

### A analogia da orquestra musical

Considere uma orquestra composta por 10 solistas excepcionais. Se adicionarmos ao palco 20 pessoas batendo tampas de metal de forma aleatoria (ruido puro) e mais 10 musicos repetindo exatamente as mesmas notas do primeiro violino (redundancia), a qualidade acustica do espetaculo se deteriora. O publico tera dificuldade em distinguir a melodia principal em meio ao barulho excessivo.

Ao retirar os 20 elementos ruidosos e os 10 redundantes:
- O som torna-se mais nitido, claro e harmonioso.
- No algoritmo de aprendizado supervisionado ocorre dinamica similar: ao remover colunas de ruido e redundancia, as arvores de decisao deixam de quebrar nos em atributos espurios, reduzindo o sobreajuste (*overfitting*) e muitas vezes elevando a capacidade de generalizacao ($F_1$-Score) no conjunto de teste cego.

**A grande sacada:** o estudo de ablacao nao busca apenas cortar dados para economizar memoria; ele investiga a fronteira de otimalidade onde a simplicidade estrutural alcanca o desempenho preditivo maximo.

| Fase da Ablacao | Variacao no Numero de Atributos | Comportamento do Desempenho ($F_1$) | Impacto Operacional |
| :--- | :--- | :--- | :--- |
| **Plato de Estabilidade** | Reducao de 40 para ~12 variaveis | Estavel ou em leve ascensao | Eliminacao de ruido sem perda de sinal diagnostico |
| **Ponto de Inflexao (*Knee Point*)** | Exatamente em $k = 10$ variaveis | Maxima retencao de $F_1$ com dimensionalidade minima | Arquitetura enxuta otima para implantacao hospitalar |
| **Zona de Colapso Preditivo** | Reducao para $k < 6$ variaveis | Queda abrupta e acentuada | Perda de biomarcadores informativos indispensaveis |

---

## Subcamada 09.2: Desenhando o conceito

O grafico abaixo sintetiza o comportamento tipico de uma curva de ablacao com o eixo horizontal invertido, ilustrando a trajetoria da esquerda (modelo completo) para a direita (modelo enxuto):

```text
F1-Score
 1.0 +-------------------------------------------------------+
     |                PLATO DE ESTABILIDADE                  |
 0.8 |      * * * * * * * * * * * * * * [Knee Point: k=10]   | <- Mantem F1 maximo
     |                                  \                    |
 0.6 |                                   \  QUEDA ABRUPTA:   | <- Colapso preditivo
     |                                    \ Descarte de      |    por falta de sinal
 0.4 |                                     \ biomarcadores   |
 0.2 |                                      \ vitais         |
 0.0 +---------------------------------------*---------------+
     40        34        28        22        10   8   6   4  2
     <--------------------------------------------------------
                Quantidade de Atributos Retidos (k)
```

O fluxo experimental de ablacao compara trajetorias de poda derivadas de diferentes criterios de selecao:

```text
[ RANKING SHAP (XAI) ]              [ RANKING RFE (TRADICIONAL) ]
          │                                      │
          └──────────────────┬───────────────────┘
                             │
                             ▼
              [ LACO PROGRESSIVO DE ABLACAO ]
              Para k em [40, 38, 36, ..., 10, ..., 2]:
                1. Extrai as top-k colunas do ranking correspondente
                2. Ajusta nova Random Forest do zero em X_train[:, top_k]
                3. Cronometra a latencia de ajuste (tempo_fit_ms)
                4. Avalia predicoes cegas em X_test[:, top_k] e registra F1
                             │
                             ▼
              [ CURVAS DE ABLACAO COMPARATIVAS ]
        (Identifica qual metodo sustenta melhor o F1 com menor k)
```

| Zona da Curva | Diagnostico Tecnico | Decisao de Engenharia |
| :--- | :--- | :--- |
| **$k \in [40, 16]$** | O modelo contem ruido tolerado pela floresta, com sobrecusto de processamento. | Seguro para eliminar colunas em lote. |
| **$k \in [14, 10]$** | Proximidade do nucleo causal do fenomeno biologico. | Regiao recomendada para definicao do modelo final. |
| **$k \le 6$** | Subdimensionamento grave (*underfitting* estrutural induzido). | Proibido podar: perda irreversivel de informacao clinica. |

---

## Subcamada 09.3: Desmistificando a teoria e a notacao formal

### A formalizacao do espaco de ablacao

Seja $\mathcal{R} = (j_1, j_2, \dots, j_p)$ um ranking ordenado de $p$ atributos, em que $j_1$ representa o atributo mais importante e $j_p$ o menos relevante segundo determinado criterio (como SHAP ou RFE).

Para cada nivel de granularidade $k \in \mathcal{K}$, define-se o subconjunto ativo de variaveis:

$$\mathcal{F}_k = \{j_1, j_2, \dots, j_k\} \subset \{1, 2, \dots, p\}$$

O estimador ajustado com o subconjunto $\mathcal{F}_k$ e denotado por:

$$\hat{f}_k = \arg\min_{f \in \mathcal{H}} \mathcal{L}_{\text{train}}\left(f(X_{[:, \mathcal{F}_k]}), \, y\right)$$

O desempenho no conjunto de teste independente e expresso em funcao de $k$:

$$\mathcal{M}(k) = \text{F1}\left(y_{\text{teste}}, \, \hat{f}_k(X_{\text{teste}, [:, \mathcal{F}_k]})\right)$$

### O criterio geometrico do Knee Point ($k^*$)

O ponto de inflexao (*knee point*) representa a solucao do problema multiobjetivo de maximizar a retencao de informacao diagnostica minimizando a complexidade dimensional. Formalmente, considerando uma tolerancia aceitavel de perda $\epsilon \ge 0$:

$$k^* = \min \left\{ k \in \mathcal{K} \mid \mathcal{M}(k) \ge \mathcal{M}(p) - \epsilon \right\}$$

Sob a otica geometrica do metodo da distancia perpendicular, $k^*$ e o ponto da curva $(k, \mathcal{M}(k))$ que maximiza a distancia ortogonal em relacao a reta secante que conecta os extremos $(p, \mathcal{M}(p))$ e $(k_{\min}, \mathcal{M}(k_{\min}))$.

| Simbolo | Significado Formal | Leitura no Projeto |
| :--- | :--- | :--- |
| $\mathcal{F}_k$ | Subconjunto contendo os top-$k$ atributos selecionados | Subconjunto de exames retidos para aquele experimento |
| $\mathcal{M}(k)$ | Metrica de generalizacao em funcao da dimensionalidade $k$ | $F_1$-Score calculado no teste cego com $k$ atributos |
| $k^*$ | Dimensionalidade do ponto de inflexao (*knee point*) | Quantidade minima otima de exames ($k=10$) |
| $\epsilon$ | Margem de perda tolerada em relacao ao baseline | No projeto, a diferenca e nula ou positiva ($\mathcal{M}(k^*) \ge \mathcal{M}(40)$) |
| $T_{\text{fit}}(k)$ | Tempo de processamento necessario para ajustar o modelo com $k$ colunas | Queda de processamento de ~600 ms para ~185 ms |

### A ordem correta evita vazamento

1. O ranking de prioridade dos atributos $\mathcal{R}$ deve ser calculado exclusivamente sobre a particao de treino.
2. Cada modelo $\hat{f}_k$ do laco de ablacao e ajustado apenas em `X_train[:, F_k]`.
3. O conjunto `X_test[:, F_k]` atua estritamente como avaliador cego final, garantindo que a curva reflita a capacidade real de generalizacao.

---

## Subcamada 09.4: Laboratorio ludico no Colab

Execute o codigo abaixo no Google Colab para simular um estudo de ablacao em uma base com 15 atributos e visualizar o joelho da curva:

```python
# =============================================================================
# CAMADA 09: LABORATORIO LUDICO DE ESTUDO DE ABLACAO
# Demonstracao: Trajetoria de Desempenho e Localizacao do Ponto de Inflexao
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

np.random.seed(42)

# 1. Base sintetica com 15 atributos (5 informativos verdadeiros e 10 ruidos)
X, y = make_classification(
    n_samples=1000,
    n_features=15,
    n_informative=5,
    n_redundant=0,
    n_classes=2,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# 2. Modelo inicial para extracao do ranking de importancia
modelo_base = RandomForestClassifier(n_estimators=50, random_state=42)
modelo_base.fit(X_train, y_train)
indices_ordenados = np.argsort(modelo_base.feature_importances_)[::-1]

# 3. Laco de ablacao regressiva: de k = 15 ate k = 1
niveis_k = list(range(15, 0, -1))
historico_f1 = []

for k in niveis_k:
    atributos_k = indices_ordenados[:k]
    clf_k = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_k.fit(X_train[:, atributos_k], y_train)
    pred_k = clf_k.predict(X_test[:, atributos_k])
    historico_f1.append(f1_score(y_test, pred_k))

# 4. Construcao visual da curva de ablacao
plt.figure(figsize=(9, 4.5))
plt.plot(niveis_k, historico_f1, marker="o", color="#1f77b4", linewidth=2.0, label="F1-Score no Teste")
plt.axvline(x=5, color="#d62728", linestyle="--", linewidth=1.5, label="Knee Point Real (k = 5)")
plt.gca().invert_xaxis()

plt.title("Curva Didatica de Ablacao: Plato e Ponto de Inflexao", fontsize=11, fontweight="bold")
plt.xlabel("Numero de Atributos Retidos (k)", fontsize=10)
plt.ylabel("F1-Score de Generalizacao", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()

print(f"F1 com 15 atributos (inclui 10 ruidos) : {historico_f1[0]:.4f}")
print(f"F1 no Knee Point (apenas os 5 vitais) : {historico_f1[10]:.4f}")
print(f"F1 com 1 unico atributo (colapso)     : {historico_f1[-1]:.4f}")
```

> **O que voce deve notar no grafico gerado:**
> 1. Entre $k=15$ e $k=5$, a metrica permanece praticamente inalterada; podar as colunas de ruido nao afeta a capacidade de predicao.
> 2. No momento em que $k$ cai abaixo de 5, ocorre a quebra acentuada da curva porque o modelo e privado de caracteristicas informativas indispensaveis.

**Mini-experimento:** altere `n_informative=5` para `n_informative=8` no gerador. Como o joelho da curva respondeu? Ele se deslocou para a nova fronteira de informacao verdadeira?

---

## Subcamada 09.5: O momento serio da nossa aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

No protocolo experimental do projeto de reducao de dimensionalidade, comparamos a trajetoria de ablacao obtida a partir do ranking SHAP (nossa proposta com XAI) contra o ranking RFE (baseline tradicional) nos 40 atributos hospitalares.

```python
# =============================================================================
# APLICACAO REAL: ESTUDO COMPARATIVO DE ABLACAO (SHAP VS. RFE)
# Base oficial: 2.000 pacientes, 40 atributos clinicos
# =============================================================================
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score

# 1. Dataset oficial padronizado
X_raw, y = make_classification(
    n_samples=2000,
    n_features=40,
    n_informative=10,
    n_redundant=10,
    n_classes=2,
    weights=[0.6, 0.4],
    flip_y=0.03,
    random_state=42
)

feature_names = (
    [f"biomarcador_{i+1:02d}" for i in range(10)] +
    [f"redundante_{i+1:02d}" for i in range(10)] +
    [f"ruido_{i+1:02d}" for i in range(20)]
)

df_clinico = pd.DataFrame(X_raw, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    df_clinico, y, test_size=0.25, stratify=y, random_state=42
)

# 2. Extracao do ranking RFE tradicional
seletor_rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=1, step=2)
seletor_rfe.fit(X_train, y_train)
ranking_rfe = np.argsort(seletor_rfe.ranking_)

# 3. Extracao do ranking via importancia aditiva (proxy computacional de TreeExplainer)
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
modelo_rf.fit(X_train, y_train)
ranking_shap = np.argsort(modelo_rf.feature_importances_)[::-1]

# 4. Avaliacao comparativa nos patamares estrategicos de dimensionalidade
degraus_k = [40, 20, 10, 6, 2]
linhas_resultado = []

print("=" * 76)
print("TRAJETORIA EXPERIMENTAL DE ABLACAO: SHAP VS. RFE")
print("=" * 76)

for k in degraus_k:
    # Cenário com Ranking SHAP
    cols_s = ranking_shap[:k]
    t0 = time.perf_counter()
    rf_s = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_s.fit(X_train.iloc[:, cols_s], y_train)
    tempo_s_ms = (time.perf_counter() - t0) * 1000
    f1_s = f1_score(y_test, rf_s.predict(X_test.iloc[:, cols_s]))
    
    # Cenário com Ranking RFE
    cols_r = ranking_rfe[:k]
    t0 = time.perf_counter()
    rf_r = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf_r.fit(X_train.iloc[:, cols_r], y_train)
    tempo_r_ms = (time.perf_counter() - t0) * 1000
    f1_r = f1_score(y_test, rf_r.predict(X_test.iloc[:, cols_r]))
    
    reducao_pct = (1.0 - (k / 40.0)) * 100.0
    
    linhas_resultado.append({
        "k_Atributos": k,
        "Reducao_Dados": f"{reducao_pct:.1f}%",
        "F1_SHAP": f"{f1_s:.4f}",
        "F1_RFE": f"{f1_r:.4f}",
        "Tempo_Treino": f"{tempo_s_ms:.1f} ms"
    })

df_ablacao = pd.DataFrame(linhas_resultado)
print(df_ablacao.to_string(index=False))
print("=" * 76)
```

### Tabela oficial de KPIs

> Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Como e calculado | Pergunta operacional |
| :--- | :--- | :--- |
| **$F_1$-Score no Knee Point ($k=10$)** | Medido no conjunto de teste cego retendo os top-10 atributos | O modelo simplificado preserva ou supera a capacidade diagnostica do modelo completo? |
| **Taxa de Reducao de Dados (%)** | $(1 - k / p) \times 100$ | Qual e o percentual de simplificacao volumetrica obtido no banco hospitalar? |
| **Velocidade de Treinamento (ms)** | Cronometragem de ajuste da floresta com $k$ colunas | Quanto tempo computacional e poupado em rotinas periodicas de re-treinamento? |
| **Margem de Resiliencia do Plato** | Intervalo $[k^*, p]$ com variacao de $F_1 \le 0.015$ | Quantos atributos podem ser descartados sem degradacao perceptivel de precisao? |

### Interpretacao clinica e de negocio

A analise do estudo de ablacao oferece subsidios estrategicos para a gestao hospitalar:

1. **Eficiencia em larga escala:** demonstrar que o modelo operando com $k=10$ exames mantem o mesmo $F_1$-score do modelo original com 40 exames representa uma reducao de 75% na demanda de coletas laboratoriais, aliviando filas e custos operacionais sem penalizar a seguranca diagnostica.
2. **Mitigacao de Falsos Negativos por foco no sinal:** a remocao de colunas de ruido metabolico impede que pacientes verdadeiramente doentes sejam classificados erroneamente como saudaveis em decorrencia de flutuacoes aleatorias de dados espurios.
3. **Identificacao do limite de seguranca:** a queda acentuada observada para $k < 8$ define um teto regulatorio inegociavel: reduzir o painel abaixo de 10 atributos compromete a integridade clinica, advertindo a administracao hospitalar contra cortes orcamentarios indiscriminados.

---

## Subcamada 09.6: Checkpoint de autonomia e fixacao ativa

Explique sem consultar o texto e depois confira sua resposta:

1. **O que e metodologicamente um estudo de ablacao e qual e o seu papel na validacao cientifica de sistemas de IA?**
2. **Como se define conceitualmente o ponto de inflexao (*knee point*) em uma curva de ablacao?**
3. **Por que em muitos cenarios a eliminacao inicial de atributos irrelevantes provoca um aumento no $F_1$-Score em vez de reducao?**
4. **Qual e a consequencia tecnica de continuar podando variaveis apos ultrapassar o *knee point* em direcao a $k=2$?**
5. **Por que a comparacao de curvas de ablacao (SHAP versus RFE) e metodologicamente superior a comparar apenas um ranking isolado em uma tabela estatica?**
6. **Como o estudo de ablacao previne o erro metodologico de recomendar uma reducao excessiva de exames com prejuizo diagnostico?**

### Mini-desafio pratico

Execute o protocolo de ablacao testando diferentes sementes de aleatoriedade e complete o quadro de controle de sensibilidade:

| Semente Aleatoria (`random_state`) | $F_1$ no Baseline ($k=40$) | $F_1$ no Knee Point ($k=10$) | Ponto de Colapso ($F_1 < 0.70$) |
| :--- | :--- | :--- | :--- |
| `seed = 42` | | | |
| `seed = 101` | | | |
| `seed = 2024` | | | |

**Pergunta reflexiva:** o comportamento geometrico da curva (presenca de plato seguido de queda abrupta apos a perda de atributos informativos) permaneceu consistente independentemente da divisao amostral avaliada?
