# Camada 12: Otimizacao Bayesiana de Hiperparametros com Optuna

**Trilha:** XAI Aplicada a Reducao de Dados em Machine Learning  
**Aplicacao:** classificacao binaria de saude ('0 = Saudavel', '1 = Patologia')  
**Codigo de referencia:** [pipeline_completo.py](../pipeline_completo.py), funcao `otimizar_optuna`

> **Objetivo da aula:** Compreender a formulacao teorica da otimizacao bayesiana via Tree-structured Parzen Estimator (TPE), diferenciando parametros aprendidos internamente de hiperparametros arquiteturais, estruturando rotinas de validacao cruzada estratificada em k-folds e demonstrando a necessidade metodologica de reajustar o modelo sobre o espaco dimensional reduzido antes da avaliacao de generalizacao final.

---

## Mapa da aula

1. [Subcamada 12.1: O conceito na vida real](#subcamada-121-o-conceito-na-vida-real)
2. [Subcamada 12.2: Desenhando o conceito](#subcamada-122-desenhando-o-conceito)
3. [Subcamada 12.3: Desmistificando a teoria e a notacao formal](#subcamada-123-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 12.4: Laboratorio ludico no Colab](#subcamada-124-laboratorio-ludico-no-colab)
5. [Subcamada 12.5: O momento serio da nossa aplicacao](#subcamada-125-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 12.6: Checkpoint de autonomia e fixacao ativa](#subcamada-126-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 12.1: O conceito na vida real

### A analogia do garimpeiro inteligente contra o turista cego

Imagine um terreno montanhoso de centenas de hectares onde se suspeita haver jazidas de minerios preciosos:
1. **O explorador exaustivo (*Grid Search*):** divide a regiao em quadrantes regulares de 10 em 10 metros e perfura todos os pontos de forma mecanica. Ele gasta recursos cavando inclusive no meio de leitos rochosos estereis onde a probabilidade geologica de encontrar minerio e nula. O processo consome meses e gera alto custo financeiro.
2. **O explorador aleatorio (*Random Search*):** sorteia coordenadas aleatorias com dados para perfurar. Pode encontrar veios promissores com menos iteracoes do que a grade fixa, mas nao aprende com os poços secos ja escavados.
3. **O garimpeiro especialista (*Optuna / Otimizacao Bayesiana*):** perfura uma sondagem inicial. Ao constatar teor mineral nulo em uma vertente, mapeia a regiao como pouco promissora. Ao identificar tracos de mineralizacao em outra colina, formula a hipotese probabilistica de que a viga mestra localiza-se naquela direcao e concentra as proximas perfuracoes ao redor do pico de concentracao.

O framework Optuna implementa esse principio probabilistico por meio do estimador TPE (Tree-structured Parzen Estimator): a cada tentativa (*trial*), ele modela a distribuicao dos hiperparametros que produziram bons resultados e direciona as amostragens subsequentes para as regioes de maximo retorno esperado.

### A analogia do terno sob medida: por que re-otimizar o modelo reduzido?

Considere um atleta que pesava 130 kg e mandou confeccionar um paleto de corte amplo. Apos um ano de preparacao fisica intensa, o atleta reduz seu peso para 85 kg de massa magra. Seria um erro conceitual vestir o mesmo paleto antigo sobre o novo porte fisico; o tecido sobraria em todas as articulacoes. E mandatorio ajustar o corte para a nova silhueta.

No aprendizado de maquina supervisionado ocorre fenomeno analogo:
- O modelo **Baseline original operava com 40 atributos**, dos quais 30 eram redundantes ou puramente ruidosos. Para encontrar padroes no meio de tanto chiado, a floresta exigia arvores profundas (`max_depth = 15`), que compensavam o espaco disperso com multiplas particoes cartesianas.
- O modelo **Reduzido final opera com 10 atributos de elite** aprovados pelo `shap-select`. Nesse novo espaco denso e informativo, manter arvores profundas provoca sobreajuste imediato nas nuances das poucas colunas restantes.
- E mandatorio re-otimizar a profundidade, o numero de estimadores e os limites de divisao das folhas para equilibrar o modelo na nova dimensao.

**A grande sacada:** re-otimizar hiperparametros apos a reducao de dimensionalidade assegura uma comparacao justa entre a versao compacta e o baseline completo, extraindo o potencial maximo de cada configuracao.

| Estrategia de Busca | Mecanismo de Amostragem | Eficiencia em Recursos | Risco Metodologico |
| :--- | :--- | :--- | :--- |
| **Grid Search** | Varredura cartesiana deterministica completa | Baixa (explosao combinatoria) | Desperdicado em regioes sem gradiente |
| **Random Search** | Amostragem estocastica uniforme independente | Moderada | Nao acumula aprendizado historico |
| **Optuna (TPE Bayesiano)** | Modelagem probabilistica de densidade condicional | Alta (converge em poucas iteracoes) | Requer definicao coerente dos espacos de busca |

---

## Subcamada 12.2: Desenhando o conceito

O diagrama abaixo ilustra a diferenca espacial entre as estrategias classicas de exploracao e a concentracao bayesiana do algoritmo Optuna:

```text
GRID SEARCH (GRADE REGULAR)       RANDOM SEARCH (ALEATORIO)        OPTUNA (TPE BAYESIANO)
max_depth                        max_depth                        max_depth
  12 |  o   o   o   o   o          12 |      o      o               12 |        * * * (PICO)
     |                                |    o          o                |      * * * * *
   8 |  o   o   o   o   o           8 |         o       o            8 |    .   * * *
     |                                |  o        o                    |   . .
   4 |  o   o   o   o   o           4 |       o          o           4 |  .
     +-----------------------         +-----------------------         +-----------------------
       50  100 150 200 n_est            50  100 150 200 n_est            50  100 150 200 n_est
  (Testa cegamente todos os nós)   (Espalha pontos sem direcao)     (Concentra tentativas no maximo)
```

Para garantir que a pontuacao de cada configuracao avaliada pelo Optuna seja estatisticamente confiavel e imune a variacoes pontuais de divisao, emprega-se a validacao cruzada estratificada em 3 dobras:

```text
[ DADOS DE TREINAMENTO (X_train, y_train) ]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ ESTRATIFICACAO EM 3 DOBRAS (StratifiedKFold)                │
├─────────────────┬─────────────────┬─────────────────────────┤
│ DOBRA 1 (Treino)│ DOBRA 2 (Treino)│ DOBRA 3 (Validacao)     │ -> Score F1 = 0.862
├─────────────────┼─────────────────┼─────────────────────────┤
│ DOBRA 1 (Treino)│ DOBRA 2 (Valid.)│ DOBRA 3 (Treino)        │ -> Score F1 = 0.858
├─────────────────┼─────────────────┼─────────────────────────┤
│ DOBRA 1 (Valid.)│ DOBRA 2 (Treino)│ DOBRA 3 (Treino)        │ -> Score F1 = 0.864
└─────────────────┴─────────────────┴─────────────────────────┘
                                │
                                ▼
         PONTUACAO CONSOLIDADA DO TRIAL = MEDIA(F1) = 0.8613
```

| Componente | Funcao no Ciclo do Optuna | Comportamento se Omitido |
| :--- | :--- | :--- |
| **Funcao objetivo (`objective`)** | Mapeia o conjunto de hiperparametros em uma metrica escalar ($F_1$) | Sem ela nao ha direcao de convergencia |
| **Amostrador TPE (`sampler`)** | Sugere novos hiperparametros baseado nos trials anteriores | Busca reverte para amostragem puramente aleatoria |
| **Validacao Cruzada (3-Fold)** | Garante que o hiperparametro generaliza bem entre particoes | Otimizacao sobreajusta a uma unica fatia acidental |
| **Conjunto de Teste (`X_test`)** | Mantido cego e isolado durante toda a otimizacao | Se exposto durante os trials, ha vazamento de dados grave |

---

## Subcamada 12.3: Desmistificando a teoria e a notacao formal

### A formalizacao matematica da busca de hiperparametros

Seja $\boldsymbol{\theta} \in \Theta$ um vetor pertencente ao espaco multidimensional de configuracoes de hiperparametros, em que:

$$\boldsymbol{\theta} = \left[ n_{\text{estimators}}, \, \text{max\_depth}, \, \text{min\_samples\_split}, \, \text{min\_samples\_leaf} \right]^T$$

O problema de otimizacao consiste em encontrar a configuracao $\boldsymbol{\theta}^*$ que maximiza a metrica de avaliacao esperada sob validacao cruzada com $K$ dobras:

$$\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta} \in \Theta} \frac{1}{K} \sum_{k=1}^K \text{F1}\left(y_{\text{val}}^{(k)}, \, \hat{f}\left(X_{\text{val}}^{(k)}; \, \boldsymbol{\theta}, \, \mathcal{D}_{\text{train}}^{(k)}\right)\right)$$

### O algoritmo Tree-structured Parzen Estimator (TPE)

Diferente de modelos gaussianos padrao que modelam $P(y \mid \boldsymbol{\theta})$, o algoritmo TPE inverte o condicionamento pela regra de Bayes, modelando a densidade dos hiperparametros condicional ao desempenho obtido:

$$p(\boldsymbol{\theta} \mid y) = \begin{cases} \ell(\boldsymbol{\theta}) & \text{se } y > y^* \\ g(\boldsymbol{\theta}) & \text{se } y \le y^* \end{cases}$$

Em que $y^*$ e um limiar quantilico (por exemplo, os 15% melhores trials observados ate o momento). O criterio de aquisicao de melhoria esperada (*Expected Improvement* - EI) e entao formulado como:

$$\text{EI}_{y^*}(\boldsymbol{\theta}) = \int_{y^*}^{\infty} (y - y^*) \, p(y \mid \boldsymbol{\theta}) \, dy \propto \frac{\ell(\boldsymbol{\theta})}{g(\boldsymbol{\theta})}$$

Para maximizar a chance de progresso, o Optuna seleciona candidatos onde a razao $\ell(\boldsymbol{\theta}) / g(\boldsymbol{\theta})$ e maxima, priorizando regioes do espaco com alta concentracao de sucessos pregressos e baixa frequencia de configuracoes ineficazes.

| Simbolo | Significado Formal | Leitura no Projeto |
| :--- | :--- | :--- |
| $\boldsymbol{\theta}$ | Vetor de hiperparametros estruturais da arvore | Parametros como `n_estimators=100`, `max_depth=6` |
| $\Theta$ | Espaco delimitado de busca hiperparametrica | Limites definidos (ex: profundidade entre 4 e 12) |
| $K$ | Numero de dobras na particao de validacao cruzada | Adotado $K=3$ dobras estratificadas |
| $y^*$ | Limiar de corte dos trials de alto desempenho | Nota de $F_1$ que separa as tentativas superiores |
| $\ell(\boldsymbol{\theta})$ | Densidade probabilistica dos hiperparametros bem-sucedidos | Zona do espaco onde o modelo apresentou $F_1$ elevado |
| $g(\boldsymbol{\theta})$ | Densidade probabilistica dos hiperparametros malsucedidos | Zona do espaco com menor poder preditivo |

### A ordem correta evita vazamento metodologico

1. A particao de teste final `X_test` e `y_test` e trancada e nao participa de nenhum trial da otimizacao.
2. O Optuna executa os trials e as $K$ dobras de validacao cruzada exclusivamente dentro de `X_train`.
3. Concluida a convergencia, instancia-se um novo classificador com a configuracao $\boldsymbol{\theta}^*$, que e ajustado na totalidade de `X_train`.
4. Esse modelo campeao e avaliado uma unica vez sobre o conjunto cego `X_test`.

---

## Subcamada 12.4: Laboratorio ludico no Colab

Execute o bloco abaixo no Google Colab para acompanhar a convergencia do Optuna em um cenario enxuto:

```python
# =============================================================================
# CAMADA 12: LABORATORIO LUDICO DE OTIMIZACAO BAYESIANA COM OPTUNA
# Demonstracao: Sintonia Fina via TPE com Validacao Cruzada Estratificada
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

# 1. Base sintetica com 500 pacientes e 6 biomarcadores limpos
X_toy, y_toy = make_classification(
    n_samples=500,
    n_features=6,
    n_informative=6,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# 2. Definicao formal da funcao objetivo
def funcao_objetivo(trial):
    n_est = trial.suggest_int("n_estimators", 25, 125, step=25)
    profundidade = trial.suggest_int("max_depth", 3, 9)
    min_split = trial.suggest_int("min_samples_split", 2, 8)
    
    clf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=profundidade,
        min_samples_split=min_split,
        random_state=42
    )
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_toy, y_toy, cv=cv, scoring="f1")
    return float(scores.mean())

# 3. Execucao do estudo bayesiano com 15 tentativas
estudo = optuna.create_study(direction="maximize")
estudo.optimize(funcao_objetivo, n_trials=15)

valores_trials = [t.value for t in estudo.trials]
melhor_acumulado = np.maximum.accumulate(valores_trials)

# 4. Visualizacao grafica da convergencia
plt.figure(figsize=(9, 4.5))
plt.plot(range(1, 16), valores_trials, marker="o", linestyle=":", color="#7f8c8d", alpha=0.7, label="Tentativa Individual (Trial)")
plt.plot(range(1, 16), melhor_acumulado, marker="s", color="#2980b9", linewidth=2.0, label="Melhor Score Acumulado")
plt.title("Trajetoria de Otimizacao Bayesiana (Optuna TPE)", fontsize=11, fontweight="bold")
plt.xlabel("Numero da Tentativa (Trial)", fontsize=10)
plt.ylabel("F1-Score Medio (3-Fold CV)", fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()

print(f"Melhor pontuacao alcancada (CV): {estudo.best_value:.4f}")
print("Configuracao otima de hiperparametros:")
for k, v in estudo.best_params.items():
    print(f"  - {k.ljust(20)}: {v}")
```

> **O que voce deve notar no grafico gerado:**
> 1. Nas primeiras tentativas, os valores oscilam a medida que o algoritmo mapeia os limites do espaco.
> 2. Com a acumulacao de historico, a linha de melhor resultado alcanca um plato de excelencia estavel, evidenciando o direcionamento bayesiano das propostas.

**Mini-experimento:** altere `n_trials=15` para `n_trials=30`. A configuracao final encontrada alcancou ganho relevante adicional ou a convergencia basica ja havia se consolidado nas rodadas intermediarias?

---

## Subcamada 12.5: O momento serio da nossa aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

No projeto de pesquisa, executamos a re-otimizacao completa via Optuna exclusivamente sobre o subconjunto enxuto de 10 atributos selecionados pelo `shap-select`, avaliando o modelo campeao no conjunto de teste cego.

```python
# =============================================================================
# APLICACAO REAL: OTIMIZACAO BAYESIANA SOBRE ATRIBUTOS ENXUTOS
# Base oficial: 2.000 pacientes, 10 atributos de elite selecionados
# =============================================================================
import time
import numpy as np
import pandas as pd
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Dataset com os 10 biomarcadores de elite aprovados
X_raw, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=10,
    n_redundant=0,
    n_classes=2,
    weights=[0.6, 0.4],
    flip_y=0.03,
    random_state=42
)

feature_names = [f"biomarcador_{i+1:02d}" for i in range(10)]
df_elite = pd.DataFrame(X_raw, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    df_elite, y, test_size=0.25, stratify=y, random_state=42
)

# 2. Desempenho antes da otimizacao (modelo com hiperparametros padrao)
rf_padrao = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_padrao.fit(X_train, y_train)
f1_padrao = f1_score(y_test, rf_padrao.predict(X_test))

# 3. Protocolo de busca bayesiana com Optuna (15 trials, 3-fold CV)
def objetivo_producao(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200, step=25)
    max_depth = trial.suggest_int("max_depth", 4, 12)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 8)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
    
    estimador = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(estimador, X_train, y_train, cv=cv, scoring="f1")
    return float(scores.mean())

print("=" * 74)
print("INICIANDO RE-OTIMIZACAO BAYESIANA VIA OPTUNA (10 ATRIBUTOS)")
print("=" * 74)

t0_optuna = time.perf_counter()
estudo = optuna.create_study(direction="maximize")
estudo.optimize(objetivo_producao, n_trials=15)
tempo_optuna = time.perf_counter() - t0_optuna

# 4. Ajuste e avaliacao cega do modelo campeao
melhores_params = estudo.best_params
modelo_campeao = RandomForestClassifier(**melhores_params, random_state=42)
modelo_campeao.fit(X_train, y_train)

y_pred_campeao = modelo_campeao.predict(X_test)
y_proba_campeao = modelo_campeao.predict_proba(X_test)[:, 1]

f1_campeao = f1_score(y_test, y_pred_campeao)
acuracia_campea = accuracy_score(y_test, y_pred_campeao)
auc_campeao = roc_auc_score(y_test, y_proba_campeao)

print(f"Tempo total de otimizacao: {tempo_optuna:.2f} segundos")
print(f"Numero de trials executados: {len(estudo.trials)}")
print("-" * 74)
print("CONFIGURACAO CAMPEA ENCONTRADA:")
for param, valor in melhores_params.items():
    print(f"  - {param.ljust(22)}: {valor}")

print("-" * 74)
print("COMPARATIVO DE DESEMPENHO NO TESTE CEGO:")
print(f"  - F1-Score Padrao (Sem Tuning) : {f1_padrao:.4f}")
print(f"  - F1-Score Campeao com Optuna  : {f1_campeao:.4f} (Ganho: +{(f1_campeao - f1_padrao)*100:+.2f}%)")
print(f"  - Acuracia no Teste Cego       : {acuracia_campea * 100:.2f}%")
print(f"  - ROC-AUC Campeao              : {auc_campeao:.4f}")
print("=" * 74)
```

### Tabela oficial de KPIs

> Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Como e calculado | Pergunta operacional |
| :--- | :--- | :--- |
| **Ganho de $F_1$ via Tuning ($\Delta F_1$)** | $F_{1, \text{campeao}} - F_{1, \text{padrao}}$ no teste cego | Quanto de rendimento diagnostico adicional a sintonia fina agregou? |
| **Tempo Total de Otimizacao (s)** | Cronometrado via `time.perf_counter()` para 15 trials de 3-fold | O custo computacional de tuning e viavel para re-calibracoes periodicas? |
| **Profundidade Maxima Campea** | Valor otimizado do hiperparametro `max_depth` | O algoritmo restringiu o crescimento para conter sobreajuste local? |
| **Estabilidade Entre Dobras ($\sigma_{\text{CV}}$)** | Desvio-padrao dos scores das 3 dobras na configuracao vencedora | A configuracao campea e uniforme ou sofre de instabilidade amostral? |

### Interpretacao clinica e de negocio

A realizacao da otimizacao bayesiana na fase final estabelece tres garantias essenciais:

1. **Prevenir subestimacao do potencial enxuto:** comparar um modelo baseline com parâmetros profundos contra um modelo reduzido sem calibracao constituiria desvantagem metodologica indevida para o modelo enxuto. O tuning assegura comparacao em igualdade de condicoes operacionais.
2. **Mitigacao de Falsos Negativos via regularizacao:** controlar `max_depth` e `min_samples_leaf` impede que folhas minusculas memorizem casos atipicos, forcando a floresta a generalizar as regras principais da patologia para pacientes novos.
3. **Eficiencia energetica e de infraestrutura:** o Optuna atinge patamares otimos com apenas 15 trials direcionados, demandando fracao minima do poder de processamento requerido por buscas exaustivas em nuvem.

---

## Subcamada 12.6: Checkpoint de autonomia e fixacao ativa

Explique sem consultar o texto e depois confira sua resposta:

1. **Qual e a distincao conceitual entre parametros internos e hiperparametros em um modelo de aprendizado de maquina?**
2. **Por que e metodologicamente obrigatorio re-otimizar os hiperparametros apos a reducao do espaco de atributos de 40 para 10 colunas?**
3. **Como o estimador TPE do Optuna utiliza o historico das tentativas anteriores para selecionar as proximas configuracoes?**
4. **Qual e o papel da validacao cruzada estratificada em 3 dobras dentro da funcao objetivo?**
5. **Por que o conjunto de teste independente (`X_test`) nao pode ser avaliado durante a execucao dos trials do Optuna?**
6. **O que ocorreria com o risco de overfitting se mantivessemos `max_depth=None` em uma base com poucas variaveis e altamente informativas?**

### Mini-desafio pratico

Execute a otimizacao testando diferentes quantidades de trials e complete o quadro de controle de convergencia:

| Tentativas (*Trials*) | Tempo de Busca (s) | Melhor $F_1$ no Treino (CV) | $F_1$ Final no Teste Cego |
| :--- | :--- | :--- | :--- |
| `n_trials = 5` | | | |
| `n_trials = 15` (Padrao) | | | |
| `n_trials = 30` | | | |

**Pergunta reflexiva:** o ganho percentual de $F_1$ alcancado ao dobrar os trials de 15 para 30 compensou o aumento proporcional do tempo computacional consumido?
