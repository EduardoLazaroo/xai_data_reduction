# Camada 04: Alta Dimensionalidade e o Fenomeno de Hughes

**Trilha:** XAI Aplicada a Reducao de Dados em Machine Learning  
**Aplicacao:** classificacao binaria de saude (`0 = Saudavel`, `1 = Patologia`)  
**Codigo de referencia:** [pipeline_completo.py](../pipeline_completo.py), funcao `gerar_dataset_sintetico_saude`

> **Objetivo da aula:** compreender a dinamica geometrica do mal da dimensionalidade, distinguir formalmente atributos informativos, redundantes e ruidos puros, e entender o Fenomeno de Hughes, onde adicionar variaveis alem de um ponto otimo comeca a prejudicar a generalizacao.

## Campo Didatico: Enxergar Dimensoes Como Espaco

Construa a intuicao em ordem: compare uma nuvem em 2D, adicione atributos informativos, adicione redundancias e por fim injete ruido. Em cada cenario registre o numero de colunas, a densidade dos dados, o F1 de teste e o gap. A pergunta-guia e: **qual atributo acrescenta sinal e qual apenas aumenta o espaco para coincidencias?**

```text
2D compreensivel -> sinais uteis -> clones -> ruido
       |                |          |       |
       v                v          v       v
   fronteira       ganho real   repeticao  Hughes
```

Observe o ponto em que mais atributos deixam de melhorar o teste. O erro comum e defender “mais dados” sem distinguir mais pacientes de mais colunas. A ponte para a Camada 05 e natural: se o modelo sofre com excesso de dimensao, precisamos perguntar quais variaveis realmente influenciam suas decisoes.

## Mapa da aula

1. [Subcamada 4.1: O conceito na vida real](#subcamada-41-o-conceito-na-vida-real)
2. [Subcamada 4.2: Desenhando o conceito](#subcamada-42-desenhando-o-conceito)
3. [Subcamada 4.3: Desmistificando a teoria e a notacao formal](#subcamada-43-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 4.4: Laboratorio ludico no Colab](#subcamada-44-laboratorio-ludico-no-colab)
5. [Subcamada 4.5: O momento serio da nossa aplicacao](#subcamada-45-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 4.6: Checkpoint de autonomia e fixacao ativa](#subcamada-46-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 4.1: O Conceito na Vida Real

### A analogia da mochila do montanhista

Imagine um montanhista se preparando para escalar uma serra ingreme:

- Se ele nao levar nada na mochila, ficara desidratado e passara frio (subajuste).
- Se ele levar exatamente os 8 itens essenciais (agua, agasalho, mapa, corda, lanterna), subira com leveza e energia (ponto otimo).
- Se ele resolver colocar na mochila 40 itens — incluindo panelas de ferro, livros velhos e pedras que achou no chao (ruido puro) —, suas costas doerao, ele andara devagar e desabara de exaustao antes do topo.

Em ciencia de dados, existe o mito de que "quanto mais colunas dermos para o modelo, mais inteligente ele ficara". Na realidade, colunas inuteis funcionam como pedras na mochila do algoritmo: aumentam a memoria, tornam o treino lento e criam correlacoes acidentais.

**A grande sacada:** coletar dados tem custo financeiro e computacional. Excesso de variaveis sem sinal util degrada a capacidade do modelo em vez de expandi-la.

### A taxonomia dos atributos clinicos

| Tipo | Definicao pratica | Exemplo na saude | Efeito no algoritmo |
|---|---|---|---|
| Informativo | Relacao de causa ou forte associacao com o alvo | Troponina elevada em infarto agudo | Fornece ganho real de informacao nos cortes |
| Redundante | Informacao valida, mas repetida por outra coluna | Glicemia em mg/dL e em g/L; IMC e peso/altura | Consome tempo e dilui a importancia dos atributos |
| Ruido Puro | Distribuicao aleatoria independente do diagnostico | Numero do calcado do paciente; ruido de sensor | Cria ilhas de memorizacao e facilita overfitting |

---

## Subcamada 4.2: Desenhando o Conceito

### O vazio dimensional (esparsidade)

```text
1 DIMENSAO (Linha)          2 DIMENSOES (Plano)           3 DIMENSOES (Cubo)
  *---*---*---*               +-------+                     +-------+
  10 pontos preenchem         | *   * |                    / *   * /|
  a linha com folga!          |   *   |                   +-------+ |
                              +-------+                   | *   * | +
                              10^2 = 100 pontos           |   *   |/
                              para cobrir o plano.        10^3 = 1.000 pontos.

Em 40 dimensoes, seriam necessarios 10^40 pacientes para manter a mesma densidade!
Com apenas 2.000 pacientes, o espaco de 40 atributos e quase 100% puro vácuo.
```

### A curva do Fenomeno de Hughes

```text
Desempenho no Teste
      ^
      |                ★ PICO OTIMO DE HUGHES
      |              /   \
      |             /     \    QUEDA POR RUIDO:
      |            /       \   o modelo se perde em correlacoes
      |           /         \  acidentais da amostra de treino
      +----------+-----------+--------------------------> Quantidade de Atributos
               Poucos       Excesso (40+)
```

---

## Subcamada 4.3: Desmistificando a Teoria e a Notacao Formal

### O volume do hipercubo unitario

Considere um espaco de $d$ atributos, onde cada atributo foi normalizado entre `0` e `1`. O volume total do espaco cresce exponencialmente com a dimensao:

$$
V = 1^d = 1
$$

Contudo, a distancia media entre dois pontos vizinhos aleatorios cresce com a raiz da dimensao:

$$
\operatorname{dist}(x, x') = \sqrt{\sum_{j=1}^{d} (x_j - x'_j)^2}
$$

Traducao simbolo por simbolo:

| Simbolo | Leitura simples |
|---|---|
| `d` | quantidade de atributos (dimensao do dataset: no projeto, `d = 40`) |
| `x_j` | valor do exame `j` para um paciente |
| `x'_j` | valor do mesmo exame para outro paciente |
| `dist` | no espaco de alta dimensao, todos os pacientes ficam distantes entre si |

Em alta dimensao, os pontos tornam-se isolados e equidistantes. Os algoritmos precisam interpolar em regioes vazias onde nao ha observacoes suficientes.

### A ordem correta evita vazamento

A remocao de atributos redundantes e ruidos deve aprender estritamente sobre a base de treino `X_train`. O teste `X_test` apenas herda a mascara de colunas selecionadas.

---

## Subcamada 4.4: Laboratorio Ludico no Colab

### Toy example: a curva de Hughes com adicao de ruido

O codigo comeca com 10 atributos informativos reais e adiciona progressivamente colunas de puro ruido gaussiano, medindo o impacto na acuracia de treino e teste.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
X_puro, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=42)

quantidades_ruido = [0, 10, 30, 60, 100]
acc_tr_lista = []
acc_te_lista = []

for n_r in quantidades_ruido:
    if n_r == 0:
        X = X_puro
    else:
        ruidos = np.random.normal(0, 1, size=(1000, n_r))
        X = np.hstack([X_puro, ruidos])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42).fit(X_tr, y_tr)
    acc_tr_lista.append(accuracy_score(y_tr, rf.predict(X_tr)))
    acc_te_lista.append(accuracy_score(y_te, rf.predict(X_te)))

plt.figure(figsize=(9, 4))
plt.plot(quantidades_ruido, acc_tr_lista, marker="o", label="Treino (memoriza)")
plt.plot(quantidades_ruido, acc_te_lista, marker="s", color="red", label="Teste (Hughes)")
plt.title("Comprovacao Experimental do Fenomeno de Hughes")
plt.xlabel("Colunas de ruido adicionadas"); plt.ylabel("Acuracia")
plt.grid(True, linestyle="--", alpha=0.5); plt.legend()
plt.tight_layout(); plt.show()
```

> **O que voce deve notar no grafico gerado:** enquanto o treino permanece elevado (o modelo continua achando regras para tudo), o teste decai conforme o volume de ruido aumenta, ilustrando o Fenomeno de Hughes.

**Mini-investigacao:** aumente `n_samples` de `1000` para `3000`. O efeito do ruido diminui? Mais amostras ajudam a diluir as coincidencias estocasticas.

---

## Subcamada 4.5: O Momento Serio da Nossa Aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

Vamos comparar a matriz completa oficial (40 atributos com 20 ruidos) contra um cenario hipotetico onde conhecemos apenas os 10 biomarcadores informativos puros.

```python
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

SEED = 42
X_raw, y = make_classification(n_samples=2000, n_features=40, n_informative=10,
    n_redundant=10, weights=[0.6, 0.4], flip_y=0.03, random_state=SEED)
nomes = ([f"biomarcador_{i+1}" for i in range(10)] +
         [f"exame_redundante_{i+1}" for i in range(10)] +
         [f"ruido_metabolico_{i+1}" for i in range(20)])
X = pd.DataFrame(X_raw, columns=nomes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)

def cronometrar(modelo, X_tr, X_te, y_tr, y_te):
    t0 = time.perf_counter(); modelo.fit(X_tr, y_tr)
    tempo_treino_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); pred = modelo.predict(X_te)
    proba = modelo.predict_proba(X_te)[:, 1]
    latencia_us = (time.perf_counter() - t0) * 1_000_000 / len(X_te)
    return {
        "atributos": X_tr.shape[1],
        "acc_teste": accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "tempo_treino_ms": tempo_treino_ms,
        "latencia_us": latencia_us
    }

base_40 = cronometrar(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
                      X_train, X_test, y_train, y_test)
enxuto_10 = cronometrar(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
                        X_train.iloc[:, :10], X_test.iloc[:, :10], y_train, y_test)

df_comp = pd.DataFrame([base_40, enxuto_10], index=["40 Atributos (Baseline)", "10 Atributos (Sem Ruido)"]).round(4)
print(df_comp)
for nome, res in [("Baseline", base_40), ("Enxuto", enxuto_10)]:
    print(f"KPI {nome}: atributos={res['atributos']}; acc_teste={res['acc_teste']:.4f}; "
          f"f1={res['f1']:.4f}; tempo_treino_ms={res['tempo_treino_ms']:.2f}; "
          f"latencia_us={res['latencia_us']:.2f}")
```

### Tabela oficial de KPIs

Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Interpretacao | O que investigar |
|---|---|---|
| Quantidade de atributos | numero de colunas submetidas ao modelo | qual e o custo financeiro de coletar cada uma? |
| Acuracia teste | taxa de acerto em pacientes novos | a remocao de colunas preservou a exatidao? |
| F1-score | equilibrio diagnostico da patologia | o modelo manteve estabilidade sem os ruidos? |
| Tempo de treino | duracao do ajuste das 100 arvores em ms | quanto tempo economizamos ao reduzir 30 colunas? |
| Latencia | tempo medio de inferencia por paciente | o tempo atende sistemas de triagem em tempo real? |

### Interpretacao clinica e de negocio

- 40 colunas exigem que o hospital colete, transporte e armazene 40 exames por paciente, aumentando filas laboratoriais e custos operacionais.
- No treinamento, o Random Forest precisa avaliar pontos de corte em subconjuntos maiores a cada no; por isso, 40 colunas demoram substancialmente mais para treinar do que 10.
- O objetivo central da nossa pesquisa sera reproduzir ou superar o desempenho diagnostico do baseline usando XAI para identificar quais sao os atributos vitais e descartar os 20 ruidos e redundancias.

---

## Subcamada 4.6: Checkpoint de Autonomia e Fixacao Ativa

Explique sem consultar o texto e depois confira sua resposta:

1. Por que dizer que "mais dados sempre melhoram o modelo" e um equivoco quando nos referimos a colunas?
2. O que afirma o Fenomeno de Hughes sobre a relacao entre dimensionalidade e acuracia de teste?
3. Qual a diferenca entre um atributo informativo e um atributo redundante?
4. Por que 2.000 pacientes em 40 dimensoes tornam o espaco matematico praticamente vazio?
5. Como colunas de puro ruido afetam o tempo de processamento de um Random Forest?
6. O que aconteceria no pronto-socorro se um sistema dependesse de 40 exames lentos para triar uma emergencia?

### Mini-desafio pratico

Execute o codigo da Subcamada 4.4 testando um cenario extremo com `quantidades_ruido = [0, 50, 150, 300]` e anote:

```text
ruidos_adicionados     acuracia_treino     acuracia_teste     gap_overfitting
0                      ...                 ...                ...
50                     ...                 ...                ...
150                    ...                 ...                ...
300                    ...                 ...                ...
```

Depois responda: **em que momento o gap de overfitting se torna clinicamente inaceitavel?**
