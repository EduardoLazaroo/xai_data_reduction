# Camada 01: Aprendizado Supervisionado e Overfitting

**Trilha:** XAI Aplicada a Reducao de Dados em Machine Learning  
**Aplicacao:** classificacao binaria de saude (`0 = Saudavel`, `1 = Patologia`)  
**Codigo de referencia:** [pipeline_completo.py](../pipeline_completo.py), funcoes `gerar_dataset_sintetico_saude` e `train_test_split`

> **Objetivo da aula:** entender como um modelo aprende com exemplos rotulados, por que separar treino e teste e indispensavel e como atributos demais, especialmente ruido, podem fazer o modelo decorar em vez de generalizar.

## Campo Didatico: Roteiro Executavel da Aula

Estude esta aula como uma sequencia de oito celulas mentais: **(1) criar dados, (2) separar treino e teste, (3) construir dois modelos, (4) treinar apenas com treino, (5) medir treino e teste, (6) desenhar as fronteiras, (7) calcular o gap e a matriz de confusao, (8) explicar a diferenca em linguagem comum**. Antes de rodar qualquer codigo, escreva o que espera observar.

```text
dados rotulados -> split protegido -> modelo simples/complexo
       |                 |                    |
       v                 v                    v
   qual e o sinal?   o teste ficou oculto?  decorou ou aprendeu?
                              |
                              v
                   treino vs teste -> generalizacao
```

Ao executar, observe tres evidencias: a fronteira do modelo, a distancia entre acuracia de treino e teste e os erros clinicos `FN/FP`. Um erro comum e chamar qualquer modelo com treino alto de excelente; o criterio correto e perguntar se o desempenho se sustenta em dados nunca vistos. A ponte para a Camada 02 e o baseline: depois de entender o problema, veremos por que uma floresta de arvores costuma ser mais estavel que uma arvore isolada.

### Roteiro de dominio

Ao terminar, voce deve conseguir explicar sem codigo: **(a)** o que e uma amostra, atributo, alvo e previsao; **(b)** por que treino e teste precisam ser separados; **(c)** como reconhecer subajuste, boa generalizacao e sobreajuste; **(d)** por que `FN` e `FP` nao sao apenas numeros; e **(e)** por que mais colunas podem criar coincidencias falsas.

Antes de rodar, faca tres previsoes: qual modelo tera maior acuracia de treino, qual tera maior gap e que tipo de ponto aparecera nas ilhas da fronteira. Depois compare previsao e resultado. Se forem diferentes, isso nao e fracasso: e justamente a evidencia que o experimento foi desenhado para revelar.

### Duvidas que esta aula responde

- **Treino e teste podem ter pacientes diferentes?** Devem ter, quando o objetivo e estimar comportamento em pacientes novos.
- **Treino perfeito sempre e ruim?** Nao. Ele vira alerta quando o teste cai, quando a amostra e pequena ou quando houve vazamento.
- **Ruido e o mesmo que erro de medicao?** Nao necessariamente: aqui ruido e uma variavel sem sinal preditivo; erro de medicao pode contaminar uma variavel que era util.
- **Um split resolve tudo?** Nao. Ele e uma primeira protecao; validacao cruzada, repeticao de sementes e validacao externa aumentam a confianca.

### Regra de explicacao Feynman

Explique assim: “Treino e a lista de exercicios; teste e a prova surpresa. Overfitting e decorar a lista. Generalizacao e conseguir resolver uma questao nova”. Se voce nao consegue explicar o `gap` usando essa historia, volte ao grafico antes de avancar.

## Mapa da aula

1. [Subcamada 1.1: O conceito na vida real](#subcamada-11-o-conceito-na-vida-real)
2. [Subcamada 1.2: Desenhando o conceito](#subcamada-12-desenhando-o-conceito)
3. [Subcamada 1.3: Desmistificando a teoria](#subcamada-13-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 1.4: Laboratorio ludico no Colab](#subcamada-14-laboratorio-ludico-no-colab)
5. [Subcamada 1.5: O momento serio da nossa aplicacao](#subcamada-15-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 1.6: Checkpoint de autonomia](#subcamada-16-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 1.1: O Conceito na Vida Real

### A historia do aluno que decorou o gabarito

Imagine dois alunos se preparando para uma prova de fisica. O primeiro entende as leis por tras dos exercicios. O segundo decora que a questao 34 tem resposta `42` e que a questao 78 tem alternativa `C`.

Se a prova repetir exatamente as mesmas questoes, os dois podem tirar nota alta. Mas, diante de uma questao nova, apenas o primeiro consegue raciocinar. O segundo aprendeu a lista, nao aprendeu a materia.

O aprendizado supervisionado funciona como um professor que mostra exemplos completos:

- **pistas:** exames, sensores e medidas;
- **gabarito:** o diagnostico ja confirmado;
- **tarefa:** descobrir uma regra que funcione tambem para um paciente novo.

Neste projeto, o modelo recebe exames e aprende a responder: `0`, saudavel, ou `1`, patologia.

**A grande sacada:** acertar exemplos conhecidos nao e a meta final. A meta e acertar casos que ainda nao foram vistos.

### Classificacao e regressao: duas perguntas diferentes

| Tipo | Pergunta | Resposta | Exemplo medico |
|---|---|---|---|
| Regressao | Quanto? | numero continuo | glicose estimada: `112,4 mg/dL` |
| Classificacao | Qual grupo? | categoria ou classe | saudavel `0` ou patologia `1` |

O projeto usa **classificacao binaria**. Nao estamos prevendo uma quantidade; estamos escolhendo entre dois rotulos.

---

## Subcamada 1.2: Desenhando o Conceito

### O modelo como detetive

```text
             PISTAS OBSERVADAS                         VEREDITO CONHECIDO
        exames e biomarcadores X                         diagnostico y

  Paciente A: [glicose, idade, IMC, ...]  ------------>  1 = patologia
  Paciente B: [glicose, idade, IMC, ...]  ------------>  0 = saudavel
  Paciente C: [glicose, idade, IMC, ...]  ------------>  1 = patologia

                         durante o treinamento
                                      |
                                      v
                         [ modelo aprende uma regra ]
                                      |
                                      v
               novo paciente ---> [ regra ] ---> 0 ou 1
```

### A separacao sagrada

```text
                         2.000 pacientes
                                |
                +---------------+---------------+
                |                               |
                v                               v
       TREINO: 80% ou 75%              TESTE: parte reservada
       o modelo pode estudar           o modelo nunca estudou
                |                               |
                v                               v
       ajusta arvores e regras         mede generalizacao
```

O teste e uma prova surpresa. Se o modelo consulta o teste enquanto aprende, a prova deixa de ser surpresa.

### O termometro do overfitting

```text
Desempenho
100% | treino  _____________
     |       /               \       modelo decorando
     |      /                 \
     |     / teste             \____ teste cai
     +------------------------------------------------> complexidade
                 suficiente              excessiva

Treino alto + teste muito menor = sinal de sobreajuste
```

| Observacao | Interpretacao |
|---|---|
| treino baixo e teste baixo | modelo ainda nao aprendeu o padrao: subajuste |
| treino alto e teste parecido | boa generalizacao |
| treino quase perfeito e teste bem menor | overfitting |

### Por que o ruido engana?

```text
10 sinais uteis + 20 colunas de ruido
                  |
                  v
     algumas coincidencias aparecem no treino
                  |
                  v
     a arvore confunde coincidencia com regra
                  |
                  v
     no teste, a coincidencia desaparece
```

Quanto mais colunas sem relacao real, mais oportunidades existem para uma coincidencia parecer uma descoberta.

---

## Subcamada 1.3: Desmistificando a Teoria e a Notacao Formal

### Dados, modelo e previsao

Depois de enxergar a historia, podemos nomear as pecas. Pense em uma planilha: cada linha e um paciente, cada coluna e uma pista.

- `X`: matriz de atributos, as pistas observadas;
- `y`: vetor de alvos, o gabarito ou diagnostico;
- `f`: regra aprendida pelo algoritmo;
- `y_hat`: previsao produzida pela regra.

A ideia pode ser escrita assim:

$$
\hat{y} = f(X)
$$

Traducao simbolo por simbolo:

| Simbolo | Leitura simples |
|---|---|
| `X` | os exames entregues ao modelo |
| `f` | a regra que o modelo aprendeu |
| `y_hat` | a resposta que o modelo previu |

No treinamento, o algoritmo procura uma regra que erre pouco nos exemplos conhecidos. O desafio e escolher uma regra que tambem funcione fora deles.

### Acuracia e gap de generalizacao

A acuracia responde: entre todas as previsoes, quantas estavam corretas?

$$
\operatorname{Acuracia} = \frac{\text{previsoes corretas}}{\text{total de casos}}
$$

O **gap de overfitting** compara o desempenho no treino com o desempenho no teste:

$$
\text{Gap} = \operatorname{Acuracia}_{treino} - \operatorname{Acuracia}_{teste}
$$

Gap grande nao e uma prova isolada de que o modelo e inutil, mas e um alerta para investigar complexidade, vazamento, tamanho da amostra e ruido.

### A matriz de confusao

Para diagnostico, nem todo erro tem o mesmo custo:

| | Real saudavel (`0`) | Real patologia (`1`) |
|---|---:|---:|
| Previsto saudavel (`0`) | TN: acerto | FN: deixou passar uma patologia |
| Previsto patologia (`1`) | FP: alarme falso | TP: acerto |

O recall da classe patologica e:

$$
\operatorname{Recall} = \frac{TP}{TP + FN}
$$

Em um cenario clinico, acompanhar `FN` e recall e essencial, porque um caso doente classificado como saudavel pode atrasar o cuidado. Por isso, a acuracia nunca deve ser a unica regua.

### Data leakage: quando a prova vaza

O vazamento acontece quando uma informacao do teste influencia o aprendizado. Exemplos:

- selecionar atributos usando todos os pacientes antes do split;
- calcular media e desvio com treino e teste juntos;
- ajustar limiares olhando a resposta dos pacientes reservados.

A ordem correta e sempre:

```text
separar ---> aprender transformacoes no treino ---> aplicar no teste ---> avaliar
```

---

## Subcamada 1.4: Laboratorio Ludico no Colab

### Toy example: uma fronteira simples contra uma fronteira decoradora

O codigo usa 120 pontos em duas dimensoes. Uma arvore rasa aceita algum ruido para manter uma regra simples; uma arvore profunda cria regioes pequenas para memorizar a amostra.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=120, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)
modelos = {"Regra controlada": DecisionTreeClassifier(max_depth=3, random_state=42),
           "Decorador": DecisionTreeClassifier(max_depth=None, random_state=42)}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (nome, modelo) in zip(axes, modelos.items()):
    modelo.fit(X_train, y_train)
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 250), np.linspace(-1, 1.5, 250))
    z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolor="k", label="treino")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", marker="*", s=70, label="teste")
    ax.set_title(f"{nome}\nTreino: {accuracy_score(y_train, modelo.predict(X_train)):.2f} | Teste: {accuracy_score(y_test, modelo.predict(X_test)):.2f}")
    ax.legend()
plt.tight_layout(); plt.show()
```

> **O que voce deve notar no grafico:** a arvore profunda tende a desenhar ilhas e recortes para capturar pontos individuais. Compare a acuracia de treino com a de teste: quando a primeira fica muito acima da segunda, a fronteira aprendeu detalhes da amostra em vez de uma regra geral.

**Mini-experimento:** mude `max_depth=3` para `1`, `5` e `None`. Registre treino, teste e gap. Procure o ponto em que aumentar a complexidade deixa de melhorar o teste.

---

## Subcamada 1.5: O Momento Serio da Nossa Aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

Usaremos a matriz oficial sintetica: 2.000 pacientes, 40 atributos, 10 informativos, 10 redundantes e 20 ruidos metabolicos. O Random Forest tera 100 arvores. O codigo separa treino e teste antes de medir os KPIs e deixa os valores serem calculados na sua propria maquina.

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

modelo = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
inicio = time.perf_counter(); modelo.fit(X_train, y_train)
treino_ms = (time.perf_counter() - inicio) * 1000
inicio = time.perf_counter(); pred = modelo.predict(X_test)
proba = modelo.predict_proba(X_test)[:, 1]
inferencia_ms = (time.perf_counter() - inicio) * 1000
pred_treino = modelo.predict(X_train)
acc_treino = accuracy_score(y_train, pred_treino)
acc_teste = accuracy_score(y_test, pred)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
resultado = {
    "acuracia_treino": acc_treino, "acuracia_teste": acc_teste,
    "gap_overfitting": acc_treino - acc_teste,
    "f1": f1_score(y_test, pred), "precision": precision_score(y_test, pred),
    "recall": recall_score(y_test, pred), "roc_auc": roc_auc_score(y_test, proba),
    "tempo_treino_ms": treino_ms,
    "latencia_us_paciente": inferencia_ms * 1000 / len(X_test),
    "TN": tn, "FP": fp, "FN": fn, "TP": tp
}
print(f"Dimensao: {X.shape[0]} pacientes x {X.shape[1]} atributos")
for nome, valor in resultado.items():
    print(f"KPI {nome}: {valor:.4f}" if isinstance(valor, float) else f"KPI {nome}: {valor}")
```

### Tabela oficial de KPIs

Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Interpretacao | O que investigar |
|---|---|---|
| Acuracia treino | desempenho nos casos estudados | o modelo conseguiu aprender? |
| Acuracia teste | desempenho em casos reservados | ele generaliza? |
| Gap de overfitting | treino menos teste | a diferenca e aceitavel? |
| F1-score | equilibrio entre precision e recall | o desempenho da classe positiva e consistente? |
| Precision | proporcao de alertas corretos | quantos alarmes sao falsos? |
| Recall | proporcao de patologias encontradas | quantos casos foram perdidos? |
| ROC-AUC | qualidade do ranking de risco | o modelo separa classes em varios limiares? |
| Latencia por paciente | tempo medio de inferencia em us | cabe no fluxo operacional? |
| Tempo de treino | duracao do ajuste em ms | o retreinamento e viavel? |
| TN, FP, FN, TP | tipos de acerto e erro | qual erro tem maior custo clinico? |

### Interpretacao clinica e de negocio

- Treino muito alto e teste bem menor sugerem que as 20 colunas de ruido oferecem oportunidades de coincidencia para as arvores.
- `FN` e o caso mais delicado desta aplicacao: o paciente tem patologia, mas recebe previsao `0`. A decisao de negocio deve considerar o custo desse erro, nao apenas a acuracia.
- Reduzir exames pode economizar coleta, armazenamento e tempo de processamento, mas o valor financeiro precisa ser calculado com custos reais do servico de saude.
- A Camada 01 estabelece o baseline. As proximas camadas devem provar, com a mesma separacao e as mesmas metricas, se SHAP, filtros e selecao conseguem reduzir atributos sem piorar a generalizacao.

---

## Subcamada 1.6: Checkpoint de Autonomia e Fixacao Ativa

Explique sem consultar o texto e depois confira sua resposta:

1. Por que uma acuracia de treino muito alta pode ser um alerta em vez de uma vitoria?
2. Explique para uma pessoa leiga a diferenca entre regressao e classificacao.
3. Como uma coluna de ruido pode parecer util no treino por pura coincidencia?
4. O que e data leakage e por que o teste precisa permanecer escondido?
5. Por que `FN` pode ser mais importante que acuracia em uma triagem de saude?
6. Qual e a diferenca entre aprender um padrao e memorizar exemplos?

### Mini-desafio pratico

Altere o experimento serio para comparar tres cenarios, mantendo a mesma semente e o mesmo split:

```text
cenario              atributos       acuracia_treino  acuracia_teste  gap  F1  recall  FN
40 atributos         40              ...               ...             ...  ... ...     ...
10 informativos      10              ...               ...             ...  ... ...     ...
40 sem ruido puro    20              ...               ...             ...  ... ...     ...
```

Para o cenario de 10 atributos, gere a base com `n_features=10`, `n_informative=10` e `n_redundant=0`. Para o cenario sem ruido puro, mantenha os 10 informativos e 10 redundantes. Depois escreva uma explicacao Feynman: **qual mudanca reduziu o gap sem aumentar o numero de falsos negativos?**
