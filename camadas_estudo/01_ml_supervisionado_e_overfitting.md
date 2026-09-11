# Camada 01: Aprendizado Supervisionado e Overfitting

**Trilha:** XAI Aplicada à Redução de Dados em Machine Learning  
**Aplicação:** classificação binária de saúde (`0 = Saudável`, `1 = Patologia`)  
**Código de referência:** [pipeline_completo.py](../pipeline_completo.py), funções `gerar_dataset_sintetico_saude` e `train_test_split`

> **Objetivo da aula:** entender como um modelo aprende com exemplos rotulados, por que separar treino e teste é indispensável e como atributos demais, especialmente ruído, podem fazer o modelo decorar em vez de generalizar.

## Campo Didático: Roteiro Executável da Aula

O experimento segue oito movimentos: **(1) criar dados, (2) separar treino e teste, (3) construir dois modelos, (4) treinar apenas com treino, (5) medir treino e teste, (6) desenhar as fronteiras, (7) calcular o gap e a matriz de confusão, (8) traduzir a diferença para a linguagem comum**.

```text
dados rotulados -> split protegido -> modelo simples/complexo
       |                 |                    |
       v                 v                    v
   qual e o sinal?   o teste ficou oculto?  decorou ou aprendeu?
                              |
                              v
                   treino vs teste -> generalização
```

O resultado deve ser lido por três evidências: a fronteira do modelo, a distância entre acurácia de treino e teste e os erros clínicos `FN/FP`. Um erro comum é chamar qualquer modelo com treino alto de excelente; o critério correto é verificar se o desempenho se sustenta em dados nunca vistos. A ponte para a Camada 02 é o baseline: depois de entender o problema, a floresta de árvores aparece como uma alternativa mais estável que uma árvore isolada.

### Roteiro de domínio

Ao final, ficam definidos cinco pontos: **(a)** amostra, atributo, alvo e previsão; **(b)** separação entre treino e teste; **(c)** subajuste, boa generalização e sobreajuste; **(d)** significado clínico de `FN` e `FP`; e **(e)** coincidências falsas produzidas por colunas em excesso.

Três previsões orientam a leitura: o modelo complexo deve ter maior acurácia de treino, provavelmente terá maior gap e deverá criar ilhas na fronteira. Se o resultado contrariar alguma previsão, isso não é fracasso; é evidência sobre a amostra, o ruído e a complexidade escolhida.

### Dúvidas que esta aula responde

- **Treino e teste podem ter pacientes diferentes?** Devem ter, quando o objetivo é estimar comportamento em pacientes novos.
- **Treino perfeito sempre é ruim?** Não. Ele vira alerta quando o teste cai, quando a amostra é pequena ou quando houve vazamento.
- **Ruído é o mesmo que erro de medição?** Não necessariamente: aqui ruído é uma variável sem sinal preditivo; erro de medição pode contaminar uma variável que era útil.
- **Um split resolve tudo?** Não. Ele é uma primeira proteção; validação cruzada, repetição de sementes e validação externa aumentam a confiança.

### Regra de explicação Feynman

Treino é a lista de exercícios; teste é a prova surpresa. Overfitting é decorar a lista. Generalização é conseguir resolver uma questão nova. O `gap` é a distância entre essas duas situações.

## Cultura, História e Referências

O problema “treinar no conhecido e funcionar no novo” é anterior ao nome Machine Learning. Em estatística, ele aparece como generalização; em engenharia, como teste fora da amostra; na cultura de competições, como a diferença entre decorar o conjunto de treino e sobreviver ao leaderboard. A documentação de [validação cruzada do scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) explica por que o teste deve permanecer reservado.

Leia também a [proposta de Dartmouth de 1956](http://www-formal.stanford.edu/jmc/history/dartmouth/dartmouth.html) para perceber que “aprender” sempre foi uma hipótese sobre comportamento, não apenas uma chamada de biblioteca. No projeto, a figura abaixo transforma essa história em evidência observável: um modelo pode parecer forte e ainda falhar em pacientes nunca vistos.

![Matriz de confusão e curva ROC do baseline](../assets/modulo1_baseline_metrics.png)

**Pergunta cultural:** o que a cultura de “100% no treino” recompensa? Muitas vezes, recompensa exibição de número, não aprendizado. A maturidade profissional começa quando se pergunta “em quais casos novos isso pode falhar?”.

## Recursos de Mídia (Visual e Áudio)

- **Visual local:** [matriz de confusão e curva ROC](../assets/modulo1_baseline_metrics.png).
- **Visual interativo:** [curvas de aprendizado do scikit-learn](https://scikit-learn.org/stable/modules/learning_curve.html), mostrando treino e validação se separarem.
- **Áudio sugerido:** a história “lista de exercícios versus prova surpresa”, em contraste com o som de um gabarito sendo decorado.
- **Imagem mental:** duas fronteiras no mesmo plano: uma suave, outra tentando abraçar cada ponto.

## 📊 Elementos de Comunidade e Status

- **Status:** `Fundamento concluído` quando treino, teste, gap e vazamento puderem ser explicados sem recorrer a definições decoradas.
- **Pergunta para discussão:** “Um modelo com 100% no treino merece parabéns ou auditoria? Em que evidência você se baseia?”
- **Conversa técnica:** uma pessoa defende o modelo, outra procura o vazamento e outra interpreta `FN` e `FP`.

## 💡 Engajamento e Conhecimento

- **Desafio relâmpago:** uma matriz de confusão diferente é calculada em acurácia, recall e erro clínico mais grave.
- **Produto da aula:** um cartão “aprendeu ou decorou?” com três evidências do gráfico e uma limitação.
- **Conexão profissional:** a saída do toy example é comparada com o baseline do projeto, mantendo o teste intocado.

## Mapa da aula

1. [Subcamada 1.1: O conceito na vida real](#subcamada-11-o-conceito-na-vida-real)
2. [Subcamada 1.2: Desenhando o conceito](#subcamada-12-desenhando-o-conceito)
3. [Subcamada 1.3: Desmistificando a teoria](#subcamada-13-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 1.4: Laboratório lúdico no Colab](#subcamada-14-laboratorio-ludico-no-colab)
5. [Subcamada 1.5: O momento sério da nossa aplicação](#subcamada-15-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 1.6: Checkpoint de autonomia](#subcamada-16-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 1.1: O Conceito na Vida Real

### A historia do aluno que decorou o gabarito

Imagine dois alunos se preparando para uma prova de física. O primeiro entende as leis por trás dos exercícios. O segundo decora que a questão 34 tem resposta `42` e que a questão 78 tem alternativa `C`.

Se a prova repetir exatamente as mesmas questões, os dois podem tirar nota alta. Mas, diante de uma questão nova, apenas o primeiro consegue raciocinar. O segundo aprendeu a lista, não aprendeu a matéria.

O aprendizado supervisionado funciona como um processo que mostra exemplos completos:

- **pistas:** exames, sensores e medidas;
- **gabarito:** o diagnóstico já confirmado;
- **tarefa:** descobrir uma regra que funcione também para um paciente novo.

Neste projeto, o modelo recebe exames e aprende a responder: `0`, saudável, ou `1`, patologia.

**A grande sacada:** acertar exemplos conhecidos não é a meta final. A meta é acertar casos que ainda não foram vistos.

### Classificacao e regressao: duas perguntas diferentes

| Tipo | Pergunta | Resposta | Exemplo medico |
|---|---|---|---|
| Regressão | Quanto? | número contínuo | glicose estimada: `112,4 mg/dL` |
| Classificação | Qual grupo? | categoria ou classe | saudável `0` ou patologia `1` |

O projeto usa **classificação binária**. Não estamos prevendo uma quantidade; estamos escolhendo entre dois rótulos.

---

## Subcamada 1.2: Desenhando o Conceito

### O modelo como detetive

```text
             PISTAS OBSERVADAS                         VEREDITO CONHECIDO
        exames e biomarcadores X                         diagnóstico y

  Paciente A: [glicose, idade, IMC, ...]  ------------>  1 = patologia
    Paciente B: [glicose, idade, IMC, ...]  ------------>  0 = saudável
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
    ajusta árvores e regras         mede generalização
```

O teste é uma prova surpresa. Se o modelo consulta o teste enquanto aprende, a prova deixa de ser surpresa.

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

| Observação | Interpretação |
|---|---|
| treino baixo e teste baixo | modelo ainda não aprendeu o padrão: subajuste |
| treino alto e teste parecido | boa generalização |
| treino quase perfeito e teste bem menor | overfitting |

### Por que o ruído engana?

```text
10 sinais úteis + 20 colunas de ruído
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

Quanto mais colunas sem relação real, mais oportunidades existem para uma coincidência parecer uma descoberta.

![Fronteira de decisão: modelo controlado e modelo complexo](../assets/modulo1_baseline_metrics.png)

---

## Subcamada 1.3: Desmistificando a Teoria e a Notação Formal

### Dados, modelo e previsão

Depois de enxergar a história, podemos nomear as peças. Pense em uma planilha: cada linha é um paciente, cada coluna é uma pista.

- `X`: matriz de atributos, as pistas observadas;
- `y`: vetor de alvos, o gabarito ou diagnóstico;
- `f`: regra aprendida pelo algoritmo;
- `y_hat`: previsão produzida pela regra.

A ideia pode ser escrita assim:

$$
\hat{y} = f(X)
$$

Tradução símbolo por símbolo:

| Símbolo | Leitura simples |
|---|---|
| `X` | os exames entregues ao modelo |
| `f` | a regra que o modelo aprendeu |
| `y_hat` | a resposta que o modelo previu |

No treinamento, o algoritmo procura uma regra que erre pouco nos exemplos conhecidos. O desafio é escolher uma regra que também funcione fora deles.

### Acurácia e gap de generalização

A acurácia responde: entre todas as previsões, quantas estavam corretas?

$$
\mathrm{Acurácia} = \frac{\text{previsões corretas}}{\text{total de casos}}
$$

O **gap de overfitting** compara o desempenho no treino com o desempenho no teste:

$$
\mathrm{Gap} = \mathrm{Acurácia}_{treino} - \mathrm{Acurácia}_{teste}
$$

Gap grande não é uma prova isolada de que o modelo é inútil, mas é um alerta para investigar complexidade, vazamento, tamanho da amostra e ruído.

### A matriz de confusão

Para diagnóstico, nem todo erro tem o mesmo custo:

| | Real saudável (`0`) | Real patologia (`1`) |
|---|---:|---:|
| Previsto saudável (`0`) | TN: acerto | FN: deixou passar uma patologia |
| Previsto patologia (`1`) | FP: alarme falso | TP: acerto |

O recall da classe patológica é:

$$
\mathrm{Recall} = \frac{TP}{TP + FN}
$$

Em um cenário clínico, acompanhar `FN` e recall é essencial, porque um caso doente classificado como saudável pode atrasar o cuidado. Por isso, a acurácia nunca deve ser a única régua.

### Data leakage: quando a prova vaza

O vazamento acontece quando uma informação do teste influencia o aprendizado. Exemplos:

- selecionar atributos usando todos os pacientes antes do split;
- calcular média e desvio com treino e teste juntos;
- ajustar limiares olhando a resposta dos pacientes reservados.

A ordem correta é sempre:

```text
separar ---> aprender transformacoes no treino ---> aplicar no teste ---> avaliar
```

---

## Subcamada 1.4: Laboratório Lúdico no Colab

### Toy example: uma fronteira simples contra uma fronteira decoradora

O código usa 120 pontos em duas dimensões. Uma árvore rasa aceita algum ruído para manter uma regra simples; uma árvore profunda cria regiões pequenas para memorizar a amostra.

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

> **O que aparece no gráfico:** a árvore profunda tende a desenhar ilhas e recortes para capturar pontos individuais. A diferença entre acurácia de treino e teste revela quando a fronteira aprendeu detalhes da amostra em vez de uma regra geral.

**Mini-experimento:** altere `max_depth=3` para `1`, `5` e `None`. Registre treino, teste e gap. Procure o ponto em que aumentar a complexidade deixa de melhorar o teste.

---

## Subcamada 1.5: O Momento Sério da Nossa Aplicação

> **Chega de brinquedo!** Agora que o conceito está cristalino, vamos para a trincheira real da nossa aplicação com os dados do projeto.

Usaremos a matriz oficial sintética: 2.000 pacientes, 40 atributos, 10 informativos, 10 redundantes e 20 ruídos metabólicos. O Random Forest terá 100 árvores. O código separa treino e teste antes de medir os KPIs e deixa os valores serem calculados no ambiente de execução.

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

Os valores abaixo são produzidos pelo código, não devem ser decorados como constantes. Tempo, latência e até pequenas variações de desempenho dependem do ambiente e da versão das bibliotecas.

| KPI | Interpretação | O que investigar |
|---|---|---|
| Acurácia treino | desempenho nos casos estudados | o modelo conseguiu aprender? |
| Acurácia teste | desempenho em casos reservados | ele generaliza? |
| Gap de overfitting | treino menos teste | a diferença é aceitável? |
| F1-score | equilíbrio entre precision e recall | o desempenho da classe positiva é consistente? |
| Precision | proporção de alertas corretos | quantos alarmes são falsos? |
| Recall | proporção de patologias encontradas | quantos casos foram perdidos? |
| ROC-AUC | qualidade do ranking de risco | o modelo separa classes em vários limiares? |
| Latência por paciente | tempo médio de inferência em us | cabe no fluxo operacional? |
| Tempo de treino | duração do ajuste em ms | o retreinamento é viável? |
| TN, FP, FN, TP | tipos de acerto e erro | qual erro tem maior custo clínico? |

### Interpretação clínica e de negócio

- Treino muito alto e teste bem menor sugerem que as 20 colunas de ruído oferecem oportunidades de coincidência para as árvores.
- `FN` é o caso mais delicado desta aplicação: o paciente tem patologia, mas recebe previsão `0`. A decisão de negócio deve considerar o custo desse erro, não apenas a acurácia.
- Reduzir exames pode economizar coleta, armazenamento e tempo de processamento, mas o valor financeiro precisa ser calculado com custos reais do serviço de saúde.
- A Camada 01 estabelece o baseline. As próximas camadas devem provar, com a mesma separação e as mesmas métricas, se SHAP, filtros e seleção conseguem reduzir atributos sem piorar a generalização.

---

## Subcamada 1.6: Checkpoint de Autonomia e Fixação Ativa

Responda às perguntas e compare as justificativas com os conceitos apresentados:

1. Por que uma acurácia de treino muito alta pode ser um alerta em vez de uma vitória?
2. Qual é a diferença entre regressão e classificação?
3. Como uma coluna de ruído pode parecer útil no treino por pura coincidência?
4. O que é data leakage e por que o teste precisa permanecer escondido?
5. Por que `FN` pode ser mais importante que acurácia em uma triagem de saúde?
6. Qual é a diferença entre aprender um padrão e memorizar exemplos?

### Mini-desafio prático

O experimento sério pode comparar três cenários, mantendo a mesma semente e o mesmo split:

```text
cenario              atributos       acuracia_treino  acuracia_teste  gap  F1  recall  FN
40 atributos         40              ...               ...             ...  ... ...     ...
10 informativos      10              ...               ...             ...  ... ...     ...
40 sem ruído puro    20              ...               ...             ...  ... ...     ...
```

No cenário de 10 atributos, use `n_features=10`, `n_informative=10` e `n_redundant=0`. No cenário sem ruído puro, mantenha os 10 informativos e 10 redundantes. A síntese final deve responder: **qual mudança reduziu o gap sem aumentar o número de falsos negativos?**
