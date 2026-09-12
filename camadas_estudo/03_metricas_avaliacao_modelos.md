# Camada 03: Metricas de Avaliacao e Rigor Diagnostico

**Trilha:** XAI Aplicada à Redução de Dados em Machine Learning  
**Aplicação:** classificação binária de saúde (`0 = Saudável`, `1 = Patologia`)  
**Código de referência:** [pipeline_completo.py](../pipeline_completo.py), funções `treinar_e_avaliar_modelo` e `plotar_graficos_baseline`

> **Objetivo da aula:** compreender por que a acuracia isolada pode mascarar erros graves em dados de saude, dominar a anatomia da matriz de confusao e entender como o equilibrio entre precision, recall, F1-score e ROC-AUC fundamenta a validacao clinica de modelos de aprendizado de maquina.

## Campo Didático: Uma Métrica e uma Pergunta

Nao comece pela formula. Comece por um caso: um paciente doente que recebe resultado saudavel e um paciente saudavel que recebe um alarme. Depois execute a avaliacao em quatro passagens: **matriz de confusao, precision, recall, F1 e ROC-AUC**. Para cada numero, escreva a pergunta operacional que ele responde e o erro que ele pode esconder.

```text
previsoes -> TN/FP/FN/TP -> metricas complementares -> limiar de decisao -> escolha
```

Observe especialmente `FN`, porque ele representa a falha clinica mais sensivel nesta aplicacao. O erro comum e escolher o modelo pela maior acuracia sem olhar o desbalanceamento ou o custo dos erros. A ponte para a Camada 04 e metodologica: depois de saber medir, investigaremos como o excesso de atributos altera essas metricas.

### Roteiro de dominio

Comece pela matriz de confusao e so depois calcule metricas. Para cada formula, complete a frase: “o denominador representa...”. Em seguida altere o limiar de decisao e observe o cabo de guerra entre precision e recall. Por fim, compare os modelos com a mesma particao e o mesmo criterio.

### Duvidas que esta aula responde

- **Acuracia alta prova um bom diagnostico?** Nao se a classe majoritaria dominar ou se `FN` for caro.
- **F1 substitui tudo?** Nao. Ele resume precision e recall, mas nao mostra custo, calibracao ou distribuicao dos erros.
- **ROC-AUC alto garante bom limiar clinico?** Nao. AUC avalia ordenacao em varios limiares; a operacao precisa escolher um corte justificado.

### Regra de explicacao Feynman

Use o alarme de incendio: `FP` e alarme sem fogo, `FN` e fogo sem alarme, `TP` e fogo detectado e `TN` e silencio correto. Depois explique por que um hospital pode preferir mais alarmes falsos a perder um incendio.

### Um caso numerico antes das formulas

Considere 100 pacientes: 80 saudáveis e 20 com patologia. Um modelo previu corretamente 76 saudáveis (`TN`), encontrou 15 doentes (`TP`), alarmou 4 saudáveis (`FP`) e deixou passar 5 doentes (`FN`).

```text
                REAL
            saudavel       patologia
previsto saudavel     76              5
previsto patologia     4             15
```

A acurácia é `(76 + 15) / 100 = 91%`. A precision é `15 / (15 + 4) = 78,9%`: entre os alertas, essa é a parcela correta. O recall é `15 / (15 + 5) = 75%`: um quarto das patologias escapou. O mesmo modelo pode parecer excelente pela acurácia e insuficiente pela lente clínica.

### Como ler o laboratório

O limiar de classificação transforma probabilidade em ação: reduzi-lo chama mais pacientes para investigação, aumentando recall e possivelmente `FP`; elevá-lo exige mais evidência, aumentando precision e possivelmente `FN`. A ROC mostra essa troca em muitos limiares; F1 resume uma troca específica, não substitui a escolha clínica.

### Dúvidas frequentes

- **Acurácia alta prova que o modelo é bom?** Não quando as classes são desbalanceadas ou os erros têm custos diferentes.
- **Recall alto é suficiente?** Não; um modelo que alerta todos tem recall alto, mas pode produzir uma quantidade impraticável de falsos positivos.
- **ROC-AUC é uma probabilidade de acerto?** Não. É uma medida de capacidade de ordenação entre positivos e negativos em vários limiares.
- **F1 deve sempre ser maximizado?** Não necessariamente; em triagem, recall pode receber prioridade explícita.

### Decisão responsável

O resultado deve ser escrito como uma tabela de erros e consequências, não como um único número vencedor. Antes de comparar modelos, ficam definidos a classe positiva, o limiar, a partição e a métrica principal. A Camada 04 usa esse rigor para observar como o excesso de atributos altera a generalização.

## Cultura, História e Referências

A curva ROC nasceu no contexto de deteccao de sinais e radares, nao de competicoes de Python. O texto de Tom Fawcett, [An introduction to ROC analysis](https://www.sciencedirect.com/science/article/pii/S016786550500303X), ajuda a entender que o limiar de decisao transforma probabilidades em acoes. Para problemas desbalanceados, leia [The Relationship Between Precision-Recall and ROC Curves](https://doi.org/10.1145/1143844.1143874), de Davis e Goadrich.

O grafico [modulo1_baseline_metrics.png](../assets/modulo1_baseline_metrics.png) deve ser lido como uma historia de decisoes: matriz de confusao mostra os custos concretos; ROC mostra separacao; F1 resume uma troca. A [documentacao de metricas do scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html) insiste que a metrica deve nascer do objetivo da aplicacao.

**Pergunta cultural:** quem decide qual erro vale mais? Nao e o algoritmo. Essa escolha pertence ao dominio, ao protocolo clinico e as pessoas que sofrerao as consequencias.

## Recursos de Mídia (Visual e Áudio)

- **Visual local:** [modulo1_baseline_metrics.png](../assets/modulo1_baseline_metrics.png), com matriz de confusão e ROC.
- **Referencia interativa:** [Metricas de avaliacao do scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html).
- **Áudio sugerido:** o caso do alarme de fumaça: alarme falso incomoda; silêncio diante do fogo é perigoso.
- **Imagem mental:** quatro quadrantes da matriz como quatro destinos clínicos diferentes.

## 📊 Elementos de Comunidade e Status

- **Status:** `Métricas defendidas` quando uma métrica puder ser justificada pela consequência de negócio ou clínica.
- **Debate:** “Qual erro deve ser priorizado no pronto-socorro: `FP` ou `FN`?”
- **Papel rotativo:** medico, gestor de custo, paciente e cientista de dados escolhem limiares diferentes.

## 💡 Engajamento e Conhecimento

- **Jogo de limiar:** probabilidades são avaliadas com cortes em `0,30`, `0,50` e `0,70`.
- **Produto da aula:** tabela com `TN`, `FP`, `FN`, `TP`, F1, recall e justificativa do limiar.
- **Conexao profissional:** consultar a documentacao antes de escolher `scoring` em uma busca de hiperparametros.

## Mapa da aula

1. [Subcamada 3.1: O conceito na vida real](#subcamada-31-o-conceito-na-vida-real)
2. [Subcamada 3.2: Desenhando o conceito](#subcamada-32-desenhando-o-conceito)
3. [Subcamada 3.3: Desmistificando a teoria e a notacao formal](#subcamada-33-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 3.4: Laboratorio ludico no Colab](#subcamada-34-laboratorio-ludico-no-colab)
5. [Subcamada 3.5: O momento serio da nossa aplicacao](#subcamada-35-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 3.6: Checkpoint de autonomia e fixacao ativa](#subcamada-36-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 3.1: O Conceito na Vida Real

### A historia do alarme de fumaca e o medico preguicoso

Imagine duas situacoes em um apartamento com sensor de fumaca:

1. O alarme dispara porque uma torrada queimou na chapa. E um susto, gera incomodo, mas ninguem se fere. Isso e um **falso positivo** (alarme falso).
2. O apartamento pega fogo de madrugada, mas a bateria do sensor acabou e ele fica em silencio. As pessoas continuam dormindo enquanto o incendio se espalha. Isso e um **falso negativo** (falha critica).

Agora pense em um hospital onde 99 de cada 100 pacientes sao saudaveis. Um medico que carimbe "Saudavel" para todo mundo sem sequer olhar os exames tera **99% de acuracia**, mas matara 100% dos pacientes que realmente precisavam de socorro.

**A grande sacada:** em problemas clinicos, os tipos de erro tem pesos humanos e financeiros completamente diferentes. Medir apenas a taxa global de acerto e perigoso.

### Metricas clinicas: quatro olhares complementares

| Metrica | Pergunta que responde | Quando priorizar? |
|---|---|---|
| Acuracia | De todas as previsoes, quantas foram certas? | Apenas em classes perfeitamente equilibradas |
| Precision | Quando o modelo alerta patologia, quantos realmente estao doentes? | Quando o custo de um alarme falso (biopsia invasiva) e alto |
| Recall | De todos os que realmente tem a doenca, quantos o modelo achou? | Quando nao podemos deixar escapar doentes (triagem/sepse) |
| F1-Score | Qual e o equilibrio medio entre precision e recall? | Na escolha do modelo campeao em dados desbalanceados |

---

## Subcamada 3.2: Desenhando o Conceito

### A matriz de confusao diagnóstica

```text
                                REALIDADE CLINICA
                       Saudavel (0)              Patologia (1)
                  +-------------------------+-------------------------+
     PREVISTO     |   Verdadeiro Negativo   |     Falso Negativo      |
   Saudavel (0)   |     (TN: acerto)        |   (FN: ERRO CRITICO)    |
                  +-------------------------+-------------------------+
     PREVISTO     |     Falso Positivo      |   Verdadeiro Positivo   |
   Patologia (1)  |   (FP: alarme falso)    |      (TP: acerto)       |
                  +-------------------------+-------------------------+
```

### O cabo de guerra: Precision contra Recall

```text
Pescador com arpao (Alta Precision)        Pescador com rede (Alto Recall)
"So atiro se tiver certeza absoluta!"      "Pego tudo o que estiver no mar!"
           *      *                               [===================]
           *  (Alvo certeiro)                      *   *   bota  lixo  *
Acerta quase todos os tiros,              Captura quase todos os peixes,
mas deixa peixes escaparem (baixo recall). mas traz lixo junto (baixa precision).
```

### O limiar de decisao e a curva ROC

O modelo emite uma probabilidade continua de `0%` a `100%`. A linha de corte padrao e `50%`.

```text
Limiar baixo (30%): facil dar alerta ---> Recall sobe, Precision cai
Limiar medio (50%): ponto de partida usual
Limiar alto  (70%): so alerta com certeza ---> Precision sobe, Recall cai

A Curva ROC desenha todos os limiares possiveis em um grafico:
AUC = 0.50 (equivalente a jogar moeda) | AUC = 1.00 (separacao perfeita)
```

---

## Subcamada 3.3: Desmistificando a Teoria e a Notacao Formal

### As formulas fundamentais

$$\mathrm{Acuracia} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\mathrm{Precision} = \frac{TP}{TP + FP} \quad \quad \mathrm{Recall} = \frac{TP}{TP + FN}$$

### Por que o F1-score usa a media harmonica?

A media aritmetica simples permitiria compensacoes desonestas: se um modelo tivesse Precision `1.0` e Recall `0.0`, a media aritmetica daria `0.50`. A media harmonica despenca se um dos lados for nulo:

$$
F_1 = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}
$$

Traducao simbolo por simbolo:

| Simbolo | Leitura simples |
|---|---|
| `TP` | pacientes doentes diagnosticados corretamente |
| `TN` | pacientes saudaveis confirmados corretamente |
| `FP` | alarmes falsos: paciente saudavel classificado como doente |
| `FN` | doentes que passaram despercebidos pelo modelo |
| `ROC-AUC` | probabilidade de o modelo ranquear um caso positivo acima de um negativo |

### A ordem correta de avaliacao

A matriz de confusao e as metricas oficiais de teste devem ser extraidas exclusivamente sobre `X_test` e `y_test`. O modelo nunca deve ajustar limiares ou treinar regras olhando para esse conjunto.

---

## Subcamada 3.4: Laboratorio Ludico no Colab

### Toy example: desmascarando um classificador preguiçoso

O codigo simula 1.000 pacientes com uma doenca rara (95% saudaveis). Ele compara um classificador dummy que sempre chuta saudavel com um classificador real.

```python
# Dataset numerico pequeno para tornar a matriz de confusao visivel.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

np.random.seed(42)  # Fixa o sorteio e permite reproduzir a aula.
X_raro = np.random.randn(1000, 5)  # 1.000 pacientes e 5 atributos sem sinal.
# p=[0.95, 0.05]: somente 5% dos pacientes tem a classe positiva.
y_raro = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])

modelos = {
    # Baseline ingenuo: sempre responde a classe mais frequente, 0.
    "Chuta sempre saudavel": DummyClassifier(strategy="most_frequent"),
    # Modelo comparativo: tenta encontrar padroes nos atributos.
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Cada painel mostra os quatro tipos de acerto e erro.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (nome, mod) in zip(axes, modelos.items()):
    # Treino e previsao: aqui o modelo produz o diagnostico de cada paciente.
    mod.fit(X_raro, y_raro)
    pred = mod.predict(X_raro)
    # Acuracia pode enganar; F1 verifica se a classe rara foi encontrada.
    acc = accuracy_score(y_raro, pred)
    f1 = f1_score(y_raro, pred, zero_division=0)
    cm = confusion_matrix(y_raro, pred)
    ax.imshow(cm, cmap="Blues", alpha=0.6)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    ax.set_title(f"{nome}\nAcuracia: {acc:.2f} | F1: {f1:.2f}")
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
plt.tight_layout(); plt.show()
```

> **O que voce deve notar no grafico gerado:** o modelo que chuta sempre zero exibe 95% de acuracia, mas seu F1 e rigorosamente zero porque ele nao encontrou nenhum dos 50 pacientes doentes. A matriz mostra 50 falsos negativos.

**Mini-experimento:** altere a proporcao `p=[0.95, 0.05]` para `[0.60, 0.40]`. Observe como a acuracia do classificador dummy cai para 60% e perde a ilusao de eficiencia.

---

## Subcamada 3.5: O Momento Serio da Nossa Aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

Vamos extrair o painel completo de diagnostico sobre a matriz oficial: 2.000 pacientes, 40 atributos, 10 informativos, 10 redundantes e 20 ruidos. O conjunto de teste possui 500 pacientes reservados.

```python
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# Semente fixa: reproduz a coorte e a divisao treino/teste.
SEED = 42
X_raw, y = make_classification(n_samples=2000, n_features=40, n_informative=10,
    n_redundant=10, weights=[0.6, 0.4], flip_y=0.03, random_state=SEED)
nomes = ([f"biomarcador_{i+1}" for i in range(10)] +
         [f"exame_redundante_{i+1}" for i in range(10)] +
         [f"ruido_metabolico_{i+1}" for i in range(20)])
X = pd.DataFrame(X_raw, columns=nomes)
X_train, X_test, y_train, y_test = train_test_split(
# O treino aprende; o teste fica intocado para medir erros reais.
    X, y, test_size=0.25, stratify=y, random_state=SEED)

rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
resultado = {
    "acuracia": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "TN": tn, "FP": fp, "FN": fn, "TP": tp
}

print(f"Total avaliado no teste: {len(y_test)} pacientes")
for k, v in resultado.items():
    print(f"KPI {k}: {v:.4f}" if isinstance(v, float) else f"KPI {k}: {v}")
```

### Tabela oficial de KPIs

Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Interpretacao | O que investigar |
|---|---|---|
| Acuracia | proporcao geral de laudos corretos | a divisao de classes no teste e proporcional? |
| F1-score | media harmonica entre precision e recall | o modelo equilibra alarmes e acertos na patologia? |
| Precision | taxa de confianca quando o alarme toca | quantos exames invasivos extras foram gerados? |
| Recall | taxa de captura dos doentes reais | quantos pacientes ficaram sem tratamento? |
| ROC-AUC | capacidade de ordenacao de risco | mexer no limiar permite salvar mais pacientes? |
| FN | quantidade absoluta de doentes ignorados | qual o custo etico e clinico desse erro? |
| FP | quantidade absoluta de alarmes falsos | qual o custo laboratorial de re-testar? |

### Interpretacao clinica e de negocio

- Em triagem hospitalar, `FN` tem prioridade de monitoramento: se um paciente patologico recebe previsao de saudavel, perde-se a janela de intervencao inicial.
- Acuracia alta combinada com recall mediano indica que o modelo acerta com facilidade os casos saudaveis (classe majoritaria), mas deixa passar uma fatia relevante dos doentes.
- O F1-score consolidado de ~0.837 sera a nossa linha de base. Nenhuma reducao de atributos futura podera degradar esse patamar.

---

## Subcamada 3.6: Checkpoint de Autonomia e Fixacao Ativa

Explique sem consultar o texto e depois confira sua resposta:

1. Por que um modelo com 99% de acuracia pode ser inutil para rastrear doencas raras?
2. O que acontece com o F1-score se a Precision for 1.0 e o Recall for 0.0?
3. Explique a diferenca entre um falso positivo (FP) e um falso negativo (FN) em um pronto-socorro.
4. O que significa um ROC-AUC de 0.50 e um ROC-AUC de 1.00?
5. Qual a consequencia de ajustar o limiar de decisao (`threshold`) para um valor muito baixo como 0.20?
6. Por que nunca devemos avaliar a matriz de confusao final nos dados em que o modelo treinou?

### Mini-desafio pratico

Modifique o limiar de corte sobre `y_proba` no codigo da Subcamada 3.5 e preencha a tabela:

```text
limiar_corte       precision       recall       F1          FN          FP
0.20               ...             ...          ...         ...         ...
0.35               ...             ...          ...         ...         ...
0.50 (padrao)      ...             ...          ...         ...         ...
0.65               ...             ...          ...         ...         ...
0.80               ...             ...          ...         ...         ...
```

Depois responda: **qual limiar minimiza os falsos negativos sem fazer o F1-score despencar?**
