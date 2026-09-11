# Camada 10: Pre-filtro Hibrido Estatistico (BOLIMES)

**Trilha:** XAI Aplicada à Redução de Dados em Machine Learning
**Aplicação:** classificação binária de saúde (`0 = Saudável`, `1 = Patologia`)
**Código de referência:** [pipeline_completo.py](../pipeline_completo.py), função `pre_filtro_hibrido`

> **Objetivo da aula:** aprender a retirar, antes da XAI, duas formas de desperdicio que a estatistica detecta rapidamente: colunas quase paradas e colunas que repetem a mesma informacao. A ideia nao e substituir SHAP ou LIME. E preparar um palco menor para que essas ferramentas trabalhem com mais foco.

## Campo Didático: A Faxina em Duas Perguntas

Execute o pre-filtro como uma triagem visual: **a coluna varia, ha um clone, quais colunas sobreviveram e o modelo manteve seus KPIs?** Primeiro rode o toy example; depois repita no `X_train` oficial, sem tocar no teste.

```text
variancia baixa? -> remover estatua -> correlacao alta? -> remover clone -> validar
```

Observe a lista de colunas antes/depois e a matriz de correlacao. O erro comum e supor que baixa correlacao significa relevancia ou usar todos os dados para aprender o filtro. A ponte para a Camada 11 e aplicar um criterio estatistico direcional e inferencial sobre as explicacoes que restaram.

### Roteiro de dominio

Antes do filtro, conte colunas e verifique escalas. Depois de cada etapa, registre quantas sobreviveram e quais foram removidas. Finalmente, treine o mesmo modelo no conjunto original e no conjunto pre-filtrado. O ganho so e valido se a selecao foi aprendida no treino e se a metrica do teste nao foi usada para decidir a poda.

### Duvidas que esta aula responde

- **Variancia baixa significa irrelevancia clinica?** Significa pouca variacao observada; a conclusao clinica exige contexto e escala de medida.
- **Correlacao alta significa duplicidade perfeita?** Nao. Indica redundancia linear forte na amostra.
- **Por que preservar a primeira coluna do par?** E uma regra deterministica de representante, nao prova de superioridade clinica.
- **O pre-filtro substitui SHAP?** Nao. Ele remove desperdicio obvio; SHAP investiga comportamento e interacoes mais complexas.

### Regra de explicacao Feynman

Explique como arrumar uma mesa antes de chamar o especialista: retire copos vazios e duplicatas, mas nao jogue fora documentos apenas porque ainda nao leu o conteudo.

### Um exemplo numerico de cada filtro

Considere duas colunas de 100 pacientes:

```text
exame_A = [5.0, 5.0, 5.0, ...]          variancia = 0
exame_B = [80.1, 79.8, 80.4, ...]        variancia > 0.01
exame_C = exame_B * 1000                 |r(B, C)| aproximadamente 1
```

O primeiro filtro elimina `exame_A` porque ele nao separa pacientes naquela amostra. O segundo mantem apenas um representante de `B` e `C`, porque duas escalas carregam quase o mesmo movimento. Nenhuma dessas regras descobre qual coluna causa a patologia; elas apenas retiram desperdicio evidente.

### Como auditar a decisao

Depois de cada etapa, imprima: colunas iniciais, colunas removidas, colunas finais, variancias e pares com `|r|` acima do limiar. Se uma coluna clinicamente importante foi removida, investigue escala, janela temporal, unidade de medida e a regra de representante antes de aceitar o resultado.

### Duvidas frequentes

- **Variancia zero significa que o exame nunca tem valor clinico?** Nao; significa que ele nao variou na base usada para treinar o filtro.
- **Pearson detecta toda redundancia?** Nao. Ele detecta associacao linear, nao relacoes em U ou interacoes.
- **Por que ajustar no treino?** Para impedir que o teste participe da escolha e produza avaliacao otimista.
- **O pre-filtro substitui SHAP?** Nao. Ele remove casos simples; SHAP investiga contribuicoes e interacoes.

### Ponte para shap-select

O pre-filtro responde “o que e parado ou repetido?”. A Camada 11 faz uma pergunta mais exigente: entre as colunas sobreviventes, quais contribuicoes explicativas apresentam direcao e evidencia estatistica?

## Cultura, História e Referências

Pearson ajudou a consolidar a linguagem moderna da correlacao e da estatistica aplicada, mas a cultura estatistica madura tambem ensina seus limites: correlacao linear nao e causalidade, nem detecta toda relacao relevante. Consulte a historia do [coeficiente de correlacao de Pearson](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) e a documentacao de [VarianceThreshold](https://scikit-learn.org/stable/modules/feature_selection.html#variance-threshold) para relacionar a ideia classica a implementacao moderna.

O pre-filtro e uma tradicao de engenharia pragmatica: usar uma ferramenta barata para nao gastar uma ferramenta sofisticada com lixo obvio. O artefato visual mais importante e a matriz de correlacao produzida no laboratorio; ela deve ser lida como evidencia amostral, nao como mapa de causalidade.

**Pergunta cultural:** por que a estatistica classica continua viva na era da XAI? Porque custo computacional, escala de dados e controle de qualidade continuam sendo problemas concretos, independentemente do nome do algoritmo.

## Recursos de Mídia (Visual e Áudio)

- **Visual local:** matriz de correlação do toy example e tabela de colunas antes/depois.
- **Imagem de apoio:** a matriz de correlação produzida pelo laboratório, com a diagonal e o triângulo superior destacados.
- **Referencia:** [VarianceThreshold](https://scikit-learn.org/stable/modules/feature_selection.html#variance-threshold) e documentacao de correlacao.
- **Áudio sugerido:** a faxina da sala antes da visita do especialista.
- **Imagem mental:** estatua parada, clone em outra unidade e biomarcador que ainda precisa de SHAP.

## 📊 Elementos de Comunidade e Status

- **Status:** `Faxina auditada` quando cada coluna removida puder ser justificada sem confundir associação com causalidade.
- **Debate:** “Quem autoriza descartar um exame: a estatistica, o modelo ou o especialista?”
- **Papel rotativo:** estatistico, engenheiro, medico e auditor de vazamento.

## 💡 Engajamento e Conhecimento

- **Atividade:** alterar `0,90` para `0,75` e `0,99`, comparando colunas e desempenho.
- **Produto da aula:** relatorio antes/depois com variancia, correlacao, colunas removidas e impacto.
- **Conexao profissional:** explicar por que o filtro aprende em `X_train` e apenas transforma `X_test`.

## Mapa da aula

1. [Subcamada 10.1: O conceito na vida real](#subcamada-101-o-conceito-na-vida-real)
2. [Subcamada 10.2: Desenhando o conceito](#subcamada-102-desenhando-o-conceito)
3. [Subcamada 10.3: Desmistificando a teoria](#subcamada-103-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 10.4: Laboratorio ludico no Colab](#subcamada-104-laboratorio-ludico-no-colab)
5. [Subcamada 10.5: O momento serio da nossa aplicacao](#subcamada-105-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 10.6: Checkpoint de autonomia](#subcamada-106-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 10.1: O Conceito na Vida Real

### A analogia: preparar a sala antes do especialista

Imagine uma consulta com um especialista muito caro. Antes da consulta, voce recebe uma ficha com 40 exames. Um exame imprime `42.000` para todas as pessoas; dois exames sao, na pratica, o mesmo exame em unidades diferentes. O especialista poderia descobrir isso, mas estaria pagando por um raciocinio sofisticado para fazer uma triagem simples.

O pre-filtro e a triagem da recepcao: remove o que esta parado e sinaliza o que esta duplicado. Depois, SHAP continua necessario para investigar relacoes nao lineares e interacoes que a triagem nao enxerga.

**A grande sacada:** uma coluna pode ser numericamente elegante e ainda assim nao acrescentar informacao. Reduzir dimensao nao significa apagar dados importantes; significa retirar repeticao e ausencia de variacao antes de pedir uma explicacao profunda.

| Movimento | Pergunta simples | Acao | O que nao prova |
|---|---|---|---|
| Variancia | Esta coluna muda entre pacientes? | Remove colunas quase constantes | Que uma coluna variavel seja util para prever `y` |
| Pearson | Duas colunas contam a mesma historia? | Mantem uma e remove a redundante | Que duas colunas pouco correlacionadas sejam independentes |

Neste projeto, **BOLIMES** nomeia a filosofia biobjetivo: primeiro reduzir complexidade com filtros estatisticos baratos; depois selecionar e explicar atributos com metodos XAI mais expressivos.

---

## Subcamada 10.2: Desenhando o Conceito

```text
                 DADOS DE TREINO
        2.000 pacientes x 40 atributos
                          |
                          v
       +----------------------------------+
       | 1. Variancia quase-zero          |
       | "A coluna praticamente se mexe?" |
       +----------------------------------+
                 | remove estatuas
                 v
       +----------------------------------+
       | 2. Pearson                       |
       | "Ha um clone com |r| > 0,90?"    |
       +----------------------------------+
                 | remove repeticoes
                 v
       ATRIBUTOS PRE-LIMPOS
                 |
                 v
       SHAP / shap-select / Optuna
```

### O triangulo superior

Uma matriz de correlacao repete cada par duas vezes: `corr(A, B)` e `corr(B, A)` sao o mesmo numero. A diagonal tambem nao ajuda: toda coluna tem correlacao `1` consigo mesma.

```text
                 A       B       C       D
             +-------+-------+-------+-------+
         A   |  1.00 |  0.94 |  0.12 |  0.03 |  <-- comparar A-B
         B   |  0.94 |  1.00 |  0.15 |  0.02 |
         C   |  0.12 |  0.15 |  1.00 |  0.88 |  <-- comparar C-D
         D   |  0.03 |  0.02 |  0.88 |  1.00 |
             +-------+-------+-------+-------+
              diagonal ignorada; metade inferior e espelho
```

`np.triu(..., k=1)` deixa visivel apenas a regiao acima da diagonal. Quando `|r| > 0,90`, a coluna mais a direita daquele par entra em `to_drop`. A primeira coluna funciona como representante daquele grupo semelhante.

| Situacao | Detectada? | Motivo |
|---|---:|---|
| Coluna constante | Sim | variancia igual a zero |
| Clones lineares | Sim | correlacao de Pearson alta |
| Relacao em forma de U | Nao necessariamente | Pearson mede associacao linear |
| Interacao entre exames | Nao | exige modelo e analise de interacao |
| Coluna variavel, mas inutil | Nao sozinho | variacao nao e relevancia preditiva |

---

## Subcamada 10.3: Desmistificando a Teoria e a Notacao Formal

### Variancia: quanto uma coluna se espalha

Imagine duas salas de termometros. Em uma, todos marcam `36.5`; na outra, marcam `35.8`, `36.5` e `39.1`. A segunda tem mais espalhamento. A variancia transforma essa intuicao em um numero:

$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2
$$

| Simbolo | Em linguagem comum |
|---|---|
| `x_i` | valor do exame para o paciente `i` |
| `mu` | media dos valores da coluna |
| `x_i - mu` | distancia do valor ate a media |
| quadrado | evita cancelamento entre distancias positivas e negativas |
| `N` | quantidade de pacientes |
| `sigma^2` | espalhamento medio |

O `VarianceThreshold(threshold=0.01)` pergunta se a variancia da coluna e menor que `0.01`. Esse limiar e decisao de engenharia e depende da escala dos atributos.

### Pearson: os movimentos combinam?

Se, quando uma pessoa acelera, outra quase sempre acelera proporcionalmente, os movimentos sao parecidos. Pearson mede esse alinhamento linear:

$$
r_{XY} = \frac{\mathrm{cov}(X,Y)}{\sigma_X\sigma_Y}
$$

- `X` e `Y`: duas colunas;
- `cov(X,Y)`: se elas tendem a subir e descer juntas;
- `sigma_X` e `sigma_Y`: espalhamento de cada coluna;
- `r`: varia de `-1` a `+1`;
- `|r|` perto de `1`: forte repeticao linear;
- `|r|` perto de `0`: pouca associacao linear.

O limiar `|r| > 0.90` nao diz que dois exames sao clinicamente identicos; diz que, neste conjunto de dados, um carrega informacao linear muito parecida com a do outro.

### A ordem correta evita vazamento

O filtro deve aprender em `X_train` e somente depois ser aplicado a `X_test`. Se todos os pacientes participam da correlacao antes da separacao, o teste influencia a selecao e a avaliacao fica otimista.

```text
X_train  ---> aprende variancia e correlacoes ---> colunas escolhidas
X_train  ---> fica com as colunas escolhidas
X_test   ---> apenas recebe as mesmas colunas
```

---

## Subcamada 10.4: Laboratorio Ludico no Colab

### Toy example: uma estatua, um clone e uma coluna nova

A coluna `temperatura_clone` acompanha `temperatura`; `exame_estatua` nunca muda.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

rng = np.random.default_rng(42)
n = 100
temperatura = rng.normal(36.5, 0.4, n)
temperatura_clone = temperatura * 1.01 + rng.normal(0, 0.01, n)
exame_estatua = np.full(n, 42.0)
pressao = rng.normal(120, 12, n)
df = pd.DataFrame({"temperatura": temperatura, "temperatura_clone": temperatura_clone,
                   "exame_estatua": exame_estatua, "pressao": pressao})
seletor = VarianceThreshold(0.01).fit(df)
df_var = df.loc[:, seletor.get_support()]
corr = df_var.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
remover = [c for c in upper.columns if (upper[c] > 0.90).any()]
df_limpo = df_var.drop(columns=remover)
print("Antes:", list(df.columns))
print("Depois:", list(df_limpo.columns))
plt.scatter(df["temperatura"], df["temperatura_clone"], alpha=0.7)
plt.xlabel("Temperatura"); plt.ylabel("Clone da temperatura")
plt.title("Duas colunas contando quase a mesma historia")
plt.show()
```

> **O que voce deve notar no grafico gerado:** os pontos formam uma faixa quase reta. No texto impresso, `exame_estatua` some por baixa variancia e `temperatura_clone` some por redundancia. `pressao` permanece.

**Mini-investigacao:** troque `0.90` por `0.75` e depois por `0.99`. Um limiar menor poda mais agressivamente; um maior exige clones ainda mais perfeitos.

---

## Subcamada 10.5: O Momento Serio da Nossa Aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

O experimento oficial usa `make_classification` com 2.000 pacientes, 40 atributos, 10 informativos, 10 redundantes, 20 ruidos, proporcao `0.6/0.4` e `flip_y=0.03`. O codigo mede o pre-filtro no treino e compara Random Forests de 100 arvores antes e depois.

```python
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

SEED = 42
X_raw, y = make_classification(n_samples=2000, n_features=40, n_informative=10,
    n_redundant=10, n_classes=2, weights=[0.6, 0.4], flip_y=0.03,
    random_state=SEED)
nomes = ([f"biomarcador_{i+1}" for i in range(10)] +
         [f"exame_redundante_{i+1}" for i in range(10)] +
         [f"ruido_metabolico_{i+1}" for i in range(20)])
X = pd.DataFrame(X_raw, columns=nomes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

def pre_filtro(X_treino, limiar_var=0.01, limiar_corr=0.90):
    t0 = time.perf_counter()
    var = VarianceThreshold(limiar_var).fit(X_treino)
    colunas_var = X_treino.columns[var.get_support()].tolist()
    X_var = X_treino[colunas_var]
    corr = X_var.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    remover = [c for c in upper.columns if (upper[c] > limiar_corr).any()]
    colunas = [c for c in colunas_var if c not in remover]
    return colunas, remover, time.perf_counter() - t0

def medir(modelo, X_treino, X_teste, y_treino, y_teste):
    inicio = time.perf_counter()
    modelo.fit(X_treino, y_treino)
    treino_s = time.perf_counter() - inicio
    inicio = time.perf_counter()
    pred = modelo.predict(X_teste)
    proba = modelo.predict_proba(X_teste)[:, 1]
    inferencia_s = time.perf_counter() - inicio
    acc_treino = accuracy_score(y_treino, modelo.predict(X_treino))
    acc_teste = accuracy_score(y_teste, pred)
    tn, fp, fn, tp = confusion_matrix(y_teste, pred).ravel()
    return {"atributos": X_treino.shape[1], "acuracia_treino": acc_treino,
        "acuracia_teste": acc_teste, "gap_overfitting": acc_treino - acc_teste,
        "f1": f1_score(y_teste, pred), "precision": precision_score(y_teste, pred),
        "recall": recall_score(y_teste, pred), "roc_auc": roc_auc_score(y_teste, proba),
        "latencia_us_paciente": inferencia_s * 1_000_000 / len(X_teste),
        "tempo_treino_ms": treino_s * 1000, "tn": tn, "fp": fp, "fn": fn, "tp": tp}

colunas, removidas, tempo_filtro_s = pre_filtro(X_train)
baseline = medir(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
                 X_train, X_test, y_train, y_test)
enxuto = medir(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
               X_train[colunas], X_test[colunas], y_train, y_test)
print(f"Dimensao: {X.shape[1]} -> {len(colunas)} atributos")
print("Removidas por Pearson:", removidas)
print(f"Tempo do pre-filtro: {tempo_filtro_s * 1000:.3f} ms")
print(pd.DataFrame([baseline, enxuto], index=["Baseline 40", "Pos-filtro"]).round(4))
for nome, resultado in [("Baseline 40", baseline), ("Pos-filtro", enxuto)]:
    print(f"KPI {nome}: acuracia_treino={resultado['acuracia_treino']:.4f}; "
          f"acuracia_teste={resultado['acuracia_teste']:.4f}; "
          f"gap_overfitting={resultado['gap_overfitting']:.4f}; "
          f"f1={resultado['f1']:.4f}; recall={resultado['recall']:.4f}; "
          f"latencia_us_paciente={resultado['latencia_us_paciente']:.2f}; "
          f"tempo_treino_ms={resultado['tempo_treino_ms']:.2f}; "
          f"TN={resultado['tn']}; FP={resultado['fp']}; "
          f"FN={resultado['fn']}; TP={resultado['tp']}")
```

### Tabela oficial de KPIs

Os valores de desempenho sao calculados na execucao. Tempo, latencia e quantidade de colunas removidas variam com CPU, versoes e ambiente; nao trate uma medicao local como constante cientifica.

| KPI | Como e calculado | Pergunta operacional |
|---|---|---|
| Acuracia treino | acertos no `X_train` | O modelo aprendeu os exemplos vistos? |
| Acuracia teste | acertos no `X_test` | Ele generaliza para pacientes novos? |
| Gap de overfitting | `acuracia_treino - acuracia_teste` | Ha distancia entre memorizar e generalizar? |
| F1-score | media harmonica de precision e recall | O equilibrio entre alarmes e casos perdidos e aceitavel? |
| Precision | `TP / (TP + FP)` | Entre os alertas, quantos eram corretos? |
| Recall | `TP / (TP + FN)` | Quantos pacientes com patologia foram encontrados? |
| ROC-AUC | area sob a curva ROC | O ranking separa classes em varios limiares? |
| Latencia | inferencia / pacientes, em us | Quanto custa avaliar um paciente? |
| Tempo de treino | duracao do `.fit`, em ms | O re-treinamento cabe na rotina? |
| Diagnostico | `TN`, `FP`, `FN`, `TP` | Que tipo de erro esta acontecendo? |

### Leitura pratica

Se F1 e recall permanecerem proximos do baseline com menos atributos, o filtro reduz memoria e torna a XAI mais manejavel. Um `FN` e uma patologia classificada como saudavel; em saude, pode ser mais grave que um `FP`, portanto acuracia sozinha nao governa a decisao. Menos atributos podem reduzir custos de coleta, armazenamento e processamento, mas nao autorizam declarar economia financeira ou vidas salvas sem custos clinicos e validacao externa. O pre-filtro prepara SHAP, `shap-select` e Optuna; nao escolhe sozinho o modelo campeao.

---

## Subcamada 10.6: Checkpoint de Autonomia e Fixacao Ativa

Explique sem consultar o texto e depois confira:

1. Por que uma coluna constante nao ajuda a distinguir pacientes?
2. Por que `corr(A, B)` e `corr(B, A)` nao precisam ser analisados separadamente?
3. O que `|r| > 0.90` autoriza afirmar e o que nao autoriza?
4. Por que usar `X_test` para aprender as colunas causa vazamento?
5. Por que `FN` e recall podem ser mais importantes que acuracia em saude?
6. O que pode escapar do Pearson se duas variaveis tiverem relacao em forma de U?

### Mini-desafio pratico

Repita o experimento com `limiar_corr=0.75` e `0.99`. Registre:

```text
limiar | atributos finais | atributos removidos | F1 | recall | FN | latencia_us
```

Depois responda: **qual configuracao reduz dimensao sem piorar o erro clinicamente mais perigoso?** Assim voce conecta regra estatistica, metrica e consequencia pratica.
