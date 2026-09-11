# Camada 05: Fundamentos de XAI (Global e Local)

**Trilha:** XAI Aplicada a Reducao de Dados em Machine Learning  
**Aplicacao:** classificacao binaria de saude (`0 = Saudavel`, `1 = Patologia`)  
**Codigo de referencia:** [pipeline_completo.py](../pipeline_completo.py), funcoes `executar_etapa_shap` e `executar_etapa_lime`

> **Objetivo da aula:** compreender por que a precisao isolada de uma caixa-preta e insuficiente em areas criticas, conhecer o caso historico em que uma IA aprendeu regras perigosas e diferenciar a explicabilidade global (o mapa da populacao com SHAP) da explicabilidade local (a auditoria individual com LIME).

## Mapa da aula

1. [Subcamada 5.1: O conceito na vida real](#subcamada-51-o-conceito-na-vida-real)
2. [Subcamada 5.2: Desenhando o conceito](#subcamada-52-desenhando-o-conceito)
3. [Subcamada 5.3: Desmistificando a teoria e a notacao formal](#subcamada-53-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 5.4: Laboratorio ludico no Colab](#subcamada-54-laboratorio-ludico-no-colab)
5. [Subcamada 5.5: O momento serio da nossa aplicacao](#subcamada-55-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 5.6: Checkpoint de autonomia e fixacao ativa](#subcamada-56-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 5.1: O Conceito na Vida Real

### A historia da oficina mecanica e o caso da pneumonia

Imagine levar seu carro a oficina e o mecanico cobrar R$ 5.000,00 dizendo apenas: "Confie em mim, o motor agora funciona, mas nao posso explicar quais pecas troquei". Ninguem aceitaria isso. Exigimos uma nota detalhada com cada servico executado.

Na medicina, a exigencia e ainda mais grave. Em um estudo real nos Estados Unidos (Caruana et al.), uma rede neural foi treinada para prever risco de morte por pneumonia. O modelo teve alta acuracia, mas descobriu-se uma regra interna alarmante: pacientes com historico de asma recebiam menor risco de morte.

Na pratica hospitalar, pacientes com asma eram encaminhados imediatamente a UTI e recebiam cuidado redobrado, o que reduzia sua mortalidade. O modelo aprendeu a associacao estatistica dos dados, mas inverteu o nexo causal. Se fosse adotado na triagem sem auditoria, mandaria pacientes graves para casa.

**A grande sacada:** acertar estatisticamente nao e garantia de aprender biologia real. A explicabilidade (XAI) e a ferramenta que audita se o modelo decidiu pelos motivos corretos.

### As duas perspectivas da explicabilidade

| Perspectiva | Pergunta que responde | Analogia | Ferramenta no projeto |
|---|---|---|---|
| Global | No conjunto todo, quais exames mais pesam nas decisoes? | Mapa de satelite de todo o percurso | SHAP (`TreeExplainer`) |
| Local | Por que ESTE paciente especifico recebeu previsao positiva? | Zoom no GPS na curva da rua | LIME (`LimeTabularExplainer`) |

---

## Subcamada 5.2: Desenhando o Conceito

### A caixa-preta contra a caixa transparente

```text
    MODELO TRADICIONAL CAIXA-PRETA               MODELO COM AUDITORIA XAI
       [ 40 exames do paciente ]                 [ 40 exames do paciente ]
                   |                                         |
                   v                                         v
       +-----------------------+                 +-----------------------+
       |   ??? REGRAS ???      |                 |     Random Forest     |
       |  100 arvores em voto  |                 +-----------+-----------+
       +-----------+-----------+                             |
                   |                                         v
                   v                             [ LAUDO DE EXPLICABILIDADE ]
       Previsao: 85% patologia                   - Troponina alta: +35% risco
       Motivo: desconhecido                      - Glicemia alta:  +15% risco
                                                 - Ruidos 1 a 20:   zero efeito
```

### O mapa global e a lupa local

```text
                  AUDITORIA DE XAI NO PROJETO
                               |
             +-----------------+-----------------+
             |                                   |
             v                                   v
      VISAO GLOBAL (SHAP)                 VISAO LOCAL (LIME)
   Mapeia toda a populacao              Examina o paciente individual
   "Quais colunas comandam o modelo?"   "Por que este paciente foi classificado assim?"
             |                                   |
             v                                   v
   Usado para podar 30 colunas inuteis  Usado para laudo e justificativa clinica
```

---

## Subcamada 5.3: Desmistificando a Teoria e a Notacao Formal

### Modelos substitutos interpretáveis

Seja $f(x)$ o modelo original caixa-preta (Random Forest) e $x$ a instancia do paciente. A explicabilidade post-hoc procura um modelo interpretavel $g \in G$ (como uma regressao linear ou regra aditiva) que seja fiel a $f$ na regiao de interesse:

$$
\operatorname{Explicacao}(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)
$$

Traducao simbolo por simbolo:

| Simbolo | Leitura simples |
|---|---|
| `f` | o modelo original complexo |
| `g` | a explicacao simples e compreensivel |
| `L` | a diferenca entre o que `f` preve e o que `g` explica |
| `pi_x` | peso de proximidade: focar na vizinhança daquele paciente |
| `Omega(g)` | penalidade de complexidade: a explicacao deve ter poucos termos |

### A ordem correta evita vazamento

A explicabilidade global para selecao de variaveis deve ser calculada estritamente sobre dados de treino. Se calcularmos valores SHAP usando a base inteira antes do split, introduzimos informacao do teste na selecao.

---

## Subcamada 5.4: Laboratorio Ludico no Colab

### Toy example: interrogando a importancia de variaveis

O codigo treina um modelo simples em 4 colunas (2 informativas e 2 ruidos puros) e inspeciona se a IA realmente aprendeu a ignorar o ruido.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n = 400
v1 = np.random.normal(0, 1, n)
v2 = np.random.normal(0, 1, n)
r1 = np.random.normal(0, 1, n)
r2 = np.random.normal(0, 1, n)

y = ((1.8 * v1 - 1.2 * v2 + np.random.normal(0, 0.4, n)) > 0).astype(int)
df = pd.DataFrame({"Biomarcador_A": v1, "Biomarcador_B": v2, "Ruido_1": r1, "Ruido_2": r2})

rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(df, y)
importancias = pd.Series(rf.feature_importances_, index=df.columns).sort_values()

plt.figure(figsize=(8, 3.5))
importancias.plot(kind="barh", color=["gray", "gray", "steelblue", "navy"])
plt.title("Auditoria de Importancia: o Modelo Confessando")
plt.xlabel("Peso relativo atribuido"); plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout(); plt.show()

print(importancias.round(4))
```

> **O que voce deve notar no grafico gerado:** os dois biomarcadores reais acumulam a maior parte do peso relativo, enquanto os ruidos ficam no rodapé. Essa confissao do modelo fundamenta a poda de atributos.

**Mini-experimento:** inverta a regra do alvo para depender apenas de `r1`. Veja como as barras se invertem imediatamente.

---

## Subcamada 5.5: O Momento Serio da Nossa Aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

Vamos calcular a explicabilidade global com `shap.TreeExplainer` no modelo treinado com 40 atributos, verificando se o algoritmo prioriza os 10 biomarcadores ou se foi iludido pelos 20 ruidos.

```python
import time
import numpy as np
import pandas as pd
import shap
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
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

rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1).fit(X_train, y_train)

t0 = time.perf_counter()
explainer = shap.TreeExplainer(rf)
amostra = X_train.iloc[:200]
sv = explainer.shap_values(amostra)
tempo_shap_ms = (time.perf_counter() - t0) * 1000

valores = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if len(sv.shape) == 3 else sv)
media_abs = np.abs(valores).mean(axis=0)
ranking = pd.Series(media_abs, index=nomes).sort_values(ascending=False)

impacto_total = ranking.sum()
impacto_top10 = ranking.head(10).sum()
impacto_ruidos = ranking[[c for c in nomes if "ruido" in c]].sum()

print(f"Tempo SHAP para {len(amostra)} pacientes: {tempo_shap_ms:.2f} ms")
print(f"Latencia media por laudo: {tempo_shap_ms / len(amostra):.2f} ms")
print(f"Concentracao de impacto no Top 10: {(impacto_top10 / impacto_total)*100:.2f}%")
print(f"Impacto total dos 20 ruidos: {(impacto_ruidos / impacto_total)*100:.2f}%")
print("\nTop 5 atributos mais influentes:")
print(ranking.head(5).round(4))
```

### Tabela oficial de KPIs

Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Interpretacao | O que investigar |
|---|---|---|
| Tempo de calculo SHAP | custo em ms para extrair explicacoes da amostra | o explicador cabe no fluxo de treinamento? |
| Concentracao no Top 10 | proporcao do impacto retida pelas 10 primeiras colunas | o sinal esta concentrado ou espalhado? |
| Impacto residual dos ruidos | soma da importancia atribuida aos 20 ruidos | o modelo aprendeu a ignorar o lixo estatistico? |
| Top 1 atributo | biomarcador de maior magnitude absoluta | esse achado tem respaldo na fisiopatologia? |

### Interpretacao clinica e de negocio

- O ranking global de SHAP mostra que os 10 biomarcadores concentram mais de 90% da massa de decisao do Random Forest, validando empiricamente o descarte das 30 colunas restantes.
- A explicabilidade transforma-se em ferramenta ativa: nao e usada apenas para justificar laudos no final, mas para orientar a reducao cirurgica de custos no hospital.
- O tempo de milissegundos por laudo viabiliza auditoria em tempo real sem comprometer o fluxo de atendimento da equipe de saude.

---

## Subcamada 5.6: Checkpoint de Autonomia e Fixacao Ativa

Explique sem consultar o texto e depois confira sua resposta:

1. O que foi o caso da pneumonia e da asma e por que ele e considerado um marco da necessidade de XAI?
2. Qual e a diferenca entre explicabilidade global e explicabilidade local?
3. Em que situacao o diretor clinico de um hospital usaria o SHAP e em que situacao usaria o LIME?
4. Por que calcular a explicabilidade sobre todos os dados antes da separacao de treino e teste e uma falha metodologica?
5. Como a concentracao de impacto no Top 10 justifica a reducao de atributos?
6. O que e um modelo substituto interpretavel e qual o seu compromisso de fidelidade?

### Mini-desafio pratico

No codigo da Subcamada 5.5, aumente a amostra de explicacao de `200` para `500` pacientes e registre:

```text
tamanho_amostra     tempo_total_ms     latencia_por_laudo_ms     top1_biomarcador
200                 ...                ...                       ...
500                 ...                ...                       ...
```

Depois responda: **o tempo cresceu de forma linear com o numero de pacientes avaliados?**
