# Camada 06: SHAP e a Teoria dos Jogos Cooperativos

**Trilha:** XAI Aplicada a Reducao de Dados em Machine Learning  
**Aplicacao:** classificacao binaria de saude (`0 = Saudavel`, `1 = Patologia`)  
**Codigo de referencia:** [pipeline_completo.py](../pipeline_completo.py), funcao `executar_etapa_shap`

> **Objetivo da aula:** dominar o metodo SHAP (SHapley Additive exPlanations), entender como a Teoria dos Jogos Cooperativos de Lloyd Shapley garante a divisao justa de credito entre atributos, conhecer os quatro axiomas de equidade e interpretar o ranking |SHAP| e o grafico Beeswarm.

## Mapa da aula

1. [Subcamada 6.1: O conceito na vida real](#subcamada-61-o-conceito-na-vida-real)
2. [Subcamada 6.2: Desenhando o conceito](#subcamada-62-desenhando-o-conceito)
3. [Subcamada 6.3: Desmistificando a teoria e a notacao formal](#subcamada-63-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 6.4: Laboratorio ludico no Colab](#subcamada-64-laboratorio-ludico-no-colab)
5. [Subcamada 6.5: O momento serio da nossa aplicacao](#subcamada-65-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 6.6: Checkpoint de autonomia e fixacao ativa](#subcamada-66-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 6.1: O Conceito na Vida Real

### A historia do trabalho em grupo na faculdade

Imagine um trabalho academico em grupo com tres colegas:

- O colega A leu todos os artigos e escreveu a fundamentacao teorica.
- O colega B rodou os codigos e gerou os graficos.
- O colega C apenas colocou o nome na capa e nao compareceu a nenhuma reuniao.

Se o professor atribuir a nota 10 igualmente aos tres, estara sendo injusto com o esforco de A e B. Mas como mensurar matematicamente a contribuicao individual de cada participante quando o resultado e coletivo?

Em 1953, o matematico Lloyd Shapley (Nobel de Economia em 2012) resolveu essa questao: para medir o valor justo de um membro, avalia-se quanto a equipe ganharia em **todas as combinacoes possiveis** com e sem a participacao daquele membro. Se a presenca de C nao altera o resultado em nenhuma combinacao, a fatia dele e rigorosamente zero.

Em 2017, Scott Lundberg adaptou essa teoria para Machine Learning:
- **Jogadores:** os 40 exames do paciente.
- **Jogo:** o modelo Random Forest.
- **Premio:** a probabilidade de patologia.
- **Valor Shapley:** quanto cada exame puxou o risco para cima ou para baixo em relacao a media.

**A grande sacada:** o SHAP e a unica tecnica de explicabilidade que garante matematicamente uma atribuicao de importancia justa, sem favorecer arbitrariamente nenhuma coluna.

### O vocabulario da Teoria dos Jogos em Machine Learning

| Teoria dos Jogos | Machine Learning | No nosso projeto de saude |
|---|---|---|
| Jogador | Atributo / Feature | `biomarcador_1`, `ruido_metabolico_1`, etc. |
| Coalizao | Subconjunto de variaveis presentes | Grupo de exames ativos no no da arvore |
| Valor do jogo | Previsao emitida $f(x)$ | Probabilidade de o paciente ter a doenca |
| Valor Shapley $\phi_i$ | Importancia aditiva da variavel | Impacto em pontos percentuais no laudo |

---

## Subcamada 6.2: Desenhando o Conceito

### O cabo de guerra diagnostico

```text
    [ RISCO MEDIO DO HOSPITAL: 40% ]  (Base Value populacional)
                   |
                   +---> Troponina muito alta : +25% de risco (empurra p/ patologia 🔴)
                   +---> Glicemia elevada     : +15% de risco (empurra p/ patologia 🔴)
                   +---> Idade jovem          : -08% de risco (empurra p/ saudavel 🔵)
                   +---> Ruido metabolico 14  : +0.01%        (neutro / nulo ⚪)
                   |
                   v
    PREVISAO FINAL DO PACIENTE: 72.01%  (Soma exata das forcas!)
```

### Como ler o grafico Beeswarm

```text
Biomarcador 1  ---🔵🔵🔵-------------|-------------🔴🔴🔴--->  (Alto valor = Alto risco)
Biomarcador 2  ---🔴🔴🔴-------------|-------------🔵🔵🔵--->  (Baixo valor = Alto risco)
Ruido 7        ---------------🔵🔴🔵🔴🔵🔴---------------->  (Sem efeito, tudo no zero)
                                     |
                             Valor SHAP (Impacto)
                 <--- Empurra p/ Saudavel   Empurra p/ Patologia --->

Legenda de cores: 🔴 Ponto vermelho = valor alto do exame | 🔵 Ponto azul = valor baixo do exame
```

---

## Subcamada 6.3: Desmistificando a Teoria e a Notacao Formal

### A formula de Lloyd Shapley

O valor Shapley da variavel $i$ e a media ponderada das suas contribuicoes marginais sobre todas as coalizoes possiveis $S$:

$$
\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \big[ v(S \cup \{i\}) - v(S) \big]
$$

Traducao simbolo por simbolo:

| Simbolo | Leitura simples |
|---|---|
| `N` | conjunto total de atributos (40 exames) |
| `S` | uma coalizao qualquer de atributos sem a presenca de `i` |
| `v(S U {i}) - v(S)` | contribuicao marginal: quanto a previsao muda quando adicionamos o atributo `i` |
| `fator fatorial` | peso estatistico para equilibrar coalizoes de diferentes tamanhos |
| `phi_i` | valor Shapley: credito liquido e justo da variavel `i` |

### Os quatro axiomas de equidade

1. **Eficiencia:** a soma de todos os valores $\phi_i$ e igual a diferenca entre a previsao do paciente e a media da base: $\sum \phi_i = f(x) - \mathbb{E}[f(X)]$.
2. **Simetria:** duas variaveis com o mesmo efeito marginal em todas as coalizoes recebem o mesmo valor.
3. **Variavel Nula (Dummy):** se uma variavel nao altera a previsao em nenhuma coalizao (como os 20 ruidos metabolicos), seu valor e rigorosamente zero ($\phi_i = 0$).
4. **Aditividade:** a explicacao de um ensemble e a soma das explicacoes de cada arvore componente.

### TreeSHAP: de 1 trilhao para 2 segundos

Calcular a formula classica exigiria avaliar $2^{40} \approx 1,1$ trilhao de coalizoes. O algoritmo **TreeSHAP** (Lundberg et al., 2020) percorre os ramos das arvores de decisao recursivamente, reduzindo a complexidade para ordem polinomial $\mathcal{O}(T \cdot L \cdot D^2)$ e calculando a solucao exata em segundos.

---

## Subcamada 6.4: Laboratorio Ludico no Colab

### Toy example: medindo créditos justos com TreeSHAP

O codigo treina uma floresta em 3 atributos (2 informativos e 1 ruido) e calcula os valores SHAP exatos.

```python
!pip install shap -q
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n = 300
idade = np.random.uniform(20, 80, n)
pressao = np.random.uniform(90, 180, n)
ruido = np.random.normal(0, 1, n)

prob = 1 / (1 + np.exp(-(-5.0 + 0.04 * idade + 0.02 * pressao)))
y = (prob > 0.5).astype(int)
df = pd.DataFrame({"Idade": idade, "Pressao": pressao, "Ruido_Sorte": ruido})

rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(df, y)
explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(df)
valores = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if len(sv.shape) == 3 else sv)

shap.summary_plot(valores, df, plot_type="bar", show=False)
plt.title("Ranking Global SHAP (|Impacto Medio|)")
plt.tight_layout(); plt.show()

print(pd.Series(np.abs(valores).mean(axis=0), index=df.columns).round(4))
```

> **O que voce deve notar no grafico gerado:** `Pressao` e `Idade` lideram o ranking absoluto. `Ruido_Sorte` recebe valor proximo de zero, respeitando o axioma da variavel nula.

**Mini-experimento:** inverta o sinal da pressao na geracao do alvo (`- 0.02 * pressao`). Plote com `shap.summary_plot(valores, df)` e observe os pontos vermelhos migrarem para o lado esquerdo.

---

## Subcamada 6.5: O Momento Serio da Nossa Aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

Vamos rodar a extracao completa de TreeSHAP sobre a matriz oficial de 40 atributos com 2.000 pacientes do [pipeline_completo.py](../pipeline_completo.py).

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
amostra = X_train.iloc[:250]
sv = explainer.shap_values(amostra)
tempo_total_ms = (time.perf_counter() - t0) * 1000

vals = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if len(sv.shape) == 3 else sv)
ranking = pd.DataFrame({
    "atributo": nomes,
    "impacto_shap": np.abs(vals).mean(axis=0),
    "tipo": ["Informativo"]*10 + ["Redundante"]*10 + ["Ruido"]*20
}).sort_values(by="impacto_shap", ascending=False).reset_index(drop=True)

total = ranking["impacto_shap"].sum()
top10 = ranking.head(10)["impacto_shap"].sum()
ruidos = ranking[ranking["tipo"] == "Ruido"]["impacto_shap"].sum()

print(f"Tempo TreeSHAP para 250 pacientes: {tempo_total_ms:.2f} ms")
print(f"Latencia por laudo individual: {tempo_total_ms / 250:.2f} ms")
print(f"Proporcao de impacto retida no Top 10: {(top10/total)*100:.2f}%")
print(f"Proporcao de impacto nos 20 ruidos: {(ruidos/total)*100:.2f}%")
print("\nTop 5 atributos segundo o SHAP:")
print(ranking.head(5).round(4).to_string(index=False))
```

### Tabela oficial de KPIs

Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Interpretacao | O que investigar |
|---|---|---|
| Latencia por laudo SHAP | tempo medio de calculo por paciente em ms | e viavel gerar explicacao durante a triagem? |
| Concentracao Top 10 | porcentagem de forca acumulada nos 10 primeiros | o descarte de 30 atributos preservara a informacao? |
| Impacto dos 20 ruidos | forca somada de todas as colunas aleatorias | o axioma da variavel nula foi confirmado numericamente? |
| Base value | risco medio historico da populacao estudada | qual e o ponto de partida do cabo de guerra? |

### Interpretacao clinica e de negocio

- O axioma da variavel nula e confirmado empiricamente: os 20 ruidos somados representam fracao insignificante do impacto global.
- O TreeSHAP permite auditar cada decisao em milissegundos, atendendo exigencias eticas e regulatorias de explicabilidade na saude.
- O ranking SHAP orientara os proximos modulos: ablacao progressiva e selecao econometrica (`shap-select`).

---

## Subcamada 6.6: Checkpoint de Autonomia e Fixacao Ativa

Explique sem consultar o texto e depois confira sua resposta:

1. O que e a contribuicao marginal de um jogador na Teoria dos Jogos?
2. Explique o Axioma da Eficiencia do SHAP usando a metafora do cabo de guerra.
3. O que afirma o Axioma da Variavel Nula sobre as 20 colunas de ruido metabolico?
4. Como o TreeSHAP consegue escapar da barreira de 1 trilhao de combinacoes de variaveis?
5. No grafico Beeswarm, o que indica um aglomerado de pontos vermelhos a direita do zero?
6. Qual a diferenca entre o modulo |SHAP| e o valor SHAP original com sinal positivo ou negativo?

### Mini-desafio pratico

Execute o calculo com uma amostra menor e anote:

```text
amostra_pacientes     tempo_ms     latencia_unitaria_ms     top1_atributo
50                    ...          ...                      ...
100                   ...          ...                      ...
250                   ...          ...                      ...
```

Depois responda: **a ordem de importancia dos atributos principais permaneceu estavel mesmo com menos pacientes avaliados?**
