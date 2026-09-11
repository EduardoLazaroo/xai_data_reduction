# Camada 07: LIME e Explicabilidade Local

**Trilha:** XAI Aplicada a Reducao de Dados em Machine Learning  
**Aplicacao:** classificacao binaria de saude ('0 = Saudavel', '1 = Patologia')  
**Codigo de referencia:** [pipeline_completo.py](../pipeline_completo.py), funcao `executar_etapa_lime`

> **Objetivo da aula:** Compreender a formulacao teorica e a dinamica pratica do LIME (Local Interpretable Model-agnostic Explanations), analisando a geracao de perturbacoes estocasticas no espaco amostral, a ponderacao por kernel gaussiano de distancia e a motivacao clinica de auditar instancias proximas ao limiar de decisao de 50%.

## Campo Didatico: Uma Lupa Para Um Paciente

O LIME deve ser estudado como uma investigacao local: **escolha um paciente, crie vizinhos artificiais, pese os vizinhos pela distancia, ajuste um modelo simples e compare a explicacao com a previsao original**. Repita com outra semente para perceber a sensibilidade.

```text
paciente escolhido -> perturbacoes -> pesos de proximidade -> modelo simples local
        |                                                   |
        v                                                   v
    previsao original ------------------------------> regras explicativas
```

Observe que uma boa explicacao local nao precisa representar toda a floresta. O erro comum e tratar uma explicacao instavel como verdade absoluta ou confundir proximidade matematica com semelhança clinica. A ponte para a Camada 08 e comparar essa leitura XAI com uma selecao tradicional de atributos.

---

### Roteiro de dominio

Repita a explicacao com sementes e tamanhos de vizinhanca diferentes. Compare quais atributos permanecem, a fidelidade local e a previsao original. Se a historia muda muito, a conclusao correta e “a explicacao e instavel sob esta configuracao”, nao escolher a versao mais conveniente.

### Duvidas que esta aula responde

- **LIME explica o modelo inteiro?** Nao. Ele aproxima o comportamento perto de uma instancia.
- **Mais perturbacoes sempre melhoram?** Ajudam a reduzir variacao, mas podem incluir pontos irreais ou tornar a vizinhanca pouco local.
- **Sigma grande e melhor?** Nao. Ele aumenta o raio e pode misturar regioes com comportamentos diferentes.
- **Perturbacao aleatoria representa pacientes reais?** Somente se respeitar escalas, limites e relacoes plausiveis.

### Regra de explicacao Feynman

Explique LIME como usar uma lupa: ela mostra detalhes de um ponto, mas nao permite afirmar como a cidade inteira funciona. Mover a lupa ou mudar o foco pode alterar o que aparece.

### Um paciente e uma vizinhanca

Considere um paciente com probabilidade `0,503` de patologia. Ele esta praticamente no limiar: uma pequena mudanca em glicemia, idade ou troponina pode alterar a classe. O LIME cria copias perturbadas, consulta a floresta e observa quais mudancas acompanham a probabilidade.

```text
paciente alvo x:       [glicemia 140, idade 62, troponina 0.8] -> 0.503
clone 1:               [glicemia 145, idade 62, troponina 0.8] -> 0.611
clone 2:               [glicemia 140, idade 62, troponina 0.5] -> 0.472
clone 3:               [glicemia 138, idade 61, troponina 0.8] -> 0.491
                                  |
                                  v
                       ajuste local -> sinais que empurram a decisao
```

Uma perturbacao distante recebe peso menor que uma perturbacao proxima. A explicacao responde “o que influenciou esta decisao nesta vizinhanca”, nao “qual e a lei universal do modelo”.

### Como interpretar a estabilidade

Rode a mesma instancia com varias sementes. Se glicemia aparece sempre com sinal positivo e ruido oscila perto de zero, a leitura e mais confiavel. Se os atributos trocam completamente de posicao, a explicacao e sensivel ao desenho da vizinhanca e deve ser apresentada com cautela. Compare tambem a fidelidade do explicador: uma reta que nao reproduz bem a floresta local nao e um bom laudo.

### Um calculo simples de proximidade

Considere o paciente alvo `x = [0, 0]` e dois clones: `z1 = [0, 0,1]` e `z2 = [1, 1]`. Supondo `sigma = 0,5`, os pesos do kernel sao:

$$
\pi_x(z_1)=e^{-0,1^2/0,5^2}\approx0,961,
\qquad
\pi_x(z_2)=e^{-2/0,5^2}\approx0,0003
$$

Mesmo que ambos recebam previsoes da floresta, `z1` influencia muito mais a reta local porque esta proximo do paciente. O LIME nao trata todos os exemplos sinteticos como igualmente importantes; ele aproxima o comportamento no bairro do caso auditado.

### Duvidas frequentes

- **Por que nao explicar a floresta inteira com uma reta?** Porque a fronteira global pode ter degraus e interacoes que uma reta nao representa.
- **Vizinho sintetico e um paciente real?** Nao. E uma sonda matematica e pode produzir combinacoes clinicamente impossiveis se o gerador nao tiver restricoes.
- **Coeficiente positivo prova risco causal?** Nao; indica associacao local com a saida do modelo.
- **Por que usar um paciente no limiar?** Porque a decisao e mais vulneravel a pequenas mudancas e merece auditoria detalhada.

### Ponte para a decisao

Depois do laudo, a equipe deve perguntar se os atributos destacados fazem sentido clinico e se a previsao original e confiavel. A Camada 08 contrasta essa auditoria local com uma selecao global e recursiva: RFE.

## Cultura, Historia e Referencias

O LIME foi apresentado por Ribeiro, Singh e Guestrin no artigo [Why Should I Trust You?](https://doi.org/10.1145/2939672.2939778), uma pergunta culturalmente poderosa: confiar em uma previsao nao e o mesmo que aceitar uma caixa-preta sem questionamento. O trabalho tornou popular a ideia de uma explicacao local simples para um modelo complexo, mas tambem abriu debates sobre instabilidade e fidelidade.

Veja o artefato [modulo3_lime_local.png](../assets/modulo3_lime_local.png) e pergunte: se eu mudar a semente, a vizinhanca ou o kernel, o laudo continua parecido? Essa pergunta faz parte da cultura profissional de explicabilidade. O [repositorio oficial do LIME](https://github.com/marcotcr/lime) permite conhecer a ferramenta original.

**Pergunta cultural:** uma explicacao que muda a cada execucao e uma explicacao ou apenas uma narrativa plausivel? A resposta exige medir estabilidade, nao escolher o grafico mais convincente.

## Recursos de Mídia (Visual e Áudio)

- **Visual local:** [Laudo LIME](../assets/modulo3_lime_local.png), destacando paciente proximo de `P=0,50`.
- **Referencia:** [Repositorio oficial do LIME](https://github.com/marcotcr/lime) e artigo [Why Should I Trust You?](https://doi.org/10.1145/2939672.2939778).
- **Audio de abertura:** narrar a palpacao medica como metafora de perturbacao local.
- **Imagem mental:** mapa global complexo e reta desenhada apenas no quarteirao do paciente.

## 📊 Elementos de Comunidade e Status

- **Status:** `Auditoria local realizada` quando o aluno comparar previsao original, vizinhanca e fidelidade.
- **Debate:** “Se a explicacao muda com a semente, o que exatamente podemos afirmar?”
- **Papel rotativo:** paciente, modelo caixa-preta, explicador e auditor de estabilidade.

## 💡 Engajamento e Conhecimento

- **Laboratorio social:** cada grupo usa um `sigma` diferente e compara coeficientes locais.
- **Produto da aula:** laudo com impacto, proximidade, estabilidade e possivel combinacao clinicamente impossivel.
- **Conexao profissional:** criar uma regra de quando uma explicacao instavel deve ser recusada.

## Mapa da aula

1. [Subcamada 07.1: O conceito na vida real](#subcamada-071-o-conceito-na-vida-real)
2. [Subcamada 07.2: Desenhando o conceito](#subcamada-072-desenhando-o-conceito)
3. [Subcamada 07.3: Desmistificando a teoria e a notacao formal](#subcamada-073-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 07.4: Laboratorio ludico no Colab](#subcamada-074-laboratorio-ludico-no-colab)
5. [Subcamada 07.5: O momento serio da nossa aplicacao](#subcamada-075-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 07.6: Checkpoint de autonomia e fixacao ativa](#subcamada-076-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 07.1: O conceito na vida real

### A analogia da Terra redonda e a regua da sala

A superficie da Terra e esferica, irregular e cheia de relevos complexos. Se um engenheiro tentar estender uma viga de aco retilinea de mil quilometros sobre a superficie, a curvatura do planeta impedira qualquer encaixe perfeito. No entanto, se o engenheiro estiver construindo o piso de uma sala cirurgica de dez metros quadrados, a curvatura terrestre e virtualmente imperceptivel. Naquela micro-regiao, o plano local pode ser tratado com precisao absoluta como perfeitamente reto e plano.

O algoritmo LIME (Local Interpretable Model-agnostic Explanations, proposto por Ribeiro, Singh e Guestrin em 2016) parte dessa exata premissa geometrica:

- **Globalmente:** um modelo de aprendizado de maquina (como uma floresta aleatoria com centenas de arvores profundas) gera fronteiras de decisao nao-lineares, cheias de quebras, degraus e interacoes de alta ordem. Tentar resumir o comportamento global inteiro em uma unica formula linear e impossivel sem perder a fidelidade.
- **Localmente:** quando se faz um zoom microscopico imediatamente ao redor de um unico paciente especifico, a fronteira do modelo comporta-se de forma suave e linear. Um modelo simples e inerentemente interpretavel (uma regressao linear ou arvore rasa ponderada) consegue mapear perfeitamente a decisao naquela vizinhanca restrita.

### A analogia clinica da palpacao medica

Quando um medico suspeita de inflamacao em uma regiao abdominal, ele nao desmonta o corpo do paciente. Ele realiza uma palpacao com pequenas pressoes pontuais:
- Pressiona um centimetro a esquerda do umbigo: sem desconforto.
- Pressiona dois centimetros abaixo: dor intensa relatada.
- Pressiona cinco centimetros acima: sem resposta relevante.

Ao "cutucar" a vizinhanca imediata e registrar as respostas, o clinico mapeia a sensibilidade local. O LIME executa esse procedimento de forma numerica: gera milhares de pequenas perturbacoes no prontuario do paciente, consulta as probabilidades que o modelo complexo atribui a cada perturbacao e ajusta uma reta local ponderada pela proximidade do paciente original.

**A grande sacada:** o LIME nao precisa saber como o modelo complexo calcula internamente suas decisoes; ele apenas precisa de permissao para enviar entradas perturbadas e observar as saidas probabilisticas resultantes (*black-box agnostic*).

| Dimensao | Explicacao Global (SHAP TreeExplainer / Gini) | Explicacao Local (LIME) |
| :--- | :--- | :--- |
| **Escopo de analise** | Toda a populacao e toda a base de treino | Um unico paciente individual selecionado |
| **Complexidade da fronteira** | Nao-linear, com descontinuidades e interacoes | Aproximacao linear tangente a vizinhanca local |
| **Metodologia basica** | Teoria dos jogos cooperativos e valores de Shapley | Perturbacao estocastica e regressao ponderada por kernel |
| **Tempo computacional** | Escala com numero total de amostras e arvores | Escala com numero de perturbacoes locais geradas |
| **Papel na auditoria clinica** | Compreender o modelo como protocolo hospitalar | Justificar o veredito perante a equipe medica e o paciente |

---

## Subcamada 07.2: Desenhando o conceito

O diagrama abaixo ilustra a diferenca entre a complexidade global do modelo e a simplificacao linear obtida pelo LIME na vizinhanca imediata de uma instancia:

```text
SUPERFICIE GLOBAL DO MODELO (COMPLEXA)        ZOOM NA VIZINHANCA DO PACIENTE
Probabilidade P(y=1)                           (APROXIMACAO LINEAR DO LIME)

  1.0 |      _.-'''-._                           1.0 |           / (Fronteira Local)
      |    .'         `.                             |          /   
      |   /             \                            |       o /  + (Clone perturbado)
  0.5 |  /               \                       0.5 |      . X .   (PACIENTE ALVO)
      | /                 \                          |     + / o
  0.0 |/                   \__                   0.0 |    /
      +------------------------                      +------------------------
           Espaco de Features                             Vizinhanca Local Imediata
      (Fronteira com degraus e dobras)               (Reta simples: g(z) = w0 + w1*z)
```

O fluxo estocastico do algoritmo organiza-se em quatro fases consecutivas:

```text
[ PACIENTE DE TESTE (x) ]
(Idade: 62, Glicemia: 140, Troponina: 0.8)
       |
       v
[ GERADOR DE PERTURBACOES ] ---------> Cria N amostras sinteticas (z) com ruido
       |                               adicionado em torno das medias e desvios.
       v
[ CONSULTA AO MODELO BLACK-BOX ] ----> Modelo complexo f(z) atribui P(Patologia)
       |                               para cada uma das N amostras geradas.
       v
[ PESO DE DISTANCIA (KERNEL PI) ] ---> Amostras coladas em x recebem peso proximo de 1.
       |                               Amostras distantes recebem peso proximo de 0.
       v
[ AJUSTE DO MODELO LINEAR LOCAL ] ---> Treina regressao linear ponderada minimizando
       |                               a divergencia local entre f(z) e g(z).
       v
[ LAUDO DE JUSTIFICATIVA LOCAL ] ----> Coeficientes lineares locais indicam quais
                                       exames empurraram a probabilidade para patologia.
```

| Componente | Papel Operacional no Algoritmo | Comportamento se Desajustado |
| :--- | :--- | :--- |
| **Instancia alvo ($x$)** | Ponto de ancoragem da auditoria | N/A |
| **Amostras perturbadas ($z'$)** | Sondagem do espaco ao redor do paciente | Poucas amostras geram coeficientes instaveis |
| **Kernel gaussiano ($\pi_x$)** | Atribui peso maximo a proximidade e minimo a distancia | Largura excessiva perde localidade; estreita causa sobreajuste |
| **Modelo explicador ($g$)** | Regressao linear interpretavel de dimensao reduzida | Deve permanecer linear para compreensao humana direta |

---

## Subcamada 07.3: Desmistificando a teoria e a notacao formal

### A formulacao da explicabilidade local

O LIME define a explicacao local como a solucao de um problema de otimizacao com dois termos concorrentes: fidelidade local e complexidade do explicador:

$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

Em que cada componente representa:

$$\mathcal{L}(f, g, \pi_x) = \sum_{z, z' \in \mathcal{Z}} \pi_x(z) \left( f(z) - g(z') \right)^2$$

### Ponderacao por kernel gaussiano de distancia

Para garantir que a explicacao reflita com rigor apenas o que acontece perto do paciente original, o peso $\pi_x(z)$ de cada instancia perturbada $z$ decai exponencialmente conforme a distancia euclidiana normalizada $D(x, z)$ aumenta:

$$\pi_x(z) = \exp\left( - \frac{D(x, z)^2}{\sigma^2} \right)$$

Se a distancia for nula ($z = x$), temos $\exp(0) = 1.0$ (peso maximo). A medida que $z$ se distancia no espaco euclidiano multidimensional, $\pi_x(z) \to 0$, impedindo que dados de regioes remotas contaminem a auditoria.

### Por que auditar o paciente no limiar de decisao de 50%?

No contexto do projeto, os pacientes com probabilidades extremas (ex: $P=0.99$ ou $P=0.01$) possuem margem de seguranca confortavel; pequenas variacoes em atributos secundarios nao alteram a conduta clinica. 

O caso mais sensivel e de maior risco juridico e diagnostico ocorre no fio da navalha: o paciente cuja probabilidade estimada aproxima-se de $0.50$:

$$i^* = \arg\min_{i \in \text{Teste}} \left| P(\hat{y}_i = 1 | x_i) - 0.50 \right|$$

Se esse paciente receber a classificacao de doente ($P=0.503$), a decisao pode ter sido determinada por um biomarcador legitimo ou por uma leve oscilacao de ruido irrelevante. Auditar a instancia de incerteza maxima permite verificar se a margem que virou o diagnostico decorre de variaveis clinicamente solidas.

| Simbolo | Significado Formal | Leitura no Contexto Clinico |
| :--- | :--- | :--- |
| $f(z)$ | Funcao de probabilidade do modelo original | Chance de patologia calculada pela Random Forest |
| $g(z')$ | Modelo explicador simples pertencente a classe $G$ | Regressao linear interpretavel com poucos coeficientes |
| $\pi_x(z)$ | Funcao de proximidade baseada em kernel | Peso atribuido a cada clone sintetico do paciente |
| $\Omega(g)$ | Medida de complexidade do modelo explicador | Restricao no numero maximo de features no laudo (ex: $K \le 8$) |
| $\sigma$ | Largura de banda do kernel exponencial | Raio da vizinhanca considerado relevante ao redor do paciente |
| $i^*$ | Indice da amostra de incerteza maxima | Paciente com probabilidade mais proxima de 50% |

### A ordem correta evita vazamento e explicacoes invalidas

1. O modelo complexo de aprendizado supervisionado deve ser treinado estritamente sobre a particao de treino.
2. O conjunto de teste e submetido a predicao probabilistica sem qualquer contato com o explicador.
3. Identifica-se a instancia alvo no conjunto de teste com base na probabilidade prevista.
4. As perturbacoes sinteticas sao geradas tomando como base as distribuicoes empíricas aprendidas do treino, evitando expor dados externos.

---

## Subcamada 07.4: Laboratorio ludico no Colab

Execute o bloco abaixo no Google Colab para construir uma demonstracao minimalista de auditoria local baseada em perturbacao e regressao ponderada:

```python
# =============================================================================
# CAMADA 07: LABORATORIO LUDICO DE EXPLICABILIDADE LOCAL
# Demonstracao: Perturbacao Estocastica e Ajuste Linear de Vizinhanca
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

np.random.seed(42)

# 1. Geracao de base sintetica com 3 atributos e 300 pacientes
n_amostras = 300
X = np.random.normal(loc=0.0, scale=1.0, size=(n_amostras, 3))
nomes_atributos = ["Marcador_A", "Marcador_B", "Ruido_Metabolico"]

# Regra: Patologia depende fortemente de Marcador_A e moderadamente de Marcador_B
logit = 1.8 * X[:, 0] - 1.2 * X[:, 1] + 0.1 * X[:, 2]
prob_real = 1.0 / (1.0 + np.exp(-logit))
y = (prob_real >= 0.50).astype(int)

# 2. Treinamento do modelo complexo
modelo = RandomForestClassifier(n_estimators=60, max_depth=5, random_state=42)
modelo.fit(X, y)

# 3. Localizacao do paciente mais proximo do limiar de 50%
probas = modelo.predict_proba(X)[:, 1]
idx_limiar = int(np.argmin(np.abs(probas - 0.50)))
x_alvo = X[idx_limiar]
p_alvo = probas[idx_limiar]

print(f"Paciente alvo selecionado: Indice #{idx_limiar}")
print(f"Probabilidade estimada de patologia: {p_alvo * 100:.2f}%\n")

# 4. Geracao de perturbacoes locais ao redor do paciente alvo
n_perturbacoes = 2000
ruido = np.random.normal(loc=0.0, scale=0.35, size=(n_perturbacoes, 3))
X_perturbado = x_alvo + ruido

# Predicao do modelo complexo sobre os dados perturbados
y_perturbado = modelo.predict_proba(X_perturbado)[:, 1]

# Ponderacao das amostras por kernel exponencial da distancia
distancias = np.linalg.norm(X_perturbado - x_alvo, axis=1)
sigma = 0.50
pesos = np.exp(- (distancias ** 2) / (sigma ** 2))

# 5. Ajuste do modelo linear local interpretavel (Ridge Regression)
reg_local = Ridge(alpha=1.0)
reg_local.fit(X_perturbado, y_perturbado, sample_weight=pesos)

coeficientes = reg_local.coef_

# 6. Visualizacao grafica do laudo explicativo local
plt.figure(figsize=(9, 4.5))
cores = ["#c0392b" if c > 0 else "#27ae60" for c in coeficientes]
barras = plt.barh(nomes_atributos, coeficientes, color=cores, edgecolor="black", height=0.55)
plt.axvline(0, color="black", linestyle="--", linewidth=1.0)
plt.title(f"Laudo Explicativo Local - Paciente no Limiar (P = {p_alvo*100:.1f}%)", fontsize=11, fontweight="bold")
plt.xlabel("Coeficiente Local (Impacto na Probabilidade)", fontsize=10)
plt.grid(axis="x", linestyle=":", alpha=0.6)

for barra, coef in zip(barras, coeficientes):
    pos = coef + 0.005 if coef >= 0 else coef - 0.025
    plt.text(pos, barra.get_y() + 0.18, f"{coef:+.3f}", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()

print("Interpretacao dos coeficientes:")
for nome, c in zip(nomes_atributos, coeficientes):
    efeito = "Eleva a suspeita de patologia" if c > 0 else "Reduz a suspeita de patologia"
    print(f"  - {nome.ljust(18)}: {c:+.4f} -> {efeito}")
```

> **O que voce deve notar no grafico gerado:**
> 1. Os coeficientes lineares locais quantificam o peso de cada feature especificamente para aquele paciente; valores positivos empurram o veredito em direcao a patologia, enquanto valores negativos reduzem o risco.
> 2. A variavel irrelevante (`Ruido_Metabolico`) recebe coeficiente proximo de zero na vizinhanca, provando que o modelo local preservou a distincao entre sinal clinico e ruido.

**Mini-experimento:** altere a variavel `sigma` de `0.50` para `3.00`. O que acontece com os coeficientes locais? Eles se aproximam ou se distanciam do comportamento global da base inteira?

---

## Subcamada 07.5: O momento serio da nossa aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

No projeto de reducao de dimensionalidade hospitalar, executamos a auditoria LIME no cenario completo com 40 atributos, localizando com precisao matematica a instancia do conjunto de teste que habita a regiao de incerteza maxima.

```python
# =============================================================================
# APLICACAO REAL: AUDITORIA FORENSE LIME NO PACIENTE DO LIMIAR CRITICO
# Base oficial: 2.000 pacientes, 40 atributos clinicos
# =============================================================================
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

# 1. Construcao do dataset oficial padronizado
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

# 2. Treinamento da floresta de referencia
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf.fit(X_train, y_train)

# 3. Localizacao do paciente critico no limiar de 50%
y_probas_test = rf.predict_proba(X_test)[:, 1]
diferenca_limiar = np.abs(y_probas_test - 0.50)
idx_relativo = int(np.argmin(diferenca_limiar))
prob_limiar = y_probas_test[idx_relativo]
paciente_alvo = X_test.iloc[idx_relativo].values
classe_predita = int(prob_limiar >= 0.50)
classe_real = int(y_test[idx_relativo])

print("=" * 72)
print("AUDITORIA FORENSE LOCAL: INSTANCIA DE DECISAO NO LIMIAR CRITICO")
print("=" * 72)
print(f"Paciente avaliado no teste : Amostra #{idx_relativo}")
print(f"Probabilidade de patologia : {prob_limiar * 100:.2f}%")
print(f"Classificacao operacional  : {'PATOLOGIA (1)' if classe_predita == 1 else 'SAUDAVEL (0)'}")
print(f"Diagnostico real verificado: {'PATOLOGIA (1)' if classe_real == 1 else 'SAUDAVEL (0)'}")
print("-" * 72)

# 4. Geracao de perturbacoes e aproximacao local com cronometragem
t0 = time.perf_counter()

n_perturbacoes = 5000
media_train = X_train.mean().values
desvio_train = X_train.std().values

# Perturbacoes gaussianas baseadas na escala real de cada exame
ruido_ponderado = np.random.normal(loc=0.0, scale=0.40, size=(n_perturbacoes, 40)) * desvio_train
amostras_sinteticas = paciente_alvo + ruido_ponderado

# Predicao da probabilidade do modelo original nas amostras sinteticas
probas_sinteticas = rf.predict_proba(amostras_sinteticas)[:, 1]

# Ponderacao de proximidade por kernel gaussiano
distancias = np.linalg.norm((amostras_sinteticas - paciente_alvo) / (desvio_train + 1e-8), axis=1)
sigma = np.sqrt(40) * 0.75
pesos_kernel = np.exp(- (distancias ** 2) / (sigma ** 2))

# Ajuste da regressao linear local interpretavel
explicador_linear = Ridge(alpha=1.0)
explicador_linear.fit(amostras_sinteticas, probas_sinteticas, sample_weight=pesos_kernel)
tempo_auditoria_ms = (time.perf_counter() - t0) * 1000

# Coeficientes das 8 features com maior magnitude de impacto
coefs = explicador_linear.coef_
indices_top8 = np.argsort(np.abs(coefs))[-8:][::-1]

# 5. Apresentacao formal do laudo
print(f"Tempo de processamento da auditoria: {tempo_auditoria_ms:.2f} ms")
print(f"Amostras estocasticas geradas      : {n_perturbacoes}")
print("-" * 72)
print("LAUDO LOCAL: AS 8 VARIAVEIS DETERMINANTES PARA ESTE CASO:")
print("-" * 72)

qtd_biomarcadores = 0
for rank, idx in enumerate(indices_top8, 1):
    nome_f = feature_names[idx]
    peso_f = coefs[idx]
    efeito = "AUMENTA RISCO" if peso_f > 0 else "REDUZ RISCO  "
    
    if "biomarcador" in nome_f:
        tipo = "Biomarcador Informativo"
        qtd_biomarcadores += 1
    elif "redundante" in nome_f:
        tipo = "Exame Redundante"
    else:
        tipo = "Ruido Nao-Correlacionado"
        
    print(f"  {rank}. {nome_f.ljust(18)}: {peso_f:+.4f} | {efeito} | [{tipo}]")

print("=" * 72)
print(f"Prevalencia de biomarcadores informativos no laudo: {qtd_biomarcadores} de 8.")
```

### Tabela oficial de KPIs

> Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Como e calculado | Pergunta operacional |
| :--- | :--- | :--- |
| **Probabilidade no Limiar** | $\min_i \|P_i - 0.50\|$ | Qual paciente habita o ponto de maxima indecisao do modelo? |
| **Fidelidade Local ($R^2$ do LIME)** | Coeficiente de determinacao ponderado entre $f(z)$ e $g(z)$ | Quao bem a reta simples consegue imitar o modelo complexo na vizinhanca? |
| **Prevalencia de Biomarcadores no Laudo** | Proporcao de atributos informativos entre as maiores magnitudes | A decisao no limiar foi orientada por biologia real ou por ruido espurio? |
| **Latencia de Auditoria (ms)** | Medida direta com `time.perf_counter()` para $N=5.000$ perturbacoes | E compativel com o tempo de espera em um consultorio clinico? |

### Interpretacao clinica e de negocio

A auditoria no limiar de 50% atua como mecanismo de protecao etica e operacional em instituicoes de saude:

1. **Evitar intervencao invasiva por oscilacao de ruido:** quando a probabilidade predita atinge $50.3\%$, o protocolo do sistema classifica o paciente como doente. Se a explicacao local comprovar que as maiores influencias positivas vieram de variaveis de ruido, a equipe medica e alertada para repetir os exames antes de submeter o paciente a um tratamento doloroso ou de alto custo.
2. **Prevenir o Falso Negativo camuflado:** se a probabilidade for de $49.7\%$, o paciente receberia alta automatica. A auditoria local permite ao medico verificar se biomarcadores severos foram compensados por artefatos tecnicos, evitando a liberacao indevida de um paciente com patologia grave.
3. **Auditoria perante comites reguladores:** o algoritmo de explicabilidade local fornece um documento reprodutivel que justifica perante auditorias de conformidade (como HIPAA ou LGPD) os exames determinantes para cada decisao individual de alta complexidade.

---

## Subcamada 07.6: Checkpoint de autonomia e fixacao ativa

Explique sem consultar o texto e depois confira sua resposta:

1. **Como o principio de aproximacao linear por vizinhanca permite explicar modelos com fronteiras altamente nao-lineares?**
2. **Qual e a funcao matematica do kernel gaussiano de distancia $\pi_x(z)$ no calculo do LIME?**
3. **Por que o LIME precisa gerar perturbacoes no espaco de features em vez de consultar novos dados de pacientes reais?**
4. **Qual e a justificativa tecnica para selecionar a instancia com probabilidade mais proxima de 50% para a auditoria principal?**
5. **O que ocorre com a estabilidade e o tempo de execucao dos coeficientes locais se reduzirmos o numero de perturbacoes de 5.000 para 100?**
6. **Qual e a diferenca conceitual entre o papel do LIME (explicador local) e o SHAP TreeExplainer (explicador aditivo baseado em teoria dos jogos)?**

### Mini-desafio pratico

Execute a simulacao com diferentes valores para a dispersao do ruido das perturbacoes (parametro `scale` da distribuicao normal) e registre os impactos observados na tabela:

| Dispersao do Ruido (`scale`) | Amostras Sinteticas ($N$) | Tempo de Execucao (ms) | Prevalencia de Biomarcadores no Top-8 |
| :--- | :--- | :--- | :--- |
| `scale = 0.10` (Hiper-local) | 5.000 | | |
| `scale = 0.40` (Padrao) | 5.000 | | |
| `scale = 1.50` (Regional) | 5.000 | | |

**Pergunta reflexiva:** se o ruido da perturbacao for excessivamente amplo, o modelo local continua sendo uma representacao confiavel da condicao individual do paciente, ou passa a capturar caracteristicas heterogeneas de outras patologias?
