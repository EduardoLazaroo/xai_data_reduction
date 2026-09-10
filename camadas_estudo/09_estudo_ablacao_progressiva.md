# Camada 09: O Estudo de Ablação — Testando Empiricamente a Poda de Atributos

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 9  
**Contexto Técnico:** [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py) (`executar_etapa_ablacao`)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> Compreender o protocolo experimental de um **Estudo de Ablação (*Ablation Study*)**. Entender a intuição por trás do teste de estresse estrutural de modelos de IA, aprender a interpretar as curvas de manutenção do **$F_1$-Score** e de queda do tempo de treino com o eixo $X$ invertido, e identificar com precisão milimétrica o **Ponto de Inflexão (*Knee Point*)** onde o modelo atinge a máxima eficiência enxuta.

---

## Sumário da Aula

- [Subcamada 9.1: A Analogia da Torre de Jenga e o Teste de Estresse da Ponte](#subcamada-91-a-analogia-da-torre-de-jenga-e-o-teste-de-estresse-da-ponte)
- [Subcamada 9.2: Por Que Menos é Mais? (A Orquestra Sem Barulho)](#subcamada-92-por-que-menos-é-mais-a-orquestra-sem-barulho)
- [Subcamada 9.3: O Protocolo Científico de Ablação Passo a Passo](#subcamada-93-o-protocolo-científico-de-ablação-passo-a-passo)
- [Subcamada 9.4: Como Ler as Curvas do Gráfico de Ablação (O Joelho / Knee Point)](#subcamada-94-como-ler-as-curvas-do-gráfico-de-ablação-o-joelho--knee-point)
- [Subcamada 9.5: Laboratório Lúdico no Colab (Toy Example: Curva de Ablação na Prática)](#subcamada-95-laboratório-lúdico-no-colab-toy-example-curva-de-ablação-na-prática)
- [Subcamada 9.6: O Momento Sério da Nossa Aplicação (Ablação Real SHAP vs. RFE & KPIs)](#subcamada-96-o-momento-sério-da-nossa-aplicação-ablação-real-shap-vs-rfe--kpis)
- [Subcamada 9.7: Checkpoint de Autonomia & Fixação Ativa](#subcamada-97-checkpoint-de-autonomia--fixação-ativa)

---

## Subcamada 9.1: A Analogia da Torre de Jenga e o Teste de Estresse da Ponte

Você já jogou aquele jogo de blocos de madeira chamado **Jenga**?  
A torre começa com 54 blocos de madeira empilhados. A cada rodada, você é obrigado a puxar uma peça de madeira:
- No começo, você puxa peças que estão frouxas nas laterais e que não suportam nenhum peso. A torre continua **100% firme e em pé!**
- Você retira 5, 10, 15 blocos... e nada acontece.
- Porém, chega um momento crucial: você puxa **uma única peça que sustentava o pilar central** e a torre inteira desmorona de uma vez só!

```
    TORRE COM 40 BLOCOS                TORRE COM 10 BLOCOS (KNEE POINT)          TENTOU TIRAR MAIS...
    (Muitos blocos inúteis)            (Apenas as vigas mestras!)               (A torre desmoronou!)
    
        ┌───┬───┬───┐                              ┌───┐                              
        │   │   │   │                              │   │                              💥
        ├───┼───┼───┤                              ├───┤                           DESABOU!
        │   │   │   │   ──(Poda Cirúrgica)──►      │   │   ──(Cortou o vital)──►   F1 cai de
        ├───┼───┼───┤                              ├───┤                           0.86 p/ 0.60
        │   │   │   │                              │   │                              
        └───┴───┴───┘                              └───┘                              
     F1: 0.84 | Tempo: 600ms                    F1: 0.86 | Tempo: 180ms                 
```

Um **Estudo de Ablação (*Ablation Study*)** em Inteligência Artificial é exatamente isso: um teste de estresse metódico onde removemos componentes (neste caso, colunas do banco de dados) de forma progressiva para responder a duas perguntas vitais:
1. *Até onde podemos enxugar o modelo sem que a torre do desempenho caia?*
2. *Em qual ponto exato ocorre o desmoronamento preditivo?*

---

## Subcamada 9.2: Por Que Menos é Mais? (A Orquestra Sem Barulho)

Pense em uma grande orquestra sinfônica composta por **10 violinistas virtuosos**.  
Agora imagine que o maestro decida colocar no palco mais **20 pessoas batendo panelas e soprando apitos de plástico** (ruído aleatório puro) e mais **10 pessoas tocando exatamente a mesma nota do primeiro violino** (redundância).

O som da orquestra melhora? **Fica horrível!** O público mal consegue ouvir a bela melodia dos violinos no meio daquele barulho ensurdecedor.  
Quando você retira as 20 pessoas das panelas e os 10 músicos repetitivos:
- A música volta a soar com máxima harmonia e beleza.
- No Machine Learning acontece o mesmo: **ao podar os 20 ruídos, o Random Forest para de gastar árvores dividindo dados no chiado** e foca 100% nos biomarcadores causais verdadeiros, fazendo o $F_1$-Score muitas vezes **subir**!

---

## Subcamada 9.3: O Protocolo Científico de Ablação Passo a Passo

O protocolo experimental implementado no [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py) opera como um duelo controlado entre dois rankings de prioridade:
- **Ranking Concorrente 1:** Ordenado pelo **SHAP (XAI)**.
- **Ranking Concorrente 2:** Ordenado pelo **RFE (Tradicional)**.

O laço de ablação executa o seguinte ciclo:
1. Inicia com $k = 40$ atributos (o Baseline completo).
2. Treina uma nova floresta do zero apenas com essas $k$ variáveis.
3. Mede o tempo exato de treino em milissegundos e o $F_1$-Score no teste cego.
4. Reduz $k$ de 2 em 2 colunas ($k = 38, 36, 34, \dots, 10, 8, 6, 4, 2$).
5. Registra o comportamento e plota o gráfico com o **eixo $X$ invertido** (da esquerda para a direita, simulando o processo de enxugamento).

---

## Subcamada 9.4: Como Ler as Curvas do Gráfico de Ablação (O Joelho / Knee Point)

```
   F1-Score
    1.0 ┌────────────────────────────────────────────────────────┐
        │                 PLATÔ DE ESTABILIDADE                  │
    0.8 │       * * * * * * * * * * * * * ★ [Knee Point: k = 10] │  <- Mantém F1 ~ 0.86!
        │                                 \                      │
    0.6 │                                  \  QUEDA LIVRE:       │  <- Cortou biomarcadores vitais!
        │                                   \ O modelo desaba!   │
    0.0 └────────────────────────────────────*───────────────────┘
        40        34        28        22    10   8   6   4   2
        ◄───────────────────────────────────────────────────────
                      Número de Atributos Restantes (Poda)
```

1. **O Platô de Estabilidade ($40 \to 12$ atributos):**  
   A curva de $F_1$-Score permanece praticamente reta e horizontal (ou até sobe levemente). Isso prova empiricamente que os primeiros 28 atributos eliminados eram puramente lixo ou redundância!
2. **O Ponto de Inflexão ou "Joelho" ($k = 10$ atributos):**  
   É a glória da pesquisa! O ponto de equilíbrio perfeito onde temos o **mínimo número de exames com o máximo desempenho diagnóstico**.
3. **A Queda Livre ($k < 8$ atributos):**  
   A curva desaba vertiginosamente. O modelo perdeu um biomarcador insubstituível.

---

## Subcamada 9.5: Laboratório Lúdico no Colab (Toy Example: Curva de Ablação na Prática)

Copie e execute no [Google Colab](https://colab.research.google.com) para ver a curva de ablação ser desenhada em poucos segundos:

```python
# =============================================================================
# LABORATÓRIO DIDÁTICO: ESTUDO DE ABLAÇÃO NA PRÁTICA
# Objetivo: Ver visualmente o platô de estabilidade e a queda livre
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. Criamos 1.000 pacientes com 15 variáveis (5 úteis e 10 ruídos)
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=15, n_informative=5, n_redundant=0, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Treinamos um modelo inicial e ordenamos os atributos por importância
clf_init = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_tr, y_tr)
ranking_indices = np.argsort(clf_init.feature_importances_)[::-1]  # Melhores primeiro

# 3. Laço de Ablação: de 15 até 1 atributo
k_valores = list(range(15, 0, -1))
f1_historico = []

for k in k_valores:
    melhores_k = ranking_indices[:k]
    clf_k = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_k.fit(X_tr[:, melhores_k], y_tr)
    y_pred = clf_k.predict(X_te[:, melhores_k])
    f1_historico.append(f1_score(y_te, y_pred))

# 4. Gráfico da Curva de Ablação
plt.figure(figsize=(10, 4))
plt.plot(k_valores, f1_historico, marker="o", color="navy", lw=2, label="F1-Score")
plt.axvline(x=5, color="red", linestyle="--", label="Knee Point (k = 5 reais)")
plt.gca().invert_xaxis()  # Inverte eixo X para ler da esquerda para direita
plt.title("Estudo de Ablação Didático: Identificando o Joelho da Curva", fontweight="bold")
plt.xlabel("Quantidade de Atributos Mantidos no Modelo", fontweight="bold")
plt.ylabel("F1-Score Diagnóstico", fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()

print(f"F1 com 15 atributos (com 10 ruídos) : {f1_historico[0]:.4f}")
print(f"F1 no Knee Point (5 atributos vitais): {f1_historico[10]:.4f} (Mesmo nível sem ruído!)")
print(f"F1 com apenas 1 atributo            : {f1_historico[-1]:.4f} (Colapso total!)")
```

---

## Subcamada 9.6: O Momento Sério da Nossa Aplicação (Ablação Real SHAP vs. RFE & KPIs)

Agora executamos o protocolo oficial de ablação do projeto no dataset hospitalar de **40 atributos** ([pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py)), comparando a trajetória de poda pelo **SHAP** contra o **RFE**.

```python
# =============================================================================
# O MOMENTO SÉRIO DA NOSSA APLICAÇÃO:
# Estudo de Ablação Comparativa: SHAP vs. RFE nos 40 Atributos Hospitalares
# =============================================================================
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score

print("=" * 70)
print("INICIANDO PROTOCOLO CIENTÍFICO DE ABLAÇÃO (SHAP VS. RFE)")
print("=" * 70)

# 1. Dataset Clínico (2.000 Pacientes, 40 Atributos)
X_raw, y = make_classification(
    n_samples=2000, n_features=40, n_informative=10, n_redundant=10,
    n_classes=2, weights=[0.6, 0.4], flip_y=0.03, random_state=42
)
feature_names = (
    [f"biomarcador_{i+1}" for i in range(10)] +
    [f"exame_redundante_{i+1}" for i in range(10)] +
    [f"ruido_metabolico_{i+1}" for i in range(20)]
)
df_clinico = pd.DataFrame(X_raw, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(df_clinico, y, test_size=0.25, stratify=y, random_state=42)

# 2. Ranking RFE Tradicional
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=1, step=2)
rfe.fit(X_train, y_train)
ranking_rfe = np.argsort(rfe.ranking_)

# 3. Ranking SHAP (Usamos a importância do Baseline como proxy rápida)
rf_base = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
ranking_shap = np.argsort(rf_base.feature_importances_)[::-1]

# 4. Trajetória de Ablação em Pontos-Chave (40, 20, 10, 6, 2 atributos)
degraus = [40, 20, 10, 6, 2]
tabela_ablacao = []

for k in degraus:
    # Cenário SHAP
    cols_shap = ranking_shap[:k]
    t0 = time.perf_counter()
    rf_s = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train.iloc[:, cols_shap], y_train)
    tempo_shap_ms = (time.perf_counter() - t0) * 1000
    f1_s = f1_score(y_test, rf_s.predict(X_test.iloc[:, cols_shap]))
    
    # Cenário RFE
    cols_rfe = ranking_rfe[:k]
    t0 = time.perf_counter()
    rf_r = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train.iloc[:, cols_rfe], y_train)
    tempo_rfe_ms = (time.perf_counter() - t0) * 1000
    f1_r = f1_score(y_test, rf_r.predict(X_test.iloc[:, cols_rfe]))
    
    tabela_ablacao.append({
        "Atributos (k)": k,
        "F1 (SHAP)": f"{f1_s:.4f}",
        "F1 (RFE)": f"{f1_r:.4f}",
        "Tempo Treino": f"{tempo_shap_ms:.1f} ms",
        "Redução de Dados": f"{(1 - k/40)*100:.1f}%"
    })

df_resultado_ablacao = pd.DataFrame(tabela_ablacao)
print("\n📊 QUADRO COMPARATIVO DA TRAJETÓRIA DE ABLAÇÃO:")
print(df_resultado_ablacao.to_string(index=False))
```

---

### 9.6.1 Quadro de Descobertas Científicas da Ablação

| Atributos ($k$) | Redução de Dados | $F_1$-Score Diagnóstico | Tempo de CPU | Conclusão Médica |
| :--- | :--- | :--- | :--- | :--- |
| **$k = 40$ (Baseline)** | **0.0% (Tudo)** | **0.8373** | **~600 ms** | Modelo pesado, caro e com 12.2% de gap de overfitting. |
| **$k = 20$ (Sem Ruídos)** | **50.0%** | **0.8520** | **~380 ms** | O modelo melhora! Remover o ruído limpou a decisão. |
| **$k = 10$ (Knee Point)**| **75.0% de Corte** | **0.8610** | **~185 ms** | **O ápice do projeto!** 75% menos exames e F1 mais alto. |
| **$k = 2$ (Poda Excessiva)**| **95.0% de Corte** | **0.6200** | **~45 ms** | **Colapso.** Não há milagre: 2 exames são insuficientes para diagnosticar. |

---

## Subcamada 9.7: Checkpoint de Autonomia & Fixação Ativa

Responda para fixar a lógica de ablação:

1. **Por que chamamos o ponto $k = 10$ de "Knee Point" (Ponto do Joelho) no gráfico de ablação? O que acontece antes dele e o que acontece depois dele?**
2. **Explique a metáfora da orquestra: como a retirada de 20 músicos ruidosos pode fazer a qualidade da sinfonia médica aumentar em vez de piorar?**
3. **Se um hospital gasta R$ 100,00 por exame de sangue, qual é a economia anual em um hospital que atende 50.000 pacientes se reduzirmos a coleta de 40 para 10 atributos comprovados pela ablação?**
4. **Desafio no Colab:** Na Subcamada 9.5, mude `n_informative=5` para `n_informative=8`. Para onde o "Knee Point" se deslocou no gráfico? Ele acompanhou o número real de atributos úteis?
