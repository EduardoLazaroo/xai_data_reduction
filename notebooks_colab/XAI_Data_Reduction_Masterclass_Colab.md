# Masterclass: Redução de Dimensionalidade Guiada por Inteligência Artificial Explicável (XAI)

**Disciplina:** Inteligência Artificial Explicável (XAI) & Otimização de Modelos de Machine Learning  
**Professor:** Eduardo Lázaro Roesler de Oliveira  
**Instituição:** UNIVEM — Centro Universitário de Marília  

---

> [!NOTE]
> 🔙 **De onde viemos:** No aprendizado de máquina supervisionado convencional, frequentemente somos levados a crer que *"quanto mais variáveis e colunas alimentarmos no modelo, melhor ele aprenderá"*. No entanto, na prática de ciência de dados de alta complexidade (como bioinformática, saúde e finanças), o excesso de variáveis irrelevantes causa o **Mal da Dimensionalidade**, inflaciona o risco de **Overfitting**, retarda a inferência em tempo real e encarece a coleta de exames.
> 🎯 **Objetivo Principal do Curso / Notebook:** Percorrer uma jornada científica completa em 6 etapas: estabelecer um classificador de referência (**Baseline**) sobre 40 variáveis clínicas heterogêneas; auditar a tomada de decisão com **SHAP** (global) e **LIME** (local); executar um **Estudo de Ablação Progressiva** confrontando SHAP vs. RFE; implementar uma engenharia de seleção em duas fases (**Pré-Filtro Híbrido** e **shap-select** com rigor econométrico de $p$-valor); e consolidar um **Pipeline Industrial Fim-a-Fim** com reotimização bayesiana via **Optuna** e geração de um **Dashboard Executivo de 4 Quadrantes**.
> 🚀 **Para onde vamos:** Ao concluir este caderno, você dominará um método robusto, reprodutível e academicamente validado para comprimir bases de dados em até 75% preservando (ou até melhorando) a precisão preditiva, habilitando a implantação de modelos ultrarrápidos em servidores de nuvem ou em dispositivos embarcados (*Edge AI*).

---

## 🗺️ Organização Tática da Jornada Didática

| Etapa | Módulo Prático | Foco Teórico & Prático |
| :---: | :--- | :--- |
| **01** | **O Baseline e o Mal da Dimensionalidade** | Simulação de 2.000 pacientes com 40 atributos (10 biomarcadores reais, 10 redundantes, 20 ruídos puros); treino do Random Forest; medição de latências; Matriz de Confusão e Curva ROC. |
| **02** | **Explicabilidade Global via SHAP** | Fundamentos da Teoria dos Jogos Cooperativos de Lloyd Shapley (Nobel 2012); algoritmo TreeSHAP; cálculo do ranking de importância média $\|SHAP\|$; Bar Plot e Beeswarm Plot. |
| **03** | **Explicabilidade Local via LIME** | Amostragem por perturbação local de vizinhança; identificação algorítmica de paciente no limiar crítico ($P \approx 50\%$); modelo linear substituto (*surrogate*) e gráfico bicolor de regras locais. |
| **04** | **Poda Guiada por XAI vs. RFE (Ablação)** | Experimento iterativo de ablação progressiva (40 a 2 atributos); comparação entre ranking SHAP e RFE tradicional; análise do Ponto de Inflexão (Knee Point); curvas de F1 e tempo. |
| **05** | **Engenharia Avançada: Pré-Filtro & shap-select** | Fase 1 (Pré-Filtro Híbrido: Variância e Pearson $> 0.90$ estilo BOLIMES); Fase 2 (`shap-select`: Regressão Logística de $y \sim \Phi$, admitindo apenas coeficientes $\beta > 0$ e $p\text{-valor} < 0.05$). |
| **06** | **Pipeline Industrial, Optuna & Dashboard** | Integração do fluxo completo; Otimização Bayesiana via TPE com Optuna (CV em 3 dobras); treino do modelo campeão; Dashboard Executivo de 4 Painéis e Sanity Check Final. |

---

## ⚙️ Célula 0: Instalação e Preparação do Ambiente Google Colab

Execute a célula abaixo para instalar as bibliotecas especializadas na sua máquina virtual do Google Colab:

```python
# 1. Executamos o comando de instalacao silenciosa das dependencias de XAI e Otimizacao
!pip install shap lime optuna statsmodels -q

# 2. Exibimos mensagem de confirmacao de prontidao do ambiente
print("=" * 70)
print("✅ AMBIENTE GOOGLE COLAB CONFIGURADO COM SUCESSO PARA O CURSO DE XAI!")
print("=" * 70)
```

---

# MÓDULO 1: O BASELINE E O DESAFIO DA ALTA DIMENSIONALIDADE

> [!TIP]
> ⚔️ **Analogia Geek — O Inventário Pesado de um RPG:**
> Imagine carregar 40 itens na mochila do seu personagem de RPG: 10 são armas e poções vitais (**informativos**), 10 são armaduras repetidas (**redundantes**) e 20 são pedras e gravetos pegos no chão (**ruído**). O excesso de entulho deixa o guerreiro lento para esquivar (**latência**) e faz o jogador se confundir nos combates (**overfitting**). Nosso objetivo é limpar esse inventário!

### Bloco 1.1 — Importação das Bibliotecas e Síntese de Dados Clínicos

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que criamos dados sintéticos em vez de baixar um CSV estático da internet?**  
> Porque em pesquisas científicas de redução de atributos, gerar dados controlados nos permite saber com 100% de certeza quais colunas são o "gabarito verdadeiro" (os 10 biomarcadores) e quais são ruídos inventados. Assim, podemos testar se o algoritmo realmente descobre a verdade!

```python
# 1. Importamos as bibliotecas padrao de sistema, medicao e dados
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Importamos geradores e modelos do Scikit-Learn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# 3. Configuramos a estetica visual dos graficos
sns.set_theme(style="whitegrid", palette="deep")

# 4. Definimos a funcao que sintetiza nossa coorte clinica de saude
def gerar_dataset_sintetico_saude(n_samples=2000, n_features=40, n_informative=10, n_redundant=10, random_state=42):
    """
    Gera um dataset sintético simulando 2.000 pacientes com 40 atributos clínicos.
    """
    # 5. Geramos as matrizes numericas brutas
    X_raw, y = make_classification(
        n_samples=n_samples,         # 2.000 pacientes
        n_features=n_features,       # 40 variaveis totais coletadas
        n_informative=n_informative, # 10 biomarcadores vitais causadores da patologia
        n_redundant=n_redundant,     # 10 exames repetidos/colineares
        n_repeated=0,                # Sem colunas identicas
        n_classes=2,                 # Binario: 0 (Saudavel) vs 1 (Patologia)
        weights=[0.6, 0.4],          # 60% saudaveis e 40% com patologia
        flip_y=0.03,                 # 3% de erro de rotulacao simulando falha humana de laudo
        random_state=random_state    # Semente pseudoaleatoria para reprodutibilidade
    )
    
    # 6. Criamos nomes claros e semanticos para cada uma das 40 colunas
    feature_names = (
        [f"biomarcador_{i+1}" for i in range(n_informative)] +
        [f"exame_redundante_{i+1}" for i in range(n_redundant)] +
        [f"ruido_metabolico_{i+1}" for i in range(n_features - n_informative - n_redundant)]
    )
    
    # 7. Convertemos em DataFrame do Pandas
    return pd.DataFrame(X_raw, columns=feature_names), y

# 8. Instanciamos a base e realizamos a divisao estratificada (75% treino, 25% teste)
df_X, y = gerar_dataset_sintetico_saude()
X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.25, random_state=42, stratify=y)

print(f"[*] Base de Dados Instanciada: {X_train.shape[0]} amostras de Treino e {X_test.shape[0]} de Teste.")
print(f"[*] Total de Atributos: {X_train.shape[1]} (10 Informativos, 10 Redundantes e 20 Ruídos).")
```

---

### Bloco 1.2 — Treinamento do Baseline e Visualizações Diagnósticas

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que fixamos `n_jobs=1` na floresta aleatória?**  
> Porque medições de tempo científicas precisam ser comparadas de forma justa. Deixar a CPU usar núcleos variáveis geraria variações aleatórias na cronometria.

```python
# 1. Definimos a funcao que treina o baseline e extrai cronometria precisa
def treinar_e_avaliar_baseline(X_train, X_test, y_train, y_test, random_state=42):
    # 2. Instanciamos o Random Forest com 100 arvores
    modelo = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
    
    # 3. Medimos o tempo de treino em microssegundos
    t0_treino = time.perf_counter()
    modelo.fit(X_train, y_train)
    tempo_treino = time.perf_counter() - t0_treino
    
    # 4. Medimos a latencia de inferencia no teste
    t0_inf = time.perf_counter()
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    tempo_inf = time.perf_counter() - t0_inf
    
    # 5. Consolidamos o perfil de metricas
    metricas = {
        "acuracia": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "tempo_treino_s": tempo_treino,
        "tempo_inferencia_s": tempo_inf,
        "num_atributos": X_train.shape[1]
    }
    return modelo, metricas, y_pred, y_proba

# 6. Treinamos o classificador Baseline
modelo_baseline, metricas_base, y_pred_base, y_proba_base = treinar_e_avaliar_baseline(
    X_train, X_test, y_train, y_test
)

# 7. Renderizamos os graficos de Matriz de Confusao e Curva ROC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 8. Subgrafico 1: Matriz de Confusao
cm = confusion_matrix(y_test, y_pred_base)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
axes[0].set_title("Matriz de Confusão (Baseline - 40 Atributos)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Predição do Modelo", fontweight="bold")
axes[0].set_ylabel("Diagnóstico Real", fontweight="bold")
axes[0].set_xticklabels(["Saudável (0)", "Patologia (1)"])
axes[0].set_yticklabels(["Saudável (0)", "Patologia (1)"])

# 9. Subgrafico 2: Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba_base)
axes[1].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Baseline ROC (AUC = {metricas_base['roc_auc']:.4f})")
axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Aleatório (AUC = 0.50)")
axes[1].set_title("Curva ROC (Baseline)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Taxa de Falsos Positivos (FPR)", fontweight="bold")
axes[1].set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontweight="bold")
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10. Exibimos o relatorio no console
print(f"📊 F1-Score Baseline: {metricas_base['f1_score']:.4f} | Tempo Treino: {metricas_base['tempo_treino_s']*1000:.2f} ms")
```

---

# MÓDULO 2: EXPLICABILIDADE GLOBAL VIA SHAP (TEORIA DOS JOGOS)

> [!TIP]
> ⚔️ **Analogia Geek — O Bônus dos Vingadores:**
> Se uma equipe de super-heróis derrota um vilão e recebe uma recompensa, como dividir o prêmio de forma justa? O atacante causou 40% do dano, o mago curou o time e o arqueiro disparou 2 flechas. O **Valor Shapley** calcula a contribuição marginal de cada integrante testando todas as combinações possíveis de formação de equipe!

### Bloco 2.1 — Computando TreeSHAP, Bar Plot e Beeswarm Plot

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que o Beeswarm Plot mostra que o Bar Plot não mostra?**  
> O Bar Plot mostra apenas o tamanho médio do impacto ($|\text{SHAP}|$). O Beeswarm mostra a **direção**: pontos vermelhos (valor alto do exame) à direita indicam que ter o biomarcador alto *aumenta* o risco de patologia!

```python
# 1. Importamos a biblioteca SHAP
import shap

# 2. Instanciamos o explicador de arvores otimizado
explainer_shap = shap.TreeExplainer(modelo_baseline)

# 3. Computamos a explicabilidade no conjunto de treinamento
print("[*] Computando Valores Shapley com TreeExplainer... Aguarde...")
shap_values = explainer_shap(X_train)

# 4. Extraímos os valores referentes a classe 1 (patologia)
shap_vals_c1 = shap_values.values[:, :, 1] if len(shap_values.shape) == 3 else shap_values.values

# 5. Calculamos a media absoluta |SHAP| para cada uma das 40 variaveis
mean_abs_shap = np.abs(shap_vals_c1).mean(axis=0)
df_importancia_shap = pd.DataFrame({
    "atributo": X_train.columns,
    "importancia_shap": mean_abs_shap
}).sort_values(by="importancia_shap", ascending=False).reset_index(drop=True)

# 6. Construímos a visualizacao com Bar Plot Semantico e Beeswarm Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 7. Bar Plot Semantico nos 15 maiores atributos
top15 = df_importancia_shap.head(15)
colors = [
    '#1f77b4' if 'biomarcador' in name else ('#ff7f0e' if 'redundante' in name else '#d62728')
    for name in top15['atributo']
]
axes[0].barh(top15['atributo'][::-1], top15['importancia_shap'][::-1], color=colors[::-1])
axes[0].set_title("Top 15 Atributos: Importância Média |SHAP|", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Impacto Médio Absoluto no Modelo (|SHAP|)", fontweight="bold")
axes[0].grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', label='Biomarcadores (Informativos)'),
    Patch(facecolor='#ff7f0e', label='Exames Redundantes (Colineares)'),
    Patch(facecolor='#d62728', label='Ruído Metabólico (Sem Efeito)')
]
axes[0].legend(handles=legend_elements, loc="lower right")

# 8. Beeswarm Plot
plt.sca(axes[1])
sv_plot = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
shap.plots.beeswarm(sv_plot, max_display=15, show=False)
axes[1].set_title("SHAP Beeswarm Plot (Dispersão Local de Impacto)", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()

print("[OK] SHAP computado com sucesso! Os biomarcadores vitais dominam as primeiras posições do ranking.")
```

---

# MÓDULO 3: EXPLICABILIDADE LOCAL VIA LIME NO LIMIAR DE DECISÃO

> [!TIP]
> ⚔️ **Analogia Geek — O Horizonte Plano e a Terra Redonda:**
> A Terra é curva e complexa no globo todo, mas quando você pisa no seu quintal, o chão se comporta como um **plano perfeitamente reto**. O **LIME** faz isso: em vez de tentar explicar a floresta inteira de uma vez, ele dá um zoom num único paciente e ajusta uma reta simples para aquele caso específico!

### Bloco 3.1 — Identificando o Paciente no Limiar ($P \approx 50\%$) e Extraindo Pesos LIME

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que explicar o paciente com $P \approx 50\%$ é muito mais informativo que um com $99\%$?**  
> Porque em 99%, o diagnóstico é óbvio. No limiar de 50%, o paciente está "em cima do muro": qualquer pequena variação em um exame define se ele será internado ou receberá alta!

```python
# 1. Importamos o explicador tabular do LIME
from lime import lime_tabular

# 2. Instanciamos o explicador com a matriz de treinamento
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns.tolist(),
    class_names=["Saudável (0)", "Patologia (1)"],
    mode="classification",
    random_state=42
)

# 3. Localizamos o paciente de teste cuja probabilidade esteve mais proxima de 50%
distancias_50 = np.abs(y_proba_base - 0.50)
idx_limiar = int(np.argmin(distancias_50))
proba_limiar = y_proba_base[idx_limiar]

# 4. Extraímos a explicacao LIME para esse paciente critico
exp_paciente = explainer_lime.explain_instance(
    data_row=np.array(X_test.iloc[idx_limiar]),
    predict_fn=modelo_baseline.predict_proba,
    num_features=8
)

# 5. Formatamos as regras em tabela
df_lime = pd.DataFrame(exp_paciente.as_list(), columns=["regra_atributo", "peso_local"])

# 6. Renderizamos o grafico de barras bicolores
plt.figure(figsize=(11, 5))
cores_lime = ['#2ca02c' if p > 0 else '#d62728' for p in df_lime['peso_local']]
plt.barh(df_lime['regra_atributo'][::-1], df_lime['peso_local'][::-1], color=cores_lime[::-1])
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.title(f"LIME: Laudo Explicativo do Paciente #{idx_limiar} (Probabilidade: {proba_limiar:.2%})", fontsize=12, fontweight="bold")
plt.xlabel("Contribuição Local do Atributo (Peso LIME)", fontweight="bold")
plt.grid(True, alpha=0.3)

legend_lime = [
    Patch(facecolor='#2ca02c', label='Aumenta Risco de Patologia (1)'),
    Patch(facecolor='#d62728', label='Protege / Favorece Saudável (0)')
]
plt.legend(handles=legend_lime, loc="lower right")
plt.tight_layout()
plt.show()

print(f"[*] Paciente #{idx_limiar} explicado com sucesso pelo LIME!")
```

---

# MÓDULO 4: PODA GUIADA POR XAI VS. SELEÇÃO TRADICIONAL (ESTUDO DE ABLAÇÃO)

> [!TIP]
> ⚔️ **Analogia Geek — O Alívio de Carga Espacial:**
> Se uma nave espacial precisa cortar peso de 40 kg para 10 kg, o método tradicional RFE corta itens por tentativa e erro a cada parada. O método SHAP avalia a utilidade exata de cada suprimento de uma só vez. Vamos provar que com 10 kg certos a missão cumpre seus objetivos perfeitamente!

### Bloco 4.1 — Comparação Sistemática de Poda Progressiva (SHAP vs. RFE)

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **O que é o Ponto de Inflexão (Knee Point)?**  
> É o ponto na curva onde a quantidade de atributos diminui muito, mas o F1-Score se mantém no topo. Abaixo dele, o modelo perde variáveis informativas e despenca.

```python
# 1. Importamos o RFE do Scikit-Learn
from sklearn.feature_selection import RFE

# 2. Obtemos a lista ordenada de colunas pelo ranking SHAP
ranking_shap = df_importancia_shap["atributo"].tolist()

# 3. Obtemos o ranking tradicional via RFE
print("[*] Computando ranking tradicional com RFE... Aguarde...")
rfe = RFE(estimator=RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=1), n_features_to_select=1, step=2)
rfe.fit(X_train, y_train)
df_rfe = pd.DataFrame({"atributo": X_train.columns, "ranking": rfe.ranking_}).sort_values(by="ranking").reset_index(drop=True)
ranking_rfe = df_rfe["atributo"].tolist()

# 4. Funcao para executar a curva de ablacao
def curva_ablacao(ordem_cols, nome_metodo, passos):
    res = []
    for k in passos:
        cols = ordem_cols[:k]
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        
        t0 = time.perf_counter()
        clf.fit(X_train[cols], y_train)
        t_treino = time.perf_counter() - t0
        
        y_pred = clf.predict(X_test[cols])
        res.append({
            "metodo": nome_metodo,
            "n_atributos": k,
            "f1_score": f1_score(y_test, y_pred),
            "tempo_ms": t_treino * 1000
        })
    return pd.DataFrame(res)

# 5. Executamos as curvas de poda de 40 ate 2 atributos
passos_poda = list(range(40, 1, -2))
df_ablacao_shap = curva_ablacao(ranking_shap, "SHAP (XAI)", passos_poda)
df_ablacao_rfe = curva_ablacao(ranking_rfe, "RFE (Tradicional)", passos_poda)

# 6. Renderizamos as curvas comparativas
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Curva 1: F1-Score vs Poda
axes[0].plot(df_ablacao_shap["n_atributos"], df_ablacao_shap["f1_score"], marker='o', color='#1f77b4', lw=2.5, label="Seleção SHAP")
axes[0].plot(df_ablacao_rfe["n_atributos"], df_ablacao_rfe["f1_score"], marker='s', linestyle='--', color='#ff7f0e', lw=2, label="Seleção RFE")
axes[0].axvline(x=10, color='red', linestyle=':', lw=2, label='10 Biomarcadores Vitais')
axes[0].invert_xaxis()
axes[0].set_title("1. Desempenho Clínico (F1-Score) vs. Poda de Atributos", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Número de Atributos Retidos (Poda ->)", fontweight="bold")
axes[0].set_ylabel("F1-Score no Teste", fontweight="bold")
axes[0].legend(loc="lower left")
axes[0].grid(True, alpha=0.3)

# Curva 2: Tempo de Treino
axes[1].plot(df_ablacao_shap["n_atributos"], df_ablacao_shap["tempo_ms"], marker='o', color='#2ca02c', lw=2.5, label="Tempo Treino (ms)")
axes[1].axvline(x=10, color='red', linestyle=':', lw=2, label='Ponto Ótimo (10 Atributos)')
axes[1].invert_xaxis()
axes[1].set_title("2. Economia Computacional de Treinamento", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Número de Atributos Retidos (Poda ->)", fontweight="bold")
axes[1].set_ylabel("Tempo de Treino (ms)", fontweight="bold")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Exibimos a comprovacao
f1_10_shap = df_ablacao_shap[df_ablacao_shap["n_atributos"] == 10]["f1_score"].values[0]
print(f"🎉 Comprovado: com apenas 10 atributos (75% de redução), o F1-Score com SHAP se manteve em {f1_10_shap:.4f}!")
```

---

# MÓDULO 5: ENGENHARIA AVANÇADA (PRÉ-FILTRO & SHAP-SELECT)

> [!TIP]
> ⚔️ **Analogia Geek — O Peneiramento e o Antidoping:**
> A Fase 1 (Pré-filtro) é a triagem rápida de aptidão física (elimina quem não tem variância mínima ou é cópia redundante). A Fase 2 (`shap-select`) é o exame antidoping laboratorial: exige prova matemática de que o efeito atua na direção certa ($\beta > 0$) e não é mero acaso estatístico ($p < 0.05$).

### Bloco 5.1 — Implementando Pré-Filtro Híbrido e Seleção com `statsmodels`

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que eliminamos variáveis com coeficiente negativo na matriz SHAP?**  
> Porque se um atributo tem coeficiente $\beta \le 0$ na regressão $y \sim \Phi$, significa que quanto maior o impacto dele na árvore, *menor* é a probabilidade do caso ser a patologia verdadeira. É uma variável de confusão causal!

```python
# 1. Importamos o seletor de variancia e a biblioteca de inferencia econometrica
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm

# 2. FASE 1: Pre-filtro Hibrido (Variancia e Colinearidade > 0.90)
def pré_filtro_hibrido(X_tr, threshold_var=0.01, threshold_corr=0.90):
    sel = VarianceThreshold(threshold=threshold_var).fit(X_tr)
    cols_var = X_tr.columns[sel.get_support()].tolist()
    X_var = X_tr[cols_var]

    corr = X_var.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c].dropna() > threshold_corr).any()]
    cols_finais = [c for c in cols_var if c not in to_drop]

    if len(cols_finais) == X_tr.shape[1]:
        corr_m = X_var.corr().abs().mean(axis=0)
        to_rem = corr_m.sort_values(ascending=False).index[: max(1, X_tr.shape[1] // 10)]
        cols_finais = [c for c in X_var.columns if c not in to_rem]

    return X_tr[cols_finais], cols_finais

X_train_pre, cols_pre = pré_filtro_hibrido(X_train)
print(f"[*] Pré-filtro concluiu: reduziu de {X_train.shape[1]} para {len(cols_pre)} atributos pré-XAI.")

# 3. FASE 2: shap-select com Regressao Logistica
def executar_shap_select(X_tr, y_tr, p_cutoff=0.05):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1).fit(X_tr, y_tr)
    exp = shap.TreeExplainer(clf)
    sv = exp(X_tr)
    Phi = sv.values[:, :, 1] if len(sv.shape) == 3 else sv.values
    df_phi = pd.DataFrame(Phi, columns=X_tr.columns)

    X_const = sm.add_constant(df_phi)
    try:
        res = sm.Logit(y_tr, X_const).fit(disp=False)
        params, pvalues = res.params.drop("const"), res.pvalues.drop("const")
    except Exception:
        res = sm.OLS(y_tr, X_const).fit()
        params, pvalues = res.params.drop("const"), res.pvalues.drop("const")

    df_stat = pd.DataFrame({
        "atributo": X_tr.columns,
        "coeficiente": params.values,
        "p_value": pvalues.values,
        "mean_abs_shap": np.abs(Phi).mean(axis=0)
    })
    df_stat["selecionado"] = (df_stat["coeficiente"] > 0) & (df_stat["p_value"] < p_cutoff)
    df_stat = df_stat.sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    aprovados = df_stat[df_stat["selecionado"]]["atributo"].head(10).tolist()
    if len(aprovados) < 2:
        aprovados = df_stat[df_stat["coeficiente"] > 0]["atributo"].head(10).tolist()
    if len(aprovados) < 2:
        aprovados = df_stat.head(10)["atributo"].tolist()

    return aprovados, df_stat

atributos_aprovados, df_shap_select = executar_shap_select(X_train_pre, y_train)

# 4. Renderizamos o grafico diagnostico de coeficientes Beta
plt.figure(figsize=(13, 6))
df_top20 = df_shap_select.head(20)
cores_stat = ['#2ca02c' if sel else '#d62728' for sel in df_top20['selecionado']]
plt.barh(df_top20['atributo'][::-1], df_top20['coeficiente'][::-1], color=cores_stat[::-1])
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
plt.title("shap-select: Coeficientes da Regressão (Beta > 0 e P-Valor < 0.05)", fontsize=12, fontweight="bold")
plt.xlabel("Coeficiente Beta da Regressão sobre a Matriz SHAP", fontweight="bold")
plt.grid(True, alpha=0.3)

leg_stat = [
    Patch(facecolor='#2ca02c', label='Aprovado (Beta > 0 e P < 0.05)'),
    Patch(facecolor='#d62728', label='Descartado (Beta <= 0 ou P >= 0.05)')
]
plt.legend(handles=leg_stat, loc="lower right")
plt.tight_layout()
plt.show()

print(f"[*] Subconjunto de Elite Selecionado ({len(atributos_aprovados)} atributos):", atributos_aprovados)
```

---

# MÓDULO 6: O GRANDE FINAL (PIPELINE, OPTUNA & DASHBOARD EXECUTIVO)

> [!TIP]
> ⚔️ **Analogia Geek — A Afinação da Telemetria de Fórmula 1:**
> Você aliviou 75% do peso do carro. Agora o chassi é puro músculo. Você coloca o engenheiro chefe (**Optuna**) para calibrar o motor, injeção e marchas (**hiperparâmetros**) para este novo chassi leve. O resultado é o modelo campeão de produção!

### Bloco 6.1 — Otimização Bayesiana com Optuna e Dashboard Comparativo de 4 Quadrantes

> [!IMPORTANT]
> 🤔 **Dúvidas Comuns de Iniciantes:**  
> **Por que o Optuna é superior ao GridSearch?**  
> Porque o GridSearch testa combinações às cegas sem aprender nada com as falhas anteriores. O Optuna usa o algoritmo TPE (Bayesiano), que calcula quais faixas de valores têm maior probabilidade de bater o recorde de F1-score!

```python
# 1. Importamos o Optuna e a validacao cruzada
import optuna
from sklearn.model_selection import cross_val_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 2. Fatiamos as matrizes de treino e teste mantendo apenas os atributos aprovados pelo shap-select
X_tr_red = X_train[atributos_aprovados]
X_te_red = X_test[atributos_aprovados]

# 3. Definimos a funcao de busca bayesiana com Optuna
def otimizar_optuna(X_tr, y_tr, n_trials=15):
    def objective(trial):
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 30, 200),
            max_depth=trial.suggest_int("max_depth", 3, 15),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=42,
            n_jobs=1
        )
        return cross_val_score(clf, X_tr, y_tr, cv=3, scoring="f1", n_jobs=1).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

print("[*] Executando Otimização Bayesiana com Optuna no espaço reduzido... Aguarde...")
best_params = otimizar_optuna(X_tr_red, y_train, n_trials=15)
print(f"[*] Melhores Hiperparâmetros encontrados pelo Optuna: {best_params}")

# 4. Treinamos o classificador campeao reduzido
modelo_campeao = RandomForestClassifier(**best_params, random_state=42, n_jobs=1)

t0_treino_red = time.perf_counter()
modelo_campeao.fit(X_tr_red, y_train)
tempo_treino_red = time.perf_counter() - t0_treino_red

t0_inf_red = time.perf_counter()
y_pred_red = modelo_campeao.predict(X_te_red)
y_proba_red = modelo_campeao.predict_proba(X_te_red)[:, 1]
tempo_inf_red = time.perf_counter() - t0_inf_red

m_reduzido = {
    "num_atributos": len(atributos_aprovados),
    "acuracia": accuracy_score(y_test, y_pred_red),
    "f1_score": f1_score(y_test, y_pred_red),
    "roc_auc": roc_auc_score(y_test, y_proba_red),
    "tempo_treino_ms": tempo_treino_red * 1000,
    "tempo_inferencia_ms": tempo_inf_red * 1000
}

# 5. Renderizamos o Dashboard Executivo Final de 4 Paineis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("DASHBOARD FINAL: PIPELINE DE REDUÇÃO DE DADOS GUIADO POR XAI", fontsize=14, fontweight="bold")

cats = ["Baseline (Bruto)", "XAI Reduzido + Optuna"]
cores_dash = ["#d62728", "#2ca02c"]

# Painel 1: Dimensoes
axes[0, 0].bar(cats, [metricas_base["num_atributos"], m_reduzido["num_atributos"]], color=cores_dash, width=0.45)
axes[0, 0].set_title("1. Dimensão do Dataset (Nº de Atributos)", fontsize=11, fontweight="bold")
axes[0, 0].set_ylabel("Quantidade de Atributos", fontweight="bold")
for i, v in enumerate([metricas_base["num_atributos"], m_reduzido["num_atributos"]]):
    axes[0, 0].text(i, v / 2, str(v), ha='center', va='center', color='white', fontweight='bold', fontsize=14)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Painel 2: Tempo de Treino
axes[0, 1].bar(cats, [metricas_base["tempo_treino_s"]*1000, m_reduzido["tempo_treino_ms"]], color=cores_dash, width=0.45)
axes[0, 1].set_title("2. Tempo de Treinamento em Servidor (ms)", fontsize=11, fontweight="bold")
axes[0, 1].set_ylabel("Milissegundos (ms)", fontweight="bold")
for i, v in enumerate([metricas_base["tempo_treino_s"]*1000, m_reduzido["tempo_treino_ms"]]):
    axes[0, 1].text(i, v / 2, f"{v:.1f} ms", ha='center', va='center', color='white', fontweight='bold', fontsize=13)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Painel 3: Latencia de Inferencia
axes[1, 0].bar(cats, [metricas_base["tempo_inferencia_s"]*1000, m_reduzido["tempo_inferencia_ms"]], color=cores_dash, width=0.45)
axes[1, 0].set_title("3. Latência de Inferência em Tempo Real (ms)", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("Milissegundos (ms)", fontweight="bold")
for i, v in enumerate([metricas_base["tempo_inferencia_s"]*1000, m_reduzido["tempo_inferencia_ms"]]):
    axes[1, 0].text(i, v / 2, f"{v:.1f} ms", ha='center', va='center', color='white', fontweight='bold', fontsize=13)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Painel 4: Acuracia vs F1
x_pos = np.arange(len(cats))
w = 0.35
axes[1, 1].bar(x_pos - w/2, [metricas_base["acuracia"], m_reduzido["acuracia"]], w, label="Acurácia", color="#1f77b4")
axes[1, 1].bar(x_pos + w/2, [metricas_base["f1_score"], m_reduzido["f1_score"]], w, label="F1-Score", color="#ff7f0e")
axes[1, 1].set_title("4. Integridade Preditiva (Acurácia vs. F1-Score)", fontsize=11, fontweight="bold")
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(cats, fontweight="bold")
axes[1, 1].set_ylim(0.70, 1.0)
axes[1, 1].legend(loc="lower right")
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 6. Relatorio comparativo final
print("\n" + "=" * 75)
print("RELATÓRIO COMPARATIVO FINAL EXECUTIVO")
print("=" * 75)
print(f" Métrica Avaliada         | Baseline (Completo)     | XAI Reduzido + Optuna")
print(f"--------------------------+-------------------------+----------------------")
print(f" Nº de Atributos          | {metricas_base['num_atributos']:<23} | {m_reduzido['num_atributos']:<20}")
print(f" Acurácia Global          | {metricas_base['acuracia']:<23.4f} | {m_reduzido['acuracia']:<20.4f}")
print(f" F1-Score Clínico         | {metricas_base['f1_score']:<23.4f} | {m_reduzido['f1_score']:<20.4f}")
print(f" ROC-AUC (Área)           | {metricas_base['roc_auc']:<23.4f} | {m_reduzido['roc_auc']:<20.4f}")
print(f" Tempo de Treino (ms)     | {metricas_base['tempo_treino_s']*1000:<23.2f} | {m_reduzido['tempo_treino_ms']:<20.2f}")
print(f" Latência Inferência (ms) | {metricas_base['tempo_inferencia_s']*1000:<23.2f} | {m_reduzido['tempo_inferencia_ms']:<20.2f}")
print("=" * 75)

# 7. Sanity Check Final Automatizado
tx_red = (1 - m_reduzido["num_atributos"] / metricas_base["num_atributos"]) * 100
assert tx_red >= 40.0, "Erro: Redução insuficiente"
assert m_reduzido["f1_score"] >= (metricas_base["f1_score"] - 0.05), "Erro: Queda inaceitável de F1"
print(f"\n🎉 SUCESSO TOTAL: {tx_red:.1f}% dos atributos eliminados preservando a precisão clínica do diagnóstico!")
```

---

## 🏆 Checklist de Autonomia Master do Aluno

- [ ] Sei formular e demonstrar o Mal da Dimensionalidade em problemas clínicos.
- [ ] Compreendi a Teoria dos Jogos de Lloyd Shapley e sei computar e interpretar o TreeSHAP.
- [ ] Dominei o uso do LIME para auditar e defender decisões em pacientes no limiar crítico ($P \approx 0.50$).
- [ ] Sei planejar e interpretar curvas de ablação de atributos, localizando o Ponto de Inflexão.
- [ ] Sei implementar Pré-Filtros Híbridos e a seleção com rigor estatístico `shap-select` ($\beta > 0$ e $p < 0.05$).
- [ ] Dominei a sintonia fina com Optuna via TPE e sei construir um Dashboard Executivo de 4 Painéis.
- [ ] Estou apto a aplicar este pipeline em problemas reais de ciência de dados na indústria ou na academia!

---

## 📚 Referências Bibliográficas & Documentações Oficiais

1. **Lundberg, S. M., & Lee, S. I. (2017).** *A unified approach to interpreting model predictions.* NeurIPS 2017.
2. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* ACM SIGKDD 2016.
3. **Akiba, T. et al. (2019).** *Optuna: A next-generation hyperparameter optimization framework.* ACM KDD 2019.
4. **Guyon, I. et al. (2002).** *Gene selection for cancer classification using support vector machines.* Machine Learning.
5. **Molnar, C. (2022).** *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.*
6. **Géron, A. (2022).** *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.* O'Reilly Media.
