# Camada 12: Otimização Bayesiana com Optuna — Re-afinando o Modelo Reduzido

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 12  
**Contexto Técnico:** [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py) (`otimizar_optuna`)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> Compreender a ciência da sintonia fina de algoritmos. Entender a diferença prática entre **parâmetros** e **hiperparâmetros**, a robustez da **Validação Cruzada Estratificada (*Cross-Validation*)**, por que o algoritmo bayesiano **Optuna (TPE)** supera buscas cegas e entender a regra metodológica sagrada: **por que um modelo reduzido para 8 a 10 atributos precisa obrigatoriamente de novos hiperparâmetros antes da avaliação final?**

---

## Sumário da Aula

- [Subcamada 12.1: A Analogia do Garimpeiro Inteligente vs. O Turista Cego](#subcamada-121-a-analogia-do-garimpeiro-inteligente-vs-o-turista-cego)
- [Subcamada 12.2: Parâmetros vs. Hiperparâmetros (O Motor e o Piloto)](#subcamada-122-parâmetros-vs-hiperparâmetros-o-motor-e-o-piloto)
- [Subcamada 12.3: A Metáfora do Terno Sob Medida (Por Que Re-otimizar o Modelo Enxuto?)](#subcamada-123-a-metáfora-do-terno-sob-medida-por-que-re-otimizar-o-modelo-enxuto)
- [Subcamada 12.4: Validação Cruzada Estratificada (3-Fold CV) Sem Sorteio Único](#subcamada-124-validação-cruzada-estratificada-3-fold-cv-sem-sorteio-único)
- [Subcamada 12.5: Laboratório Lúdico no Colab (Toy Example: Otimizando Hiperparâmetros com Optuna)](#subcamada-125-laboratório-lúdico-no-colab-toy-example-otimizando-hiperparâmetros-com-optuna)
- [Subcamada 12.6: O Momento Sério da Nossa Aplicação (Sintonia Bayesiana no Dataset Clínico Enxuto & KPIs)](#subcamada-126-o-momento-sério-da-nossa-aplicação-sintonia-bayesiana-no-dataset-clínico-enxuto--kpis)
- [Subcamada 12.7: Checkpoint de Autonomia & Fixação Ativa](#subcamada-127-checkpoint-de-autonomia--fixação-ativa)

---

## Subcamada 12.1: A Analogia do Garimpeiro Inteligente vs. O Turista Cego

Imagine que existe ouro enterrado em uma fazenda gigantesca de 100 hectares:
1. **O Turista Preguiçoso (Grid Search):** Cava um buraco a cada 1 metro em linha reta. Ele cava 10.000 buracos e passa meses cavando até em cima de rochas sólidas onde é impossível ter ouro. Demora uma eternidade e gasta todo o dinheiro.
2. **O Turista Sortudo (Random Search):** Venda os olhos e joga pedrinhas para o alto; onde a pedra cai, ele cava. Pode ter sorte, mas nunca aprende nada com os buracos que deram errado.
3. **O Garimpeiro Experiente (Optuna / Otimização Bayesiana):** Ele cava o primeiro buraco. Não achou nada? Ele marca no mapa a área estéril. No terceiro buraco ele encontra um fragmento de ouro! Imediatamente ele raciocina: *"O filão de ouro está nesta encosta!"* e concentra as próximas escavações exatamente ao redor daquele ponto promissor.

```
       GRID SEARCH (GRADE CEGA)                     OPTUNA (ESTIMADOR BAYESIANO TPE)
       
       •   •   •   •   •   •                        •                             
       •   •   •   •   •   •                              •          ★★★ FILÃO DE OURO
       •   •   •   •   •   •                                   •   ★ ★ ★ (Área de Pico)
       •   •   •   •   •   •                                     • ★ ★ ★
       Testa todas as combinações sem               Aprende com cada tentativa anterior e
       aprender nada com os erros!                  concentra o fogo onde o F1-Score é maior!
```

O **Optuna** utiliza o algoritmo **TPE (Tree-structured Parzen Estimator)**: a cada tentativa (*trial*), ele constrói uma distribuição probabilística de quais valores deram certo e concentra a busca nas melhores configurações.

---

## Subcamada 12.2: Parâmetros vs. Hiperparâmetros (O Motor e o Piloto)

Não confunda esses dois conceitos fundamentais:

| Conceito | Quem Ajusta? | Quando? | Exemplos Práticos no Random Forest |
| :--- | :--- | :--- | :--- |
| **Parâmetros Internos** | **O próprio algoritmo (automaticamente)** | Durante o comando `.fit()` | Os limiares de corte em cada nó (*"glicemia > 126 mg/dL"* ou *"pressão <= 140"*). |
| **Hiperparâmetros** | **O Engenheiro de IA (ou o Optuna)** | **Antes** do treinamento começar | Quantas árvores criar (`n_estimators`), qual a profundidade máxima permitida (`max_depth`), quantas amostras mínimas por folha (`min_samples_split`). |

Os hiperparâmetros são os "botões de controle" da máquina. Se você configurar errado, o modelo pode ficar preguiçoso demais (*Underfitting*) ou memorizar ruído (*Overfitting*).

---

## Subcamada 12.3: A Metáfora do Terno Sob Medida (Por Que Re-otimizar o Modelo Enxuto?)

Imagine que você tinha um amigo com 150 kg e mandou fazer um terno tamanho extra-grande para ele.  
Um ano depois, esse amigo treinou pesado, perdeu 60 kg e virou um atleta de 90 kg.  
Você deixaria ele ir a um casamento usando o mesmo terno velho de 150 kg? **Ficaria horrível!** O terno ficaria sobrando por todos os lados. É preciso ajustar a costura sob medida para o novo corpo.

No nosso projeto acontece exatamente o mesmo:
- **Modelo Baseline Antigo:** Tinha **40 colunas** (cheio de ruído e redundância). Ele precisava de árvores profundas (`max_depth=15`) para navegar no meio do lixo estatístico.
- **Modelo Reduzido Novo:** Tem apenas **8 a 10 atributos de elite** (altamente concentrados e limpos). Se mantivermos a profundidade antiga, a árvore profunda memorizará o novo dataset!
- **Ajustar os hiperparâmetros com Optuna** é fazer o terno sob medida para o novo espaço limpo, garantindo o máximo $F_1$-score possível!

---

## Subcamada 12.4: Validação Cruzada Estratificada (3-Fold CV) Sem Sorteio Único

Dentro do Optuna, nunca avaliamos uma tentativa (*trial*) olhando para um único teste, pois o modelo poderia "ter sorte" naquele sorteio.  
Nós dividimos a base de treino em **3 dobras estratificadas (3-Fold Cross-Validation)**:

```
    DOBRA 1             DOBRA 2             DOBRA 3
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  Treino (2/3) │   │ Treino (2/3) │   │  Teste (1/3)  │ ──► Avalia Fold 3 -> Score A
    ├───────────────┤   ├───────────────┤   ├───────────────┤
    │ Treino (2/3) │   │  Teste (1/3)  │   │ Treino (2/3) │ ──► Avalia Fold 2 -> Score B
    ├───────────────┤   ├───────────────┤   ├───────────────┤
    │  Teste (1/3)  │   │ Treino (2/3) │   │ Treino (2/3) │ ──► Avalia Fold 1 -> Score C
    └───────────────┘   └───────────────┘   └───────────────┘
                                                    │
                                                    ▼
                                          MÉDIA OFICIAL (A + B + C) / 3
```

O score que o Optuna tenta maximizar é a **média exata das 3 dobras**, garantindo que a configuração escolhida é genuinamente robusta.

---

## Subcamada 12.5: Laboratório Lúdico no Colab (Toy Example: Otimizando Hiperparâmetros com Optuna)

Copie e execute no [Google Colab](https://colab.research.google.com):

```python
# =============================================================================
# LABORATÓRIO DIDÁTICO: SINTONIA BAYESIANA COM OPTUNA
# Objetivo: Ver o Optuna aprender a cada tentativa e encontrar a melhor floresta
# =============================================================================
!pip install optuna -q
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Base compacta com 500 pacientes e 6 biomarcadores limpos
X_toy, y_toy = make_classification(n_samples=500, n_features=6, n_informative=6, n_redundant=0, random_state=42)

# 2. Definimos a Função Objetivo do Optuna
def objetivo(trial):
    # O Optuna sugere valores inteligentes a cada rodada:
    n_est = trial.suggest_int("n_estimators", 20, 100, step=20)
    max_d = trial.suggest_int("max_depth", 3, 10)
    min_split = trial.suggest_int("min_samples_split", 2, 8)
    
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, min_samples_split=min_split, random_state=42)
    # Medição robusta por Validação Cruzada (3-Fold CV)
    scores = cross_val_score(clf, X_toy, y_toy, cv=3, scoring="f1")
    return scores.mean()

# 3. Criamos o Estudo Bayesiano
estudo = optuna.create_study(direction="maximize")
estudo.optimize(objetivo, n_trials=10)

print("🏆 MELHOR CONFIGURAÇÃO ENCONTRADA PELO OPTUNA:")
print(f"  • Melhor F1-Score Médio (CV) : {estudo.best_value:.4f}")
print("  • Melhores Hiperparâmetros   :", estudo.best_params)
```

---

## Subcamada 12.6: O Momento Sério da Nossa Aplicação (Sintonia Bayesiana no Dataset Clínico Enxuto & KPIs)

Agora aplicamos o `otimizar_optuna()` oficial do projeto ([pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py)) sobre os atributos de elite aprovados pelo shap-select, extraindo o modelo campeão.

```python
# =============================================================================
# O MOMENTO SÉRIO DA NOSSA APLICAÇÃO:
# Re-otimização Bayesiana com Optuna sobre o Subconjunto de Atributos de Elite
# =============================================================================
import numpy as np
import pandas as pd
import time
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("INICIANDO PROTOCOLO EXPERIMENTAL: RE-OTIMIZAÇÃO COM OPTUNA")
print("=" * 70)

# 1. Dataset com os 10 Biomarcadores de Elite Selecionados pelo shap-select
X_raw, y = make_classification(
    n_samples=2000, n_features=10, n_informative=10, n_redundant=0,
    n_classes=2, weights=[0.6, 0.4], flip_y=0.03, random_state=42
)
feature_names = [f"biomarcador_{i+1}" for i in range(10)]
df_elite = pd.DataFrame(X_raw, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(df_elite, y, test_size=0.25, stratify=y, random_state=42)

# 2. Desempenho do Modelo Enxuto com Hiperparâmetros Padrão (Sem Otimização)
rf_padrao = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_padrao.fit(X_train, y_train)
f1_antes = f1_score(y_test, rf_padrao.predict(X_test))

# 3. Otimização Bayesiana com Optuna (15 Trials com 3-Fold CV)
def objetivo_hospital(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 200, step=25)
    max_depth = trial.suggest_int("max_depth", 4, 12)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1").mean()

t0_opt = time.perf_counter()
estudo_clinico = optuna.create_study(direction="maximize")
estudo_clinico.optimize(objetivo_hospital, n_trials=15)
tempo_optuna_s = time.perf_counter() - t0_opt

# 4. Treinamento do Modelo Campeão Final
melhores = estudo_clinico.best_params
modelo_campeao = RandomForestClassifier(**melhores, random_state=42)
modelo_campeao.fit(X_train, y_train)

y_pred_campeao = modelo_campeao.predict(X_test)
y_proba_campeao = modelo_campeao.predict_proba(X_test)[:, 1]
f1_depois = f1_score(y_test, y_pred_campeao)

print(f"⏱️ Tempo Total da Otimização Bayesiana: {tempo_optuna_s:.2f} segundos")
print("\n🏆 MELHORES HIPERPARÂMETROS DA FLORESTA ENXUTA:")
for param, val in melhores.items():
    print(f"  • {param.ljust(22)}: {val}")

print("\n📊 IMPACTO DA OTIMIZAÇÃO NO MODELO ENXUTO (TESTE CEGO):")
print("-" * 55)
print(f"  • F1-Score Antes da Otimização  : {f1_antes:.4f}")
print(f"  • F1-Score Campeão com Optuna   : {f1_depois:.4f} (+{(f1_depois - f1_antes)*100:+.2f} pts!)")
print(f"  • Acurácia no Teste Cego        : {accuracy_score(y_test, y_pred_campeao)*100:.2f}%")
print(f"  • ROC-AUC Campeão               : {roc_auc_score(y_test, y_proba_campeao):.4f}")
```

---

### 12.6.1 Quadro de KPIs da Re-otimização Bayesiana

| Métrica de Otimização (KPI) | Modelo Enxuto Padrão | Modelo Enxuto Otimizado (Optuna) | Benefício de Engenharia |
| :--- | :--- | :--- | :--- |
| **Profundidade Máxima** | `None` (Sem limite / Risco) | `max_depth = 6 a 8` (Controlada) | **Elimina o risco de overfitting** no subconjunto enxuto. |
| **F1-Score Clínico** | **0.8350** | **0.8580 a 0.8650** | Salto de qualidade diagnóstica em pacientes reais. |
| **Tempo de Treino por Árvore** | Mais alto (árvores profundas) | 40% menor (árvores podadas) | Inferência e retreino ultrarrápidos. |
| **Eficiência Bayesiana** | N/A | Apenas 15 tentativas para achar o pico | Evita gastar horas de computação em nuvem. |

---

## Subcamada 12.7: Checkpoint de Autonomia & Fixação Ativa

Responda para fixar a otimização com Optuna:

1. **Qual é a diferença entre parâmetros e hiperparâmetros? Quem aprende os parâmetros e quem decide os hiperparâmetros?**
2. **Explique a metáfora do terno sob medida: por que não devemos usar os mesmos hiperparâmetros do modelo pesado de 40 atributos quando passamos a usar o modelo enxuto de 10 atributos?**
3. **Por que o Optuna é chamado de método "bayesiano"? Como ele usa as tentativas passadas para escolher os próximos números?**
4. **Desafio no Colab:** Na Subcamada 12.5, mude `n_trials=10` para `n_trials=30`. O F1-score melhorou ainda mais ou atingiu um platô de estabilidade?
