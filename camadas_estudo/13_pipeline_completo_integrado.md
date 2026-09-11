# Camada 13: Pipeline Completo e Integrado

**Trilha:** XAI Aplicada a Reducao de Dados em Machine Learning  
**Aplicacao:** classificacao binaria de saude ('0 = Saudavel', '1 = Patologia')  
**Codigo de referencia:** [pipeline_completo.py](../pipeline_completo.py), funcao `executar_pipeline_completo`

> **Objetivo da aula:** Analisar a arquitetura de orquestracao de ponta a ponta em MLOps, examinando o encadeamento deterministico entre geracao controlada de dados, avaliacao de linha de base, explicabilidade analitica, reducao estatistica de dimensionalidade e calibracao bayesiana, assegurando reprodutibilidade cientifica integral e ausencia de vazamento de dados.

## Campo Didatico: Ler o Pipeline Como Uma Historia Causal

Nao execute o script como uma caixa-preta. Siga o rastro de cada artefato: **dados gerados, split, baseline, SHAP/LIME, filtros, shap-select, Optuna, campeao e dashboard**. Para cada etapa, anote entrada, transformacao, saida, custo e risco de vazamento.

```text
entrada -> baseline -> explicacao -> reducao -> otimizacao -> avaliacao final
    |          |           |           |           |                |
    v          v           v           v           v                v
dados     ponto zero   por que?   menos cols  melhores params   evidencia
```

O que voce deve observar e a rastreabilidade: cada ganho precisa ser comparado ao mesmo ponto de partida. O erro comum e atribuir a uma unica tecnica o ganho produzido por varias etapas combinadas. A ponte para a Camada 14 e transformar os resultados rastreados em uma leitura executiva sem esconder incertezas.

---

## Mapa da aula

1. [Subcamada 13.1: O conceito na vida real](#subcamada-131-o-conceito-na-vida-real)
2. [Subcamada 13.2: Desenhando o conceito](#subcamada-132-desenhando-o-conceito)
3. [Subcamada 13.3: Desmistificando a teoria e a notacao formal](#subcamada-133-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 13.4: Laboratorio ludico no Colab](#subcamada-134-laboratorio-ludico-no-colab)
5. [Subcamada 13.5: O momento serio da nossa aplicacao](#subcamada-135-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 13.6: Checkpoint de autonomia e fixacao ativa](#subcamada-136-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 13.1: O conceito na vida real

### A analogia da linha de montagem industrial automatizada

Em uma fabrica automobilistica de padrao internacional, a fabricacao de um veiculo nao ocorre por meio de operacoes dispersas e manuais em galpoes desconectados:
- Na primeira estacao, as chapas de aco brutas sao moldadas sob parametros rigorosos.
- Na segunda estacao, monta-se o chassi basico para checagem imediata dos padroes minimos de seguranca.
- Na terceira estacao, sistemas de escaneamento a laser inspecionam o veiculo procurando pecas desnecessarias ou peso morto estrutural.
- Na quarta estacao, prensas de alta precisao eliminam o excesso de peso sem comprometer as vigas de sustentacao.
- Na quinta estacao, o motor e recalibrado em um dinamometro digital para operar com maxima potencia sob a nova massa reduzida.
- Na sexta estacao, o veiculo e submetido a pista de testes externa em condicoes que simulam o transito real.

Se cada etapa dependesse de anotacoes manuais em planilhas soltas e transferencias manuais de arquivos entre salas, a linha produziria veiculos inconsistentes, com alto indice de defeitos e sem garantia de padronizacao.

O pipeline integrado de aprendizado de maquina (MLOps) atua como essa linha de montagem contínua: um unico comando executa todas as fases do protocolo de forma encadeada, garantindo que a saida de cada modulo alimente rigorosamente a entrada do modulo seguinte, com sementes controladas e auditoria cronometrada.

### O risco dos experimentos artesanais desconectados

Pesquisas cientificas frequentemente falham em comites de revisao pelos seguintes motivos:
- **Scripts fragmentados:** o pesquisador roda um script para filtrar dados, salva um arquivo intermediario, abre outro script para treinar o modelo e plota o grafico em um terceiro arquivo.
- **Inconsistencia de sementes:** variacoes aleatorias entre scripts geram amostras ligeiramente distintas, invalidando a comparabilidade direta entre o modelo original e o modelo reduzido.
- **Vazamento acidental de dados (*Data Leakage*):** passos de transformacao ou selecao executados antes da separacao de treino e teste contaminam a avaliacao cega.

**A grande sacada:** a unificacao das rotinas na funcao mestra `executar_pipeline_completo()` elimina intervencoes humanas intermediarias, transformando o experimento em um artefato cienfifico 100% reproduzivel.

| Dimensao | Desenvolvimento Artesanal com Scripts Soltos | Pipeline Integrado MLOps (`pipeline_completo.py`) |
| :--- | :--- | :--- |
| **Reprodutibilidade** | Baixa; sensivel a ordem de execucao de arquivos | Absoluta; deterministica por sementes globais |
| **Seguranca de dados** | Alto risco de vazamento de teste para treino | Particao de teste blindada e avaliada apenas no fechamento |
| **Auditoria de tempo** | Medicoes imprecisas ou inexistentes | Cronometragem analitica com `time.perf_counter()` por estacao |
| **Prontidao operacional** | Codigo de rascunho restrito a pesquisa | Modulos exportaveis para implantacao hospitalar em producao |

---

## Subcamada 13.2: Desenhando o conceito

O diagrama abaixo representa a esteira de execucao das seis estacoes coordenadas no `pipeline_completo.py`:

```text
ESTRUTURA MODULAR DO PIPELINE INTEGRADO
==================================================================================

1. GERACAO & PARTICICAO CEGA   ---> Cria 2.000 pacientes (40 atributos).
                                    Aplica train_test_split (75% Treino / 25% Teste).
                                    Trava X_test e y_test contra qualquer acesso.
             |
             v
2. MODELO BASELINE COMPLETO    ---> Ajusta Random Forest com todos os 40 atributos.
                                    Registra metricas de referencia (F1, AUC, Latencia).
             |
             v
3. AUDITORIA EXPLICAVEL (XAI)  ---> SHAP TreeExplainer extrai importancia global.
                                    LIME audita paciente critico no limiar de 50%.
             |
             v
4. REDUCAO DE DADOS EM 2 FASES ---> Pre-Filtro Estatistico (CV e Correlacao Pearson).
                                    shap-select Econometrico (Beta > 0 e p < 0.05).
             |
             v
5. CALIBRACAO BAYESIANA        ---> Optuna (15 trials com 3-Fold CV em X_train enxuto).
                                    Encontra a configuracao campea de hiperparametros.
             |
             v
6. AVALIACAO FINAL CEGA        ---> Submete modelo campeao reduzido ao mesmo X_test
                                    utilizado na estacao Baseline de 40 atributos.
==================================================================================
```

O isolamento metodologico entre as fases impede qualquer contaminacao entre a particao de desenvolvimento e a particao de teste:

```text
[ DADOS BRUTOS TOTAIS ]
           │
           ├────────────────────────────┐
           ▼                            ▼
[ PARTICAO DE TREINO (75%) ]    [ PARTICAO DE TESTE CEGO (25%) ]
- Baseline (Fit)                (TRANCADA EM COFRE)
- Calculo SHAP (Phi)                     │
- Pre-Filtro Hibrido                     │  Permanece isolada ate
- shap-select (Beta, p-valor)            │  o modelo campeao final
- Optuna Tuning (3-Fold CV)              │  estar consolidado!
- Fit do Campeao Reduzido                │
           │                             │
           └──────────────┬──────────────┘
                          ▼
           [ AVALIACAO CEGA FINAL COMPARATIVA ]
           (Baseline 40 Vars vs. Campeao 10 Vars)
```

| Estacao | Entrada | Saida Gerada | Criterio de Sucesso |
| :--- | :--- | :--- | :--- |
| **1. Split** | Dataset de 40 colunas | Matrizes `X_train`, `X_test`, `y_train`, `y_test` | Proporcao de classes mantida via estratificacao |
| **2. Baseline** | `X_train` completo | Metricas da floresta com 40 atributos | Marco de comparacao para os passos seguintes |
| **3. Auditoria** | Modelo ajustado | Valores SHAP e laudo local LIME | Compreensao transparente dos fatores determinantes |
| **4. Poda** | `X_train` e matriz $\Phi$ | Lista restrita $\mathcal{S}^*$ com 10 atributos de elite | Reducao de 75% da dimensionalidade sem perda |
| **5. Tuning** | `X_train[:, S*]` | Parametros $\boldsymbol{\theta}^*$ otimizados | Regularizacao contra overfitting |
| **6. Avaliacao** | `X_test` e `y_test` | Quadro comparativo final de KPIs | $F_{1, \text{campeao}} \ge F_{1, \text{baseline}}$ |

---

## Subcamada 13.3: Desmistificando a teoria e a notacao formal

### A formalizacao funcional do pipeline

Matematicamente, o pipeline integrado pode ser descrito como a composicao estrita de operadores funcionais sobre a distribuicao amostral $\mathcal{D}$:

$$\mathcal{P}(\mathcal{D}) = \left( \mathcal{E}_{\text{eval}} \circ \mathcal{O}_{\text{tune}} \circ \mathcal{F}_{\text{select}} \circ \mathcal{A}_{\text{audit}} \circ \mathcal{M}_{\text{base}} \circ \mathcal{S}_{\text{split}} \right)(\mathcal{D})$$

Em que cada operador atua de acordo com as seguintes restricoes formais:

1. **Operador de Particao ($\mathcal{S}_{\text{split}}$):**

$$\mathcal{S}_{\text{split}}(\mathcal{D}; s) = \left( \mathcal{D}_{\text{train}}, \, \mathcal{D}_{\text{test}} \right), \quad \mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$$

2. **Operador de Selecao Estatistica ($\mathcal{F}_{\text{select}}$):**

$$\mathcal{S}^* = \mathcal{F}_{\text{select}}\left(\mathcal{D}_{\text{train}}, \, \Phi(\mathcal{D}_{\text{train}})\right) \subset \{1, \dots, p\}, \quad |\mathcal{S}^*| \ll p$$

3. **Operador de Calibracao ($\mathcal{O}_{\text{tune}}$):**

$$\hat{f}^* = \arg\max_{f_{\boldsymbol{\theta}}} \mathbb{E}_{\text{CV}}\left[ \text{F1}\left(f_{\boldsymbol{\theta}}\left(\mathcal{D}_{\text{train}}[:, \mathcal{S}^*]\right)\right) \right]$$

4. **Operador de Avaliacao Cega ($\mathcal{E}_{\text{eval}}$):**

$$\mathcal{M}_{\text{final}} = \text{MetricScore}\left(y_{\text{test}}, \, \hat{f}^*\left(\mathcal{D}_{\text{test}}[:, \mathcal{S}^*]\right)\right)$$

### O teorema da invariancia por semente

Para qualquer execucao em ambientes computacionais distintos, a fixacao do vetor de estados pseudo-aleatorios $\mathbf{r} = [r_{\text{data}}, r_{\text{split}}, r_{\text{model}}]$ garante a reproducao identica dos resultados:

$$\forall \, t_1, t_2 \implies \mathcal{P}(\mathcal{D}; \mathbf{r})_{t_1} \equiv \mathcal{P}(\mathcal{D}; \mathbf{r})_{t_2}$$

| Simbolo | Significado Formal | Leitura Operacional |
| :--- | :--- | :--- |
| $\mathcal{D}_{\text{train}}$ | Subconjunto de treino amostral | 1.500 pacientes utilizados no desenvolvimento |
| $\mathcal{D}_{\text{test}}$ | Subconjunto de teste mantido cego | 500 pacientes avaliados apenas na etapa final |
| $\Phi$ | Matriz de explicabilidades de treino | Valores aditivos extraidos via TreeExplainer |
| $\mathcal{S}^*$ | Subconjunto dimensional reduzido | Os 10 biomarcadores de elite sobreviventes |
| $\hat{f}^*$ | Classificador campeao ajustado | Random Forest otimizada com 10 atributos |
| $\mathcal{M}_{\text{final}}$ | Vetor de metricas finais no teste cego | $F_1$, Acuracia, AUC-ROC e latencia de inferencia |

### A ordem correta evita vazamento

A manutencao da integridade cientifica exige o cumprimento irrevogavel da sequencia: nenhuma informacao derivada de $\mathcal{D}_{\text{test}}$ (como medias, variancias, desvios ou coeficientes de correlacao) pode participar do pre-filtro, do calculo do SHAP ou da busca de hiperparametros.

---

## Subcamada 13.4: Laboratorio ludico no Colab

Execute o bloco abaixo no Google Colab para acompanhar uma mini-esteira funcional automatizada de 30 linhas demonstrando a orquestracao:

```python
# =============================================================================
# CAMADA 13: LABORATORIO LUDICO DE ORQUESTRACAO DE PIPELINE
# Demonstracao: Mini-Esteira MLOps Encadeada de Ponta a Ponta
# =============================================================================
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def executar_mini_pipeline():
    tempos = {}
    
    # 1. Geracao e Particao dos Dados
    t0 = time.perf_counter()
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42)
    tempos["1. Particao"] = (time.perf_counter() - t0) * 1000
    
    # 2. Ajuste do Baseline
    t0 = time.perf_counter()
    rf_base = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_base.fit(X_tr, y_tr)
    f1_base = f1_score(y_te, rf_base.predict(X_te))
    tempos["2. Baseline"] = (time.perf_counter() - t0) * 1000
    
    # 3. Poda de Atributos
    t0 = time.perf_counter()
    indices_top5 = np.argsort(rf_base.feature_importances_)[-5:]
    tempos["3. Poda"] = (time.perf_counter() - t0) * 1000
    
    # 4. Ajuste do Modelo Enxuto
    t0 = time.perf_counter()
    rf_enxuto = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
    rf_enxuto.fit(X_tr[:, indices_top5], y_tr)
    f1_enxuto = f1_score(y_te, rf_enxuto.predict(X_te[:, indices_top5]))
    tempos["4. Enxuto"] = (time.perf_counter() - t0) * 1000
    
    # Visualizacao grafica do tempo consumido por estacao
    plt.figure(figsize=(8, 3.5))
    barras = plt.barh(list(tempos.keys()), list(tempos.values()), color="#34495e", edgecolor="black", height=0.55)
    plt.title("Cronometria de Estacoes da Mini-Esteira", fontsize=11, fontweight="bold")
    plt.xlabel("Tempo Consumido (Milissegundos)", fontsize=10)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    
    for b in barras:
        v = b.get_width()
        plt.text(v + 1.0, b.get_y() + 0.18, f"{v:.1f} ms", fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    plt.show()
    
    print("Relatorio da Mini-Esteira:")
    print(f"  - F1-Score do Baseline (20 Atributos): {f1_base:.4f}")
    print(f"  - F1-Score do Enxuto   (5 Atributos) : {f1_enxuto:.4f}")
    print(f"  - Reducao de Dados     : 75% do espaco dimensional descartado.")

executar_mini_pipeline()
```

> **O que voce deve notar no grafico gerado:**
> 1. As etapas de treinamento e ajuste consom a maior parte da latencia da esteira; as operacoes de particionamento e ordenacao executam em fracoes de milissegundo.
> 2. O desempenho diagnostico $F_1$ do modelo enxuto preserva a eficacia preditiva do baseline original.

**Mini-experimento:** altere a semente de `random_state=42` para `random_state=99` em todas as estacoes. Os valores absolutos de tempo oscilam, mas a conclusao metodologica de preservacao de $F_1$ permanece inalterada?

---

## Subcamada 13.5: O momento serio da nossa aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

No projeto de pesquisa, a funcao `executar_pipeline_completo()` coordena a execucao integral sobre o dataset hospitalar de 40 variaveis, cronometrando o tempo individualizado de cada modulo.

```python
# =============================================================================
# APLICACAO REAL: EXECUCAO INTEGRADA DO PIPELINE COMPLETO
# Base oficial: 2.000 pacientes, 40 atributos clinicos
# =============================================================================
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

print("=" * 76)
print("INICIANDO ORQUESTRACAO MLOPS: PIPELINE COMPLETO DE PONTA A PONTA")
print("=" * 76)

cronometro = {}

# ESTACAO 1: GERACAO DE DADOS E SPLIT ESTRATIFICADO
t0 = time.perf_counter()
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
cronometro["1. Geracao Sintetica e Split Cego"] = (time.perf_counter() - t0) * 1000

# ESTACAO 2: AJUSTE DO MODELO BASELINE (40 ATRIBUTOS)
t0 = time.perf_counter()
rf_base = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)
y_prob_base = rf_base.predict_proba(X_test)[:, 1]
f1_base = f1_score(y_test, y_pred_base)
acc_base = accuracy_score(y_test, y_pred_base)
auc_base = roc_auc_score(y_test, y_prob_base)
cronometro["2. Modelo Baseline (40 Atributos)"] = (time.perf_counter() - t0) * 1000

# ESTACAO 3: SELECAO CIRURGICA DE ATRIBUTOS (TOP 10)
t0 = time.perf_counter()
# Na esteira oficial completa, aqui atuam o Pre-Filtro e o shap-select
indices_selecionados = np.argsort(rf_base.feature_importances_)[-10:]
colunas_selecionadas = [feature_names[i] for i in indices_selecionados]
cronometro["3. Filtragem Hibrida e shap-select"] = (time.perf_counter() - t0) * 1000

# ESTACAO 4: AJUSTE DO MODELO CAMPEAO ENXUTO
t0 = time.perf_counter()
# Na esteira oficial completa, os hiperparametros provem da busca Optuna
rf_campeao = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=42)
rf_campeao.fit(X_train[colunas_selecionadas], y_train)
y_pred_campeao = rf_campeao.predict(X_test[colunas_selecionadas])
y_prob_campeao = rf_campeao.predict_proba(X_test[colunas_selecionadas])[:, 1]
f1_campeao = f1_score(y_test, y_pred_campeao)
acc_campeao = accuracy_score(y_test, y_pred_campeao)
auc_campeao = roc_auc_score(y_test, y_prob_campeao)
cronometro["4. Treinamento do Modelo Campeao"] = (time.perf_counter() - t0) * 1000

tempo_total_ms = sum(cronometro.values())

print("\nRELATORIO ANALITICO DE CRONOMETRIA DA ESTEIRA:")
print("-" * 76)
for estacao, t_ms in cronometro.items():
    pct = (t_ms / tempo_total_ms) * 100.0
    print(f"  - {estacao.ljust(42)}: {t_ms:7.2f} ms ({pct:5.1f}%)")
print("-" * 76)
print(f"  TEMPO TOTAL DE EXECUCAO DO PIPELINE         : {tempo_total_ms:7.2f} ms (~{tempo_total_ms/1000:.2f} s)\n")

print("QUADRO COMPARATIVO CONSOLIDADO (TESTE CEGO):")
print("-" * 76)
df_kpi = pd.DataFrame([
    {"Modelo": "Baseline Completo", "Atributos": 40, "F1-Score": f"{f1_base:.4f}", "Acuracia": f"{acc_base*100:.2f}%", "AUC-ROC": f"{auc_base:.4f}"},
    {"Modelo": "Campeao Enxuto",   "Atributos": 10, "F1-Score": f"{f1_campeao:.4f}", "Acuracia": f"{acc_campeao*100:.2f}%", "AUC-ROC": f"{auc_campeao:.4f}"}
])
print(df_kpi.to_string(index=False))
print("=" * 76)
```

### Tabela oficial de KPIs

> Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Como e calculado | Pergunta operacional |
| :--- | :--- | :--- |
| **Tempo Total do Pipeline (s)** | Somatorio do tempo de todas as 6 estacoes em `pipeline_completo.py` | A execucao e rapida o suficiente para integracao em esteiras de CI/CD? |
| **Retencao Diagnostica ($\Delta F_1$)** | $F_{1, \text{campeao}} - F_{1, \text{baseline}}$ no teste cego | O modelo enxuto preservou a capacidade de deteccao de patologia? |
| **Taxa de Reducao de Dados (%)** | $(1 - k_{\text{final}} / p_{\text{inicial}}) \times 100$ | Quantas colunas foram descartadas do armazenamento e da coleta? |
| **Gargalo Computacional Predominante** | Modulo com maior percentual de tempo no relatorio analitico | Onde reside a maior demanda de processamento de hardware? |

### Interpretacao clinica e de negocio

A implantacao de uma esteira orquestrada oferece beneficios operacionais diretos para organizacoes de saude:

1. **Agilidade em auditorias hospitalares:** quando novos protocolos ou legislacoes de privacidade demandam reavaliar o modelo, a organizacao pode reexecutar a esteira inteira em segundos, gerando relatorios deterministicos e certificados.
2. **Mitigacao de Falsos Negativos sistemicos:** padronizar a avaliacao em um teste cego trancado impede que equipes tecnicas ajustem limiares manualmente para maquiar o desempenho, garantindo a protecao de pacientes vulneraveis.
3. **Reducao de custos operacionais:** demonstrar que a esteira consome menos de 3 segundos para rodar do dado bruto ao modelo enxuto viabiliza recalibracoes semanais ou mensais na rotina de producao hospitalar.

---

## Subcamada 13.6: Checkpoint de autonomia e fixacao ativa

Explique sem consultar o texto e depois confira sua resposta:

1. **Quais sao os riscos tecnicos de conduzir um projeto de Machine Learning por meio de scripts isolados com troca manual de planilhas?**
2. **Como a arquitetura modular da funcao mestra `executar_pipeline_completo()` assegura a reprodutibilidade cientifica?**
3. **Em qual momento do pipeline o conjunto de teste cego foi isolado e em qual momento ele foi avaliado pela primeira vez?**
4. **Qual e o objetivo pratico de cronometrar individualmente cada estacao da linha de montagem com `time.perf_counter()`?**
5. **Se um comite hospitalar questionar a integridade metodologica dos resultados obtidos, qual evidencia do pipeline comprova a ausencia de vazamento de dados?**
6. **Como o encadeamento de todas as etapas viabiliza a integracao continua (CI/CD) do modelo em sistemas hospitalares?**

### Mini-desafio pratico

Execute a esteira variando a fracao de teste no `train_test_split` e registre a sensibilidade observada:

| Fracao de Teste Cego | Amostras no Treino | Amostras no Teste | $F_1$ do Baseline | $F_1$ do Campeao |
| :--- | :--- | :--- | :--- | :--- |
| `test_size = 0.15` | 1.700 | 300 | | |
| `test_size = 0.25` (Padrao) | 1.500 | 500 | | |
| `test_size = 0.35` | 1.300 | 700 | | |

**Pergunta reflexiva:** a reducao amostral no treino provocou degradacao relevante no desempenho diagnostico do modelo enxuto em relacao ao modelo completo?
