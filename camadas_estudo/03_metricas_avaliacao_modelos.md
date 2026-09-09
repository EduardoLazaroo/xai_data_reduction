# Camada 03: Métricas de Avaliação — O Rigor Diagnóstico Além da Acurácia

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 3  
**Contexto Técnico:** [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py) (`treinar_e_avaliar_modelo` e `plotar_graficos_baseline`)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> Dominar o arsenal matemático de métricas de avaliação para problemas de classificação binária médica. Entender por que a **Acurácia é uma armadilha perigosa** em dados desbalanceados, a diferença crucial entre **Precisão** e **Recall**, por que o **$F_1$-Score** utiliza uma média harmônica e como interpretar intuitivamente a área sob a curva **ROC-AUC**.

---

## 1. O que Estudar em Profundidade?

### 1.1 A Matriz de Confusão: A Tabela da Verdade Diagnóstica
Toda métrica de classificação binária nasce da contagem cruzada entre o que o paciente realmente tem e o que o modelo previu:

| | Modelo Previu Negativo ($0$) | Modelo Previu Positivo ($1$) |
| :--- | :---: | :---: |
| **Real Negativo ($0$ - Saudável)** | **Verdadeiro Negativo (VN)** *(Acerto)* | **Falso Positivo (FP)** *(Alarme Falso)* |
| **Real Positivo ($1$ - Patologia)** | **Falso Negativo (FN)** *(Erro Crítico!)* | **Verdadeiro Positivo (VP)** *(Acerto)* |

- **Verdadeiro Positivo (VP):** O paciente tem a doença e o modelo detectou corretamente.
- **Verdadeiro Negativo (VN):** O paciente é saudável e o modelo confirmou que está tudo bem.
- **Falso Positivo (FP) — Erro Tipo I:** O paciente é saudável, mas o modelo disse que ele está doente. (Consequência: ansiedade, repetição de exames e custos adicionais).
- **Falso Negativo (FN) — Erro Tipo II:** O paciente está com a patologia, mas o modelo disse que ele é saudável! (**Consequência Gravíssima:** O paciente não recebe tratamento e pode ir a óbito).

---

### 1.2 Por que a Acurácia Engana? (O Paradoxo da Acurácia)
A fórmula da acurácia mede simplesmente a proporção total de acertos:
$$\text{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN}$$

> [!CAUTION]
> 🚨 **O Exemplo Clínico que Assusta:**  
> Imagine um hospital rastreando uma doença rara que atinge **1 em cada 100 pacientes** (1% de prevalência; 99% saudáveis).  
> Um programador preguiçoso cria um modelo com uma única linha de código: `return 0` (chutar sempre "Saudável" para todo mundo).  
> - Esse modelo terá **99% de acurácia**!  
> - Porém, ele tem **zero de utilidade médica**: dos 10 pacientes doentes que passaram pelo hospital, o modelo não detectou nenhum e todos os 10 ficaram sem tratamento!  
> Por isso, em saúde e ciência de dados profissional, **nunca confiamos apenas na Acurácia**.

---

### 1.3 O Duelo: Precisão vs. Recall (Sensibilidade)

1. **Precisão (*Precision*):** *"De todos os pacientes que o modelo disse que têm a doença, quantos realmente tinham?"*
   $$\text{Precisão} = \frac{VP}{VP + FP}$$
   - Importante quando o custo de um falso alarme é alto (ex: uma cirurgia cerebral invasiva de alto risco).

2. **Recall ou Sensibilidade (*True Positive Rate*):** *"De todos os pacientes que realmente estavam doentes no hospital, quantos o modelo conseguiu capturar?"*
   $$\text{Recall} = \frac{VP}{VP + FN}$$
   - Importante quando não podemos deixar escapar nenhum doente (ex: triagem de COVID ou detecção de câncer em estágio inicial).

---

### 1.4 Por que o F1-Score Usa a Média Harmônica?
O **$F_1$-Score** é o indicador de ouro do nosso projeto porque ele equilibra a Precisão e o Recall em um único número:
$$F_1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}$$

*Por que não fazer uma média aritmética simples $\frac{\text{Precisão} + \text{Recall}}{2}$?*  
Porque a média aritmética permite compensações desonestas! Se um modelo tem **Precisão = 100%** e **Recall = 0%**, a média aritmética daria **50%** (parecendo mediano). A **média harmônica**, por sua natureza matemática, despenca para **0%** se qualquer um dos dois lados for nulo, forçando o algoritmo a ser simultaneamente bom em ambos!

---

### 1.5 A Curva ROC e o ROC-AUC
Um classificador não produz apenas 0 e 1 rígidos; ele calcula uma **probabilidade contínua** (ex: 82% de chance de patologia).  
- Por padrão, usamos o limiar de corte (*threshold*) em $0.50$. Mas e se mudássemos o corte para $0.30$ ou $0.70$?
- A **Curva ROC (Receiver Operating Characteristic)** desenha no gráfico a Taxa de Verdadeiros Positivos (TPR) contra a Taxa de Falsos Positivos (FPR) para **todos os limiares de corte possíveis de 0.0 a 1.0**!
- O **ROC-AUC (Área sob a Curva ROC)** resume quão bem o modelo separa as classes:
  - $\text{AUC} = 0.50$: Equivale a jogar uma moeda para o alto (chute aleatório).
  - $\text{AUC} = 1.00$: Separação diagnóstica perfeita sem nenhuma sobreposição de classes.

---

## 2. Por que isso Importa para o Projeto?

Nosso projeto inteiro é uma pesquisa comparativa:
- Se nós reduzirmos o dataset de 40 para 10 atributos e o $F_1$-score cair de $0.85$ para $0.60$, nosso método falhou.
- Se o $F_1$-score se mantiver em $0.85$ (ou subir para $0.86$), provamos cientificamente que **os 30 atributos cortados eram lixo computacional**!

---

## 3. Onde Aparece no Código do Projeto?

No arquivo [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py):
- Linhas 69 a 85: A função `treinar_e_avaliar_modelo()` extrai acurácia, F1-score, precisão, recall e ROC-AUC usando as funções oficiais do `sklearn.metrics`.
- Linhas 88 a 118: A função `plotar_graficos_baseline()` renderiza a matriz de confusão e calcula `roc_curve(y_test, y_proba)`.
- Linha 341: O teste de sanidade final valida: `assert m_reduzido["f1_score"] >= (m_baseline["f1_score"] - 0.05)`.

---

## 4. Checkpoint de Autonomia do Estudante

Responda antes de seguir para a Camada 04:

> [!IMPORTANT]
> 🧠 **Pergunta do Checkpoint:**  
> **Se o $F_1$-score do modelo reduzido (com apenas 10 atributos) resultou em $0.8640$ e o modelo baseline original (com todos os 40 atributos) tinha $0.8350$, o que isso significa na prática médica e computacional?**  
> *(Dica: Pense na remoção do ruído que antes confundia o modelo e no ganho de capacidade de generalização).*
