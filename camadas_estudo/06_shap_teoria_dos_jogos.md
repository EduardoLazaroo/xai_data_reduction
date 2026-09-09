# Camada 06: SHAP — Explicabilidade Global e a Teoria dos Jogos Cooperativos

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 6  
**Contexto Técnico:** [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py) (`executar_etapa_shap`)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> Compreender os fundamentos matemáticos e a intuição por trás do **SHAP (SHapley Additive exPlanations)**. Explorar como a Teoria dos Jogos Cooperativos de Lloyd Shapley (Nobel 2012) resolve o problema da atribuição justa de crédito entre variáveis, entender a aceleração computacional do **TreeExplainer** e aprender a interpretar a **Importância Média $|SHAP|$** e o **Beeswarm Summary Plot**.

---

## 1. O que Estudar em Profundidade?

### 1.1 A Intuição dos Valores Shapley (Teoria dos Jogos)
Em **1953, o matemático Lloyd Shapley** fez uma pergunta seminal:
> *"Se um grupo de jogadores se une em uma coalizão para produzir um valor conjunto (um prêmio ou vitória), qual é a fatia justa que cabe a cada jogador?"*

Pense em um time de futebol que vence uma partida por 3 a 0. O atacante marcou os gols, o goleiro fez defesas difíceis e o zagueiro desarmou os adversários. Se dividirmos o bônus de vitória igualmente, seremos injustos com quem teve maior impacto.  
O **Valor Shapley ($\phi_i$)** calcula a **contribuição marginal** do jogador $i$ avaliando quanto a equipe ganharia em **todas as combinações possíveis** com e sem aquele jogador!

Em Machine Learning (Scott Lundberg, 2017):
- **Os Jogadores:** São as 40 colunas do paciente (biomarcadores, exames redundantes, ruídos).
- **O Jogo:** É o modelo preditivo (Random Forest).
- **A Pontuação:** É a predição contínua emitida (probabilidade de patologia).
- **O Valor Shapley ($\phi_i$):** É o "crédito" ou "débito" (positivo ou negativo) que o atributo $i$ adicionou para empurrar o paciente para longe da média da população!

```mermaid
graph LR
    A["🎲 Base Value (Média: 40%)"] --> B["🧪 Biomarcador 1 (+24%)"]
    B --> C["🧪 Biomarcador 2 (+16%)"]
    C --> D["📋 Exame Redundante (-3%)"]
    D --> E["🌪️ Ruído Metabólico (+0.01%)"]
    E --> F["🏁 Predição Final: 77.01% Patologia"]
```

---

### 1.2 Os 4 Axiomas que Tornam o SHAP Único e Confiável
O SHAP é o único método de explicabilidade que satisfaz matematicamente quatro axiomas de equidade:
1. **Eficiência:** A soma das contribuições de todos os atributos é exatamente igual à diferença entre a previsão final do paciente e a previsão média da base: $\sum \phi_i = f(x) - E[f(X)]$.
2. **Simetria:** Se dois atributos contribuem exatamente o mesmo valor marginal em todas as coalizões, seus valores Shapley são estritamente idênticos.
3. **Dummy (Variável Nula):** Se uma coluna não altera a previsão em nenhuma coalizão (como nossos ruídos metabólicos puros), seu valor Shapley é **rigorosamente zero** ($\phi_i = 0$).
4. **Aditividade:** Se somarmos dois modelos ensemble, os valores Shapley da soma são a soma dos valores Shapley individuais.

---

### 1.3 Por que o `TreeExplainer` é Tão Revolucionário?
Calcular os Valores Shapley exatos pela fórmula clássica exige avaliar todas as $2^M$ combinações de atributos. Em 40 colunas:
$$2^{40} \approx 1.099.511.627.776 \text{ combinações (mais de 1 trilhão!)}$$
Se usássemos o explicador genérico `KernelExplainer`, o computador levaria dias para rodar uma única aula!  
O algoritmo **TreeExplainer** (Lundberg et al., 2020) resolve isso aproveitando a topologia interna dos nós das árvores de decisão. Ele percorre os galhos da floresta de forma recursiva, reduzindo o tempo de cálculo para uma ordem polinomial $O(T \cdot L \cdot D^2)$, calculando os valores em **menos de 3 segundos**!

---

### 1.4 A Importância Média Global $|SHAP|$
Para saber quais atributos são os mais relevantes no cômputo global de todos os 2.000 pacientes, calculamos a **média do valor absoluto das contribuições**:

$$I_j = \frac{1}{N} \sum_{i=1}^N |\phi_{i,j}|$$

- Usamos o **módulo absoluto** ($|\cdot|$) porque um exame que empurra fortemente para "Saudável" ($\phi = -0.30$) é tão importante quanto um que empurra para "Doente" ($\phi = +0.30$). Se fizéssemos média simples, os valores positivos e negativos se cancelariam e pareceria que o atributo não tem importância!

---

## 2. Por que isso Importa para o Projeto?

O ranking por importância média $|SHAP|$ é a nossa **bússola científica de seleção**:
- É através dele que descobrimos a hierarquia das variáveis.
- É essa ordenação decrescente que alimenta o estudo de ablação da Camada 09 e a seleção econométrica da Camada 11.

---

## 3. Onde Aparece no Código do Projeto?

No arquivo [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py#L124):
- A função `executar_etapa_shap()` instancia `shap.TreeExplainer(modelo)`, extrai os valores da classe patológica (`shap_values.values[:, :, 1]`) e calcula `mean_abs_shap = np.abs(shap_vals_c1).mean(axis=0)`.
- As saídas visuais geradas são salvas em `assets/modulo2_shap_summary.png`.

---

## 4. Checkpoint de Autonomia do Estudante

Responda com precisão antes de seguir para a Camada 07:

> [!IMPORTANT]
> 🧠 **Pergunta do Checkpoint:**  
> **Você consegue explicar com suas próprias palavras por que o `biomarcador_1` aparece no topo do ranking SHAP com um valor médio $|SHAP| \approx 0.18$, enquanto o `ruido_metabolico_5` aparece no final com um valor próximo de $0.005$?**  
> *(Dica: Pense no ganho de informação que o biomarcador real produz nas divisões das árvores vs. o impacto irrelevante do ruído que quase nunca é escolhido pelas árvores).*
