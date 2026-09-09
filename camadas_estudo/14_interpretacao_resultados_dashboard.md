# Camada 14: Interpretação dos Resultados Numéricos e o Dashboard Executivo

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 14  
**Contexto Técnico:** [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py) (`gerar_dashboard_executivo` e tabela de saída)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> Aprender a ler, analisar criticamente e defender os números finais gerados pelo experimento. Interpretar as 4 dimensões do **Dashboard Executivo Comparativo** (Nº de Atributos, Tempo de Treinamento, Latência de Inferência e Desempenho Preditivo), e compreender a discussão sutil sobre o **custo de explicabilidade durante o desenvolvimento vs. o ganho de eficiência em produção**.

---

## 1. O que Estudar em Profundidade?

### 1.1 A Tabela Comparativa de Resultados
Ao final da execução do [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py), o terminal imprime uma tabela comparativa com o seguinte perfil empírico típico:

| Métrica / Dimensão Avaliada | Baseline (Bruto / 40 Atributos) | XAI Reduzido + Optuna (~10 Atributos) | Impacto Observado |
| :--- | :---: | :---: | :---: |
| **Quantidade de Atributos ($M$)** | **40** | **10** (ou menos) | **75.0% de redução na coleta de dados!** |
| **Acurácia Global** | $\approx 0.8350$ | $\approx 0.8520$ | Mantida / Leve ganho (+1.7%) |
| **$F_1$-Score Clínico (Harmônico)** | $\approx 0.8410$ | $\approx 0.8600$ | **Ganho de generalização (+1.9%)** |
| **ROC-AUC (Área Separação)** | $\approx 0.9150$ | $\approx 0.9280$ | Preservação total da separabilidade |
| **Tempo de Treinamento em Servidor** | $\approx 280\text{ ms}$ | $\approx 110\text{ ms}$ | **Aceleração de mais de 60%!** |
| **Latência de Inferência (Tempo Real)** | $\approx 22\text{ ms}$ | $\approx 9\text{ ms}$ | **Respostas mais que 2x mais rápidas!** |

---

### 1.2 A Leitura dos 4 Quadrantes do Dashboard Executivo

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ 1. Dimensão do Dataset (Colunas)     │ 2. Tempo de Treinamento (Servidor)   │
│    [■ Baseline: 40]                  │    [■ Baseline: 280 ms]              │
│    [■ XAI Reduzido: 10] (-75%)       │    [■ XAI Reduzido: 110 ms] (-60%)   │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ 3. Latência de Inferência (Produção) │ 4. Desempenho Clínico (Acurácia/F1)  │
│    [■ Baseline: 22 ms]               │    [■ Baseline F1: 0.84]             │
│    [■ XAI Reduzido: 9 ms] (-59%)     │    [■ XAI Reduzido F1: 0.86] (+0.02) │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

1. **Quadrante 1 (Dimensão):** Comprova a economia de custos. Em um sistema de saúde, pedir 10 exames em vez de 40 reduz filas em laboratórios, custos com reagentes e exposição desnecessária do paciente a radiação ou punções.
2. **Quadrante 2 (Treinamento em Servidor):** Comprova que quando o modelo precisar ser retreinado semanalmente com novos prontuários na nuvem (AWS/Azure/GCP), o consumo de energia e faturamento de GPU/CPU cairá em mais de 60%.
3. **Quadrante 3 (Latência de Inferência):** Comprova a viabilidade para **dispositivos médicos de borda (*Edge AI*)**. Um aparelho de monitoramento portátil de UTI precisa emitir alertas em milissegundos; cortar pela metade o tempo de inferência salva vidas!
4. **Quadrante 4 (Integridade Diagnóstica):** É a prova cabal de que a economia de tempo e dinheiro **não foi conquistada às custas da saúde dos pacientes**: o $F_1$-score não apenas não caiu, como subiu ligeiramente graças à eliminação do ruído que antes causava overfitting!

---

### 1.3 Um Ponto Crítico de Discussão Acadêmica: O Custo de Seleção vs. O Ganho em Produção
Em uma banca de mestrado ou revisão de artigo científico, um avaliador experiente pode fazer a seguinte pergunta provocativa:
> *"Mas calcular o SHAP e rodar 15 tentativas do Optuna não gastou tempo extra de computador durante o experimento?"*

**Sua Resposta Técnica Perfeita:**
> *"Sim! A etapa de descoberta (cálculo de SHAP e Optuna) tem um custo computacional pontual durante a fase de P&D (Pesquisa e Desenvolvimento). No entanto, esse custo é pago **uma única vez** no laboratório! Uma vez encontrado o subconjunto de elite de 10 atributos e os hiperparâmetros campeões, o modelo final entrará em produção em hospitais atendendo **milhões de pacientes por anos**, acumulando uma economia exponencial de milhões de milissegundos e milhões de exames ao longo do seu ciclo de vida!"*

---

## 2. Por que isso Importa para o Projeto?

Esta interpretação numérica é exatamente a matéria-prima que preencherá as seções de **Resultados** e **Discussão** do seu artigo acadêmico (Camadas 15 e 16).

---

## 3. Onde Aparece no Código do Projeto?

No arquivo [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py#L358):
- A função `gerar_dashboard_executivo()` desenha os 4 gráficos em `assets/dashboard_final_comparativo.png` e imprime a tabela final estruturada no console.

---

## 4. Checkpoint de Autonomia do Estudante

Responda antes de seguir para a Camada 15:

> [!IMPORTANT]
> 🧠 **Pergunta do Checkpoint:**  
> **Você consegue escrever, com suas próprias palavras em 3 frases curtas e objetivas, um parágrafo de síntese executiva resumindo exatamente o que os números do dashboard final comprovam?**  
> *(Dica: Frase 1: Redução percentual de dados; Frase 2: Comportamento do F1-score; Frase 3: Ganho de velocidade).*
