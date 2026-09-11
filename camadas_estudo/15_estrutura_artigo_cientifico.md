# Camada 15: Estrutura de Artigo Cientifico e Formulacao de Hipoteses

**Trilha:** XAI Aplicada a Reducao de Dados em Machine Learning  
**Aplicacao:** classificacao binaria de saude ('0 = Saudavel', '1 = Patologia')  
**Codigo de referencia:** [gerar_artigo_word.py](../gerar_artigo_word.py) e [docs/artigo_xai_reduction.docx](../docs/artigo_xai_reduction.docx)

> **Objetivo da aula:** Compreender a arquitetura de comunicacao cientifica baseada no formato canonico IMRaD (Introducao, Metodologia, Resultados e Discussao), estabelecendo a demarcacao metodologica estrita entre o relato objetivo de resultados e a interpretacao analitica da discussao, alem de formalizar o teste quantitativo das hipoteses de equivalencia diagnostica (H1) e eficiencia computacional (H2).

---

## Mapa da aula

1. [Subcamada 15.1: O conceito na vida real](#subcamada-151-o-conceito-na-vida-real)
2. [Subcamada 15.2: Desenhando o conceito](#subcamada-152-desenhando-o-conceito)
3. [Subcamada 15.3: Desmistificando a teoria e a notacao formal](#subcamada-153-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 15.4: Laboratorio ludico no Colab](#subcamada-154-laboratorio-ludico-no-colab)
5. [Subcamada 15.5: O momento serio da nossa aplicacao](#subcamada-155-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 15.6: Checkpoint de autonomia e fixacao ativa](#subcamada-156-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 15.1: O conceito na vida real

### A analogia da catedral e o laudo pericial

Uma catedral nao e construida pelo empilhamento desordenado de blocos de pedra. Sem uma planta arquitetonica estruturada, as forcas de tensao provocam rachaduras e o desabamento do teto:
- Os **alicerces subterraneos** sustentam o peso da obra (Introducao e Fundamentacao Teorica).
- Os **pilares de contraforte** suportam as cargas laterais com precisao geometrica (Metodologia Experimental).
- O **altar central** exibe os fatos materiais em sua forma bruta e visivel (Resultados Numericos).
- A **cupula iluminada pelos vitrais** conecta a obra com o ambiente e sintetiza o seu proposito (Discussao e Conclusao).

No meio academico, um artigo cientifico funciona de forma analoga ao relatorio pericial que um detetive forense entrega ao juiz:
1. Descreve a ocorrencia e o problema investigado (Introducao).
2. Detalha os procedimentos tecnicos de coleta, reagentes e instrumentos adotados (Metodologia).
3. Apresenta as medidas laboratoriais e laudos sem emitir juizo subjetivo (Resultados).
4. Explica por que aqueles numeros confirmam ou refutam as teses alegadas pelas partes (Discussao).
5. Formaliza a conclusao perante o tribunal (Conclusao).

### A demarcacao fundamental: Resultados versus Discussao

O motivo mais comum de rejeicao preliminar (*desk reject*) por revisores em periodicos indexados (como IEEE, Elsevier ou Springer) e a contaminacao da secao de Resultados com especulacoes ou adjetivos:
- Na secao de **Resultados**, o texto deve ser frio, factual e objetivo: *"O modelo enxuto apresentou F1-score de 0.8610 no teste cego, enquanto o baseline atingiu 0.8373."*
- Na secao de **Discussao**, o autor tem a obrigacao analitica de explicar o mecanismo causal: *"O ganho de 2.37 pontos percentuais corrobora a hipotese de que a eliminacao de colunas de ruido atenuou o sobreajuste estrutural das arvores, alinhando-se as predicoes do Fenomeno de Hughes."*

**A grande sacada:** registrar e testar hipoteses formais ($H_1$ e $H_2$) impede o erro metodologico de HARKing (*Hypothesizing After Results are Known*), conferindo credibilidade inatacavel ao experimento perante a comunidade cientifica.

| Secao IMRaD | Pergunta Central | Tempo Verbal Predominante | Conteudo Permitido |
| :--- | :--- | :--- | :--- |
| **Introducao** | *Qual e a lacuna e por que ela e relevante?* | Presente do indicativo | Contextualizacao, formulacao do problema e definicao de $H_1$ e $H_2$. |
| **Metodologia** | *Como o protocolo foi desenhado para teste?* | Passado (voz passiva) | Descricao detalhada do dataset, particoes, algoritmos e formulas. |
| **Resultados** | *Quais medidas numericas foram registradas?* | Passado simples | Tabelas, graficos e metricas brutas sem adjetivacao ou conjecturas. |
| **Discussao** | *O que os numeros significam perante a teoria?* | Presente / Passado | Interpretacao causal, confronto com literatura, limitacoes e aplicacoes. |

---

## Subcamada 15.2: Desenhando o conceito

O diagrama abaixo ilustra o modelo conceitual do funil cientifico IMRaD, demonstrando como o texto comeca abrangente, afunila na coleta experimental e reabre para o contexto social e clinico:

```text
O FUNIL CIENTIFICO IMRaD
==================================================================================
 \   1. INTRODUCAO (AMPLA)          /  -> Panorama geral da saude e sobrecarga
  \                                /      de exames laboratoriais desnecessarios.
   \  2. METODOLOGIA              /    -> Protocolo estrito e reproduzivel:
    \  (PRECISA E CONTROLADA)    /        2.000 amostras, TreeSHAP, shap-select, Optuna.
     | 3. RESULTADOS            |      -> Gargalo do funil: relatorio factual frio.
     |   (DADOS BRUTOS NEUTROS) |         Tabelas de KPIs, graficos e latencias apuradas.
    / 4. DISCUSSAO ANALITICA     \     -> Reabertura: interpretacao do ganho de F1,
   /   (CONFRONTO COM A TEORIA)   \       confronto com RFE e fechamento do overfitting.
  /   5. CONCLUSAO (IMPACTO)       \   -> Veredito final: viabilidade de implantacao
 /                                  \     e direcionamento para futuras pesquisas.
==================================================================================
```

A organizacao modular do gerador automatizado de artigos ([gerar_artigo_word.py](../gerar_artigo_word.py)) estabelece a conversao programatica de metricas em documento formal:

```text
[ pipeline_completo.py ]
(Executa os experimentos numericos e salva tabelas e figuras em assets/)
             │
             ▼
[ gerar_artigo_word.py ]
(Carrega templates, insere metadados, formata tabelas ABNT/APA e insere graficos)
             │
             ▼
[ docs/artigo_xai_reduction.docx ]
- Artigo cientifico completo com 14 secoes estruturadas
- Conformidade tipografica para submissao a periodicos Qualis A
```

| Elemento do Artigo | Papel Metodologico | Risco de Inconformidade |
| :--- | :--- | :--- |
| **Hipotese $H_1$** | Define a meta de preservacao diagnostica ($\Delta F_1 \ge -0.05$) | Sem ela, a reducao dimensional pode ser taxada de prejudicial |
| **Hipotese $H_2$** | Define a meta de eficiencia operacional ($\Delta T \ge 40\%$) | Sem ela, o ganho de tempo carece de criterio de sucesso a priori |
| **Tabelas de Resultados** | Apresentacao concisa de dados do teste cego | Inclusao de interpretacoes gera censura dos revisores |
| **Limitacoes do Estudo** | Reconhecimento de fronteiras experimentais | Omissao de limitacoes e interpretada como fragilidade cientifica |

---

## Subcamada 15.3: Desmistificando a teoria e a notacao formal

### A formalizacao das hipoteses de pesquisa

Para conferir rigor cientifico a investigacao, a metodologia do projeto fundamenta-se no teste formal de duas hipoteses direcionais:

#### Hipotese 1 ($H_1$): Equivalencia diagnostica sob reducao dimensional

Define-se a diferenca de desempenho diagnostico entre o modelo compacto ($\mathcal{M}_{\text{enxuto}}$) e o modelo baseline de forca bruta ($\mathcal{M}_{\text{base}}$) sobre a particao de teste:

$$\Delta F_1 = F_1\left(\mathcal{M}_{\text{enxuto}}\right) - F_1\left(\mathcal{M}_{\text{base}}\right)$$

Sob o teste de nao-inferioridade com margem de equivalencia clinica tolerada de $\epsilon = 0.05$ e corte de pelo menos 50% dos atributos:

$$H_{1,0}: \Delta F_1 < -0.05 \quad \text{contra} \quad H_{1,1}: \Delta F_1 \ge -0.05, \quad \text{com } \left(1 - \frac{p_{\text{enxuto}}}{p_{\text{base}}}\right) \ge 0.50$$

A rejeicao da hipotese nula $H_{1,0}$ comprova que a reducao dimensional preservou a capacidade diagnostica do modelo dentro do limiar de seguranca clinica.

#### Hipotese 2 ($H_2$): Superioridade em eficiencia operacional e latencia

Sejam $T_{\text{train}}$ o tempo de ajuste do modelo e $\tau_{\text{infer}}$ a latencia media de inferencia individual. Define-se a hipotese de reducao simultanea de pelo menos 40% nos custos computacionais:

$$H_{2,1}: \left( 1 - \frac{T_{\text{train}}(\text{enxuto})}{T_{\text{train}}(\text{base})} \ge 0.40 \right) \;\wedge\; \left( 1 - \frac{\tau_{\text{infer}}(\text{enxuto})}{\tau_{\text{infer}}(\text{base})} \ge 0.40 \right)$$

| Simbolo | Significado Formal | Leitura no Projeto |
| :--- | :--- | :--- |
| $\Delta F_1$ | Variacao do $F_1$-score no teste cego | Observado: $+0.0237$ (ganho de 2.37 pts) |
| $\epsilon$ | Margem maxima tolerada de degradacao | Fixado *a priori* em $0.05$ |
| $p_{\text{base}}, p_{\text{enxuto}}$ | Dimensionalidade inicial e final | 40 atributos reduzidos para 10 atributos (corte de 75.0%) |
| $T_{\text{train}}$ | Tempo de treinamento da floresta (ms) | Queda de 600 ms para 185 ms (reducao de 69.2%) |
| $\tau_{\text{infer}}$ | Latencia de inferencia por paciente ($\mu$s) | Queda de 30.8 $\mu$s para 12.4 $\mu$s (reducao de 59.7%) |

### A ordem correta evita o vicio de HARKing

As metas quantitativas das hipoteses $H_1$ ($\ge 50\%$ de corte, tolerância de $\pm 0.05$ em $F_1$) e $H_2$ ($\ge 40\%$ de ganho computacional) devem ser formalizadas no projeto de pesquisa antes da execucao dos experimentos. Ajustar as metas apos a observacao dos numeros descaracteriza a metodologia hipotetico-dedutiva.

---

## Subcamada 15.4: Laboratorio ludico no Colab

Execute o bloco abaixo no Google Colab para simular o motor de avaliacao logica das hipoteses cientificas:

```python
# =============================================================================
# CAMADA 15: LABORATORIO LUDICO DE VALIDACAO DE HIPOTESES CIENTIFICAS
# Demonstracao: Testador Logico-Estatistico das Hipoteses H1 e H2
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt

# 1. Resultados consolidados do experimento oficial
dados_baseline = {"atributos": 40, "f1": 0.8373, "tempo_treino_ms": 601.7, "latencia_us": 30.8}
dados_campeao  = {"atributos": 10, "f1": 0.8610, "tempo_treino_ms": 185.2, "latencia_us": 12.4}

# 2. Avaliacao formal da Hipotese H1 (Preservacao Diagnostica)
taxa_corte = (1.0 - (dados_campeao["atributos"] / dados_baseline["atributos"])) * 100.0
delta_f1 = dados_campeao["f1"] - dados_baseline["f1"]
h1_valida = (taxa_corte >= 50.0) and (delta_f1 >= -0.05)

# 3. Avaliacao formal da Hipotese H2 (Eficiencia Computacional)
ganho_treino = (1.0 - (dados_campeao["tempo_treino_ms"] / dados_baseline["tempo_treino_ms"])) * 100.0
ganho_latencia = (1.0 - (dados_campeao["latencia_us"] / dados_baseline["latencia_us"])) * 100.0
h2_valida = (ganho_treino >= 40.0) and (ganho_latencia >= 40.0)

print("PARECER CIENTIFICO DE CONFORMIDADE EXPERIMENTAL:")
print("=" * 68)
print(f"Hipotese H1 (Preservacao Clinica sob Reducao de Dados):")
print(f"  - Taxa de Reducao de Atributos : {taxa_corte:.1f}% (Meta minima: >= 50.0%)")
print(f"  - Variacao de F1-Score (Delta) : {delta_f1:+.4f} (Margem de tolerancia: >= -0.0500)")
print(f"  - Conclusao de H1              : {'CONFIRMADA COM SUCESSO' if h1_valida else 'REJEITADA'}")
print("-" * 68)
print(f"Hipotese H2 (Eficiencia Computacional e Inferencia):")
print(f"  - Reducao no Tempo de Treino   : {ganho_treino:.1f}% (Meta minima: >= 40.0%)")
print(f"  - Reducao na Latencia          : {ganho_latencia:.1f}% (Meta minima: >= 40.0%)")
print(f"  - Conclusao de H2              : {'CONFIRMADA COM SUCESSO' if h2_valida else 'REJEITADA'}")
print("=" * 68)

# 4. Visualizacao grafica do teste de margem
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh(["H1: Delta F1"], [delta_f1], color="#27ae60", edgecolor="black", height=0.4)
ax.axvline(0, color="black", linestyle="-", linewidth=1.0)
ax.axvline(-0.05, color="#c0392b", linestyle="--", linewidth=1.2, label="Margem Critica de Tolerancia (-0.05)")
ax.set_xlim(-0.08, 0.05)
ax.set_title("Validacao Geometrica da Hipotese H1 no Teste Cego", fontsize=11, fontweight="bold")
ax.set_xlabel("Variacao no F1-Score (Delta F1)", fontsize=10)
ax.grid(axis="x", linestyle=":", alpha=0.6)
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()
```

> **O que voce deve notar no grafico gerado:**
> 1. O delta observado de $F_1$ posiciona-se no campo positivo ($+0.0237$), superando confortavelmente a margem de equivalencia tolerada de $-0.05$.
> 2. Ambas as hipoteses foram validadas com folga estatistica sobre os dados independentes.

**Mini-experimento:** altere os dados do modelo campeao para simular um cenario onde o $F_1$ tivesse caido para `0.7500`. Como o testador responderia? Qual argumento da secao de Discussao precisaria ser reformulado?

---

## Subcamada 15.5: O momento serio da nossa aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

No fluxo de publicacao academica do projeto, o script [gerar_artigo_word.py](../gerar_artigo_word.py) compila automaticamente o documento formal [docs/artigo_xai_reduction.docx](../docs/artigo_xai_reduction.docx) consolidando as hipoteses e os quadros de evidencias.

```python
# =============================================================================
# APLICACAO REAL: AUDITORIA DAS HIPOTESES SOBRE OS RESULTADOS DO PROJETO
# Base oficial: 2.000 pacientes, dados sincronizados com artigo_xai_reduction.docx
# =============================================================================
import pandas as pd

quadro_hipoteses = pd.DataFrame({
    "Hipotese de Pesquisa": [
        "H1: Equivalencia Diagnostica",
        "H2: Eficiencia Computacional (Treino)",
        "H2: Eficiencia de Inferencia (Latencia)"
    ],
    "Meta Pre-Estabelecida": [
        "Corte >= 50% de atributos com Delta F1 >= -0.05",
        "Reducao no tempo de ajuste de pelo menos 40%",
        "Reducao na latencia por paciente de pelo menos 40%"
    ],
    "Resultado Numerico Obtido": [
        "75.0% de corte (40 -> 10) com Delta F1 = +0.0237",
        "Tempo caiu de 601.7 ms para 185.2 ms (queda de 69.2%)",
        "Latencia caiu de 30.8 us para 12.4 us (queda de 59.7%)"
    ],
    "Veredito Formal": [
        "CONFIRMADA E SUPERADA",
        "CONFIRMADA E SUPERADA",
        "CONFIRMADA E SUPERADA"
    ]
})

print("=" * 82)
print("AUDITORIA ACADEMICA DE HIPOTESES CIENTIFICAS (PROJETO XAI REDUCTION)")
print("=" * 82)
print(quadro_hipoteses.to_string(index=False))
print("=" * 82)
```

### Tabela oficial de KPIs

> Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Como e calculado | Pergunta operacional |
| :--- | :--- | :--- |
| **Taxa de Confirmacao de Hipoteses** | Proporcao de hipoteses cientificas formalmente validadas | As metas teoricas estabelecidas *a priori* foram corroboradas pelos dados? |
| **Margem de Seguranca de $H_1$** | $\Delta F_1 - (-\epsilon)$ em relacao ao limiar critico | Qual foi a folga estatistica acima da margem de perda clinica tolerada? |
| **Margem de Eficiencia de $H_2$** | Reducao percentual observada subtraida da meta minima de 40% | Quanto a otimizacao excedeu a expectativa minima de aceleracao computacional? |
| **Conformidade Estrutural IMRaD** | Checagem de presenca de todas as 14 secoes no `.docx` gerado | O artigo cumpre os padroes internacionais exigidos para submissao a revistas Qualis A? |

### Interpretacao clinica e de negocio

A comprovacao das hipoteses $H_1$ e $H_2$ produz consequencias diretas para a sustentabilidade da pesquisa:

1. **Aprovacao por comites de etica e revisao clinica:** demonstrar que a Hipotese 1 foi superada (com sensibilidade superior no modelo enxuto) garante que a simplificacao de exames nao expoe pacientes ao risco de Falsos Negativos camuflados.
2. **Robustez metodologica perante revisores de alto impacto:** a separacao estrita entre a secao neutra de Resultados e a secao causal de Discussao atende rigorosamente as diretrizes editoriais internacionais, reduzindo o tempo de tramitacao e revisao por pares.
3. **Escalabilidade industrial:** a validacao da Hipotese 2 certifica que a infraestrutura hospitalar existente e plenamente compativel com a solucao proposta, dispensando investimentos vultosos em clusters de processamento.

---

## Subcamada 15.6: Checkpoint de autonomia e fixacao ativa

Explique sem consultar o texto e depois confira sua resposta:

1. **Qual e a diferenca fundamental de proposito, conteudo e tempo verbal entre a secao de Resultados e a secao de Discussao no formato IMRaD?**
2. **Por que a formulacao *a priori* de hipoteses cientificas como $H_1$ e $H_2$ e essencial para a integridade de uma pesquisa em computacao?**
3. **O que e o vicio metodologico de HARKing e como o pipeline do projeto se protege contra ele?**
4. **Como o conceito de teste de nao-inferioridade foi aplicado na definicao da Hipotese 1?**
5. **Se os dados tivessem indicado um corte de 80% dos atributos com queda de 0.15 no $F_1$-score, qual seria o procedimento cientifico correto a ser relatado no artigo?**
6. **Qual e a funcao do script `gerar_artigo_word.py` na esteira de publicacao da pesquisa?**

### Mini-desafio pratico

Analise a sensibilidade das conclusoes simulando diferentes niveis de tolerancia para a margem clinica $\epsilon$ e preencha a tabela:

| Margem de Tolerancia ($\epsilon$) | $\Delta F_1$ Observado ($+0.0237$) | Veredito de $H_1$ | Comportamento Clinico |
| :--- | :--- | :--- | :--- |
| $\epsilon = 0.01$ (Margem Estrita) | $+0.0237$ | | |
| $\epsilon = 0.05$ (Padrao do Projeto) | $+0.0237$ | | |
| $\epsilon = 0.10$ (Margem Ampla) | $+0.0237$ | | |

**Pergunta reflexiva:** o fato de o $\Delta F_1$ observado ser estritamente positivo dispensaria inclusive a necessidade de recorrer a margem de tolerancia para validar a equivalencia diagnostica?
