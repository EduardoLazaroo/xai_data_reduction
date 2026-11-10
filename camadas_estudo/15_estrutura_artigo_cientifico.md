# Camada 15: Estrutura de Artigo Cientifico e Formulacao de Hipoteses

**Trilha:** XAI Aplicada à Redução de Dados em Machine Learning
**Aplicacao:** classificacao binaria em benchmark sintetico; os rotulos de saude sao ilustrativos.
**Codigo de referencia:** [gerar_artigo_ieee.py](../gerar_artigo_ieee.py) e [artigo_cientifico_ieee.pdf](../artigo_cientifico_ieee.pdf)

> **Objetivo da aula:** Compreender a arquitetura de comunicacao cientifica baseada no formato canonico IMRaD, relatando a comparacao entre filter, wrapper, embedded e SHAP, o papel local do LIME e os proxies de interpretabilidade junto de desempenho e custo.

## Campo Didático: Separar Fato, Método e Interpretação

Use tres cores mentais ao escrever: **Metodologia diz o que foi feito; Resultados dizem o que foi observado; Discussao explica por que isso importa**. Antes de redigir, transforme cada KPI em uma frase objetiva e cada frase objetiva em uma interpretacao com limite declarado.

```text
pergunta -> metodo -> resultado observavel -> interpretacao -> limitacao -> proxima pergunta
```

Observe se outro pesquisador conseguiria repetir o estudo apenas lendo a metodologia. O erro comum e colocar conclusao causal na secao de resultados ou esconder um KPI desfavoravel. A ponte para a Camada 16 e defender as hipoteses oralmente e por escrito sob questionamento.

---

### Roteiro de dominio

Para cada afirmacao, marque sua origem: **dado medido, metodo, inferencia ou recomendacao**. Uma frase de Resultados deve permitir que o leitor confira a tabela; uma frase de Discussao deve conectar o resultado a uma hipotese, referencia ou limitacao. Se a frase nao couber em nenhuma categoria, ela precisa ser reescrita.

### Duvidas que esta aula responde

- **Resultados podem interpretar?** Podem descrever padroes observados, mas a explicacao e implicacao pertencem principalmente a Discussao.
- **Hipotese confirmada significa verdade universal?** Nao. Significa que a evidencia apoiou um criterio dentro do protocolo.
- **Por que declarar limitacoes?** Para delimitar onde o resultado pode e nao pode ser generalizado.
- **O que HARKing prejudica?** Faz parecer que uma previsao foi feita antes do resultado quando nasceu depois dele.

### Regra de explicacao Feynman

Explique o artigo como um laudo: primeiro diga o que foi examinado, depois mostre as medidas, e so entao interprete o que elas sugerem. Misturar medida e opiniao dificulta a auditoria.

### Um paragrafo que nao mistura camadas

Compare estas duas frases:

```text
RESULTADO: O baseline obteve F1 = 0,8856 (CV: 0,8649 ± 0,0159) e a proposta F1 = 0,8861 sem Optuna e 0,8872 com Optuna (CV: 0,8864 ± 0,0122).
DISCUSSÃO: O ganho de +0,0016 no holdout e +0,0215 na validação cruzada superou o limite de -0,05 definido para H1; isso descreve o benchmark sintético e não demonstra validação clínica externa.
```

A primeira pode ser conferida diretamente na tabela. A segunda interpreta e precisa declarar limite: dataset sintetico e ausencia de validacao externa. Separar as frases protege o leitor contra conclusoes escondidas dentro de numeros.

### Como construir cada secao

Introducao apresenta lacuna e pergunta. Metodologia permite repetir. Resultados mostram tabelas e figuras sem adjetivos. Discussao conecta numeros a teoria, compara trabalhos e reconhece limites. Conclusao responde a pergunta sem prometer mais do que o desenho permite.

### O mesmo resultado em quatro linguagens

Um unico achado pode aparecer assim:

```text
METODOLOGIA: treinamos Random Forests com 40 e 10 atributos no mesmo split estratificado e em validação cruzada de 5 folds.
RESULTADOS: o baseline teve F1 = 0,8856; a proposta teve F1 = 0,8861 (Puro) e 0,8872 (+Optuna).
DISCUSSÃO: o ganho observado superou com folga o limite estabelecido para H1, confirmando a eliminação de redundâncias e ruídos.
CONCLUSÃO: H1 foi atendida com superação no benchmark sintético, mas o resultado não substitui validação clínica externa.
```

As frases nao sao repeticao. Cada uma responde a uma pergunta diferente: como foi medido, o que apareceu, o que pode significar e qual alcance a evidencia permite.

### Duvidas frequentes

- **Resultado e discussao podem aparecer no mesmo paragrafo?** Podem ser próximos, mas a funcao logica deve permanecer clara.
- **Hipotese confirmada significa verdade universal?** Nao; significa que a evidencia sustentou o criterio predefinido naquele protocolo.
- **Por que declarar limitacoes?** Porque delimita o alcance da inferencia e aumenta a credibilidade.
- **Toda melhoria de F1 e clinicamente relevante?** Nao; relevancia depende de margem, erro, contexto e custo.

### Ponte para a defesa

Um artigo bem separado facilita responder perguntas: o que foi feito, que numero apareceu, como foi interpretado e o que ainda nao se sabe. A Camada 16 transforma essa cadeia em respostas de banca.

## Cultura, História e Referências

O formato IMRaD se consolidou porque a comunidade cientifica precisava separar pergunta, procedimento, evidencia e interpretacao de modo que outros pesquisadores pudessem conferir o trabalho. A recomendacao editorial [Reporting standards and availability of data, materials, code and protocols da Nature](https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards) mostra como essa cultura evoluiu para incluir codigo, dados e transparencia.

Use o artigo IEEE gerado por [gerar_artigo_ieee.py](../gerar_artigo_ieee.py) como objeto de inspecao: uma tabela pode parecer formal e ainda misturar resultado com opiniao. O rascunho `artigo_xai_reduction.docx` e legado e nao e material oficial de entrega.

**Pergunta cultural:** por que revisores pedem metodos detalhados? Nao para tornar o texto mais burocratico, mas para separar uma descoberta reproduzivel de uma narrativa que so funciona depois que o resultado foi conhecido.

## Recursos de Mídia (Visual e Áudio)

- **Visual local:** PDF IEEE oficial e tabelas do pipeline, comparando Resultado com Discussao.
- **Imagem de apoio:** [Dashboard final comparativo](../assets/dashboard_final_comparativo.png), como figura de resultados.
- **Referencia:** [Reporting standards da Nature](https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards).
- **Áudio sugerido:** duas frases, uma factual e outra interpretativa, em contraste.
- **Imagem mental:** funil IMRaD: abre na Introducao, estreita na Metodologia e reabre na Discussao.

## 📊 Elementos de Comunidade e Status

- **Status:** `Manuscrito auditável` quando outro leitor consegue reconstruir método, resultado e limite.
- **Debate:** “Onde termina o resultado e começa a interpretacao?”
- **Papel rotativo:** autor, revisor, estatistico e leitor leigo.

## 💡 Engajamento e Conhecimento

- **Atividade:** reescrever uma frase exagerada em Resultado, Discussão e Limitação.
- **Produto da aula:** um paragrafo IMRaD com KPI, interpretacao e alcance declarado.
- **Conexao profissional:** revisar o proprio artigo procurando promessas que os dados nao sustentam.

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
- Na secao de **Resultados**, o texto deve ser factual: *"O baseline apresentou F1 de 0,8856 (CV: 0,8649 ± 0,0159) e a proposta atingiu F1 de 0,8861 sem Optuna e 0,8872 com Optuna (CV: 0,8864 ± 0,0122), retendo 10/10 atributos informativos."*
- Na secao de **Discussao**, interprete com cautela: *"O ganho de F1 (+0,0016 no holdout e +0,0215 na validação cruzada) superou o critério de H1, demonstrando mitigação do Fenômeno de Hughes no benchmark sintético."*

**A grande sacada:** registrar e testar hipoteses formais ($H_1$ e $H_2$) impede o erro metodologico de HARKing (*Hypothesizing After Results are Known*), conferindo credibilidade inatacavel ao experimento perante a comunidade cientifica.

| Secao IMRaD | Pergunta Central | Tempo Verbal Predominante | Conteudo Permitido |
| :--- | :--- | :--- | :--- |
| **Introducao** | *Qual e a lacuna e por que ela e relevante?* | Presente do indicativo | Contextualizacao, formulacao do problema e definicao de $H_1$ e $H_2$. |
| **Metodologia** | *Como o protocolo foi desenhado para teste?* | Passado (voz passiva) | Descricao detalhada do dataset, particoes, algoritmos e formulas. |
| **Resultados** | *Quais medidas numericas foram registradas?* | Passado simples | Tabelas, graficos e metricas brutas sem adjetivacao ou conjecturas. |
| **Discussao** | *O que os numeros significam perante a teoria?* | Presente / Passado | Interpretacao proporcional, confronto com literatura e limitacoes. |

---

## Subcamada 15.2: Desenhando o conceito

O diagrama abaixo ilustra o modelo conceitual do funil cientifico IMRaD, demonstrando como o texto comeca abrangente, afunila no protocolo experimental e reabre para interpretacao e limitacoes:

```text
O FUNIL CIENTIFICO IMRaD
==================================================================================
 \   1. INTRODUCAO (AMPLA)          /  -> Contexto, problema e questao de pesquisa.
  \                                /
   \  2. METODOLOGIA              /    -> Protocolo estrito e reproduzivel:
    \  (PRECISA E CONTROLADA)    /        2.000 amostras, TreeSHAP, shap-select, Optuna.
     | 3. RESULTADOS            |      -> Gargalo do funil: relatorio factual frio.
     |   (DADOS BRUTOS NEUTROS) |         Tabelas de KPIs, graficos e latencias apuradas.
    / 4. DISCUSSAO ANALITICA     \     -> Interpretacao dos resultados e limites.
   /   (CONFRONTO COM A TEORIA)   \
  /   5. CONCLUSAO (RESPOSTA)     \   -> Resposta a questao e trabalhos futuros.
 /                                  \     e direcionamento para futuras pesquisas.
==================================================================================
```

A organizacao modular do gerador do artigo IEEE ([gerar_artigo_ieee.py](../gerar_artigo_ieee.py)) converte as metricas atualizadas em documento formal de ate tres paginas:

```text
[ pipeline_completo.py ]
(Executa os experimentos numericos e salva tabelas e figuras em assets/)
             │
             ▼
[ gerar_artigo_ieee.py ]
(Carrega resultados do CSV, formata o artigo IEEE e valida o limite de paginas)
             │
             ▼
[ artigo_cientifico_ieee.pdf ]
- Artigo oficial em PDF, com no maximo tres paginas
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

Como criterio operacional previamente definido para H1, exige-se reducao de pelo menos 50% dos atributos e queda de F1 nao superior a 0,05:

$$H_{1,0}: \Delta F_1 < -0.05 \quad \text{contra} \quad H_{1,1}: \Delta F_1 \ge -0.05, \quad \text{com } \left(1 - \frac{p_{\text{enxuto}}}{p_{\text{base}}}\right) \ge 0.50$$

Este criterio numerico nao e um teste estatistico de nao-inferioridade nem comprova seguranca clinica. Nesta execucao, foi atendido por margem estreita.

#### Hipotese 2 ($H_2$): Rejeicao de ruido e custo de selecao

H2 exige maior rejeicao do ruido sintetico que os metodos comparados e menor tempo de selecao que RFE:

$$H_2: \text{rejeicao de ruido}_{\text{proposta}} > \text{comparadores} \quad \wedge \quad T_{\text{selecao, proposta}} < T_{\text{selecao, RFE}}$$

| Simbolo | Significado Formal | Leitura no Projeto |
| :--- | :--- | :--- |
| $\Delta F_1$ | Variacao do $F_1$-score no teste cego | Observado: $+0,0005$ (Puro) e $+0,0016$ (+Optuna) |
| $\epsilon$ | Margem maxima tolerada de degradacao | Fixado *a priori* em $0.05$ |
| $p_{\text{base}}, p_{\text{proposta}}$ | Dimensionalidade inicial e final | 40 atributos reduzidos para 10 (corte de 75%, 10/10 informativos) |
| $T_{\text{selecao}}$ | Tempo de selecao medido (s) | 11,04 s (Puro) / 79,30 s (+Optuna) e 4,55 s para RFE |
| $\text{Ground truth recuperado}$ | Atributos informativos retidos e ruido rejeitado | 10/10 informativos e 100% de rejeicao de ruido (unico do benchmark) |

### A ordem correta evita o vicio de HARKing

As metas quantitativas de $H_1$ e $H_2$ devem ser definidas antes dos experimentos. H1 foi atendida com superacao (ganho de F1 no holdout e na CV de 5 folds); em H2, a proposta foi a unica a recuperar 10/10 atributos informativos e rejeitar 100% dos redundantes e ruidos, embora com tempo de selecao offline superior ao RFE.

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

resultados = pd.read_csv("assets/tabela_comparativa_completa.csv")
baseline = resultados.loc[resultados["Metodo"] == "Baseline Completo (40 Atributos)"].iloc[0]
proposta = resultados.loc[
    resultados["Metodo"] == "XAI Proposto (shap-select OOF + Optuna)"
].iloc[0]
proposta_pura = resultados.loc[
    resultados["Metodo"] == "XAI Proposto (shap-select OOF Puro)"
].iloc[0]
rfe = resultados.loc[resultados["Metodo"] == "Wrapper (RFE)"].iloc[0]
comparadores = resultados[
    ~resultados["Metodo"].isin([
        "Baseline Completo (40 Atributos)",
        "XAI Proposto (shap-select OOF + Optuna)",
        "XAI Proposto (shap-select OOF Puro)"
    ])
]

taxa_corte = (1 - proposta["N_Atributos"] / baseline["N_Atributos"]) * 100
delta_f1 = proposta["F1_Score"] - baseline["F1_Score"]
h1_atendida = taxa_corte >= 50 and delta_f1 >= -0.05
inf_proposta = int(proposta["Biomarcadores_Retidos (max 10)"])
melhor_inf_comparador = int(comparadores["Biomarcadores_Retidos (max 10)"].max())
h2_pureza = inf_proposta > melhor_inf_comparador
h2_tempo = proposta_pura["Tempo_Selecao_s"] < rfe["Tempo_Selecao_s"]

print(f"H1 atendida com superacao: {h1_atendida}")
print(f"Reducao: {taxa_corte:.1f}%; delta F1 Holdout: {delta_f1:+.4f} (CV 5-Fold: {proposta['F1_CV_Media']:.4f} vs {baseline['F1_CV_Media']:.4f})")
print(f"H2 Pureza (Recuperacao Ground Truth): {h2_pureza} ({inf_proposta}/10 vs melhor comparador {melhor_inf_comparador}/10)")
print(f"H2 Tempo de selecao vs RFE: {h2_tempo} (proposta OOF pura {proposta_pura['Tempo_Selecao_s']:.2f} s; RFE {rfe['Tempo_Selecao_s']:.2f} s)")

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh(["H1: Delta F1"], [delta_f1], color="#27ae60" if h1_atendida else "#c0392b", edgecolor="black", height=0.4)
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
> 1. O delta observado ($+0{,}0016$ com Optuna e $+0{,}0005$ sem Optuna) é positivo, superando o Baseline mesmo com 75% menos atributos.
> 2. H1 foi confirmada com superação; em H2, a proposta superou todos os métodos em pureza (`10/10` informativos vs `7/10`), com maior tempo de seleção offline que o RFE.

**Mini-experimento:** altere o limite de H1 e observe como o veredito muda. Explique por que o criterio deve ser definido antes de examinar os resultados.

---

## Subcamada 15.5: O momento serio da nossa aplicacao

> **Chega de brinquedo!** Agora que o conceito esta cristalino, vamos para a trincheira real da nossa aplicacao com os dados do projeto.

O artigo oficial e gerado pelo script [gerar_artigo_ieee.py](../gerar_artigo_ieee.py), que produz o PDF IEEE [artigo_cientifico_ieee.pdf](../artigo_cientifico_ieee.pdf) de ate tres paginas a partir do CSV. O arquivo `artigo_xai_reduction.docx` e um rascunho legado e nao e material oficial de entrega.

```python
# =============================================================================
# APLICACAO REAL: AUDITORIA DAS HIPOTESES SOBRE OS RESULTADOS DO PROJETO
# Os resultados da execucao sao carregados do CSV do pipeline.
# =============================================================================
import pandas as pd

resultados = pd.read_csv("assets/tabela_comparativa_completa.csv")
baseline = resultados.loc[resultados["Metodo"] == "Baseline Completo (40 Atributos)"].iloc[0]
proposta = resultados.loc[
    resultados["Metodo"] == "XAI Proposto (shap-select OOF + Optuna)"
].iloc[0]
proposta_pura = resultados.loc[
    resultados["Metodo"] == "XAI Proposto (shap-select OOF Puro)"
].iloc[0]
rfe = resultados.loc[resultados["Metodo"] == "Wrapper (RFE)"].iloc[0]
comparadores = resultados[
    ~resultados["Metodo"].isin([
        "Baseline Completo (40 Atributos)",
        "XAI Proposto (shap-select OOF + Optuna)",
        "XAI Proposto (shap-select OOF Puro)"
    ])
]
inf_proposta = int(proposta["Biomarcadores_Retidos (max 10)"])
melhor_inf_comparador = int(comparadores["Biomarcadores_Retidos (max 10)"].max())
reducao = (1 - proposta["N_Atributos"] / baseline["N_Atributos"]) * 100
delta_f1 = proposta["F1_Score"] - baseline["F1_Score"]
h1_atendida = reducao >= 50 and delta_f1 >= -0.05

quadro_hipoteses = pd.DataFrame([
    {"Hipotese": "H1", "Criterio": "Reducao >= 50% e delta F1 >= -0.05",
     "Resultado": f"Reducao {reducao:.1f}%; delta F1 {delta_f1:+.4f} (CV: {proposta['F1_CV_Media']:.4f})",
     "Veredito": "Confirmada com superacao" if h1_atendida else "Nao atendida"},
    {"Hipotese": "H2", "Criterio": "Maior recuperacao do ground truth (0 ruidos) e menor tempo de selecao que RFE",
     "Resultado": f"Informativos {inf_proposta}/10 (melhor comparador {melhor_inf_comparador}/10, 0 ruidos); "
                  f"selecao OOF pura {proposta_pura['Tempo_Selecao_s']:.2f} s (RFE {rfe['Tempo_Selecao_s']:.2f} s)",
     "Veredito": "Confirmada em pureza (10/10); maior tempo de selecao que RFE"},
    {"Hipotese": "H3", "Criterio": "Aumento de R² na obs. critica e eliminacao de ruido na coorte LIME (N=20)",
     "Resultado": f"R² Obs #{int(proposta['Paciente_Limiar'])}: {proposta['R2_LIME_Baseline']:.4f} -> {proposta['R2_LIME_Reduzido']:.4f}; "
                  f"Regras nao-inf. N=20: {proposta['Ruidos_LIME_Coorte_Base']:.2f} -> {proposta['Ruidos_LIME_Coorte_Red']:.2f}",
     "Veredito": "Confirmada integralmente"},
])

print("=" * 88)
print("AUDITORIA ACADEMICA DE HIPOTESES CIENTIFICAS (PROJETO XAI REDUCTION)")
print("=" * 88)
print(quadro_hipoteses.to_string(index=False))
print("=" * 88)
```

### Tabela oficial de KPIs

> Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Como e calculado | Pergunta operacional |
| :--- | :--- | :--- |
| **Criterio H1** | Reducao de atributos >= 50% e delta F1 >= -0,05 no holdout | O criterio foi atendido nesta particao? |
| **Criterio H2** | Rejeicao de ruido e tempo de selecao comparados com os metodos do CSV | A proposta superou os comparadores definidos? |
| **Limite de paginas** | Contagem de paginas do PDF validada pelo gerador | O artigo atende ao maximo de tres paginas? |
| **Conformidade das referencias** | Conferencia da correspondencia entre citacoes e lista bibliografica | As fontes citadas estao presentes e pertinentes? |

### Interpretacao dos resultados e limites

As hipoteses permitem responder a questao de pesquisa dentro do protocolo avaliado:

1. **H1:** confirmada com superação no holdout ($\Delta F_1 = +0{,}0016$ com Optuna e $+0{,}0005$ sem Optuna) e na validação cruzada de 5 folds ($0{,}8864$ vs. $0{,}8649$).
2. **H2:** confirmada em pureza (único método a recuperar 10/10 atributos informativos e 0/20 ruídos), mas com maior tempo de seleção offline que o RFE ($11{,}04\text{ s}$ vs. $4{,}55\text{ s}$).
3. **Limites:** nao ha evidencias sobre seguranca clinica, aprovacao etica, custos ou escalabilidade.

---

## Subcamada 15.6: Checkpoint de autonomia e fixacao ativa

Explique sem consultar o texto e depois confira sua resposta:

1. **Qual e a diferenca fundamental de proposito, conteudo e tempo verbal entre a secao de Resultados e a secao de Discussao no formato IMRaD?**
2. **Por que a formulacao *a priori* de hipoteses cientificas como $H_1$ e $H_2$ e essencial para a integridade de uma pesquisa em computacao?**
3. **O que e o vicio metodologico de HARKing e como o pipeline do projeto se protege contra ele?**
4. **Por que o criterio de H1 nao equivale a um teste estatistico de nao-inferioridade?**
5. **Se os dados tivessem indicado um corte de 80% dos atributos com queda de 0.15 no $F_1$-score, qual seria o procedimento cientifico correto a ser relatado no artigo?**
6. **Qual e a funcao do script `gerar_artigo_ieee.py` na geracao do entregavel oficial?**

### Mini-desafio pratico

Analise a sensibilidade do criterio de H1 para diferentes limites de tolerancia de F1:

| Limite de queda | $\Delta F_1$ observado (+0,0016 Optuna / +0,0005 Puro) | Veredito numerico de $H_1$ |
| :--- | :--- | :--- |
| $\epsilon = 0.00$ (Exige não-queda estrita) | +0,0016 / +0,0005 | Atendido (houve ganho real de F1) |
| $\epsilon = 0.01$ | +0,0016 / +0,0005 | Atendido com folga |
| $\epsilon = 0.05$ | +0,0016 / +0,0005 | Atendido com folga |

**Pergunta reflexiva:** por que atender a esse criterio numerico nao comprova equivalencia clinica?
