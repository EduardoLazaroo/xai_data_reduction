# Camada 16: Escrita Academica e Defesa de Hipoteses

**Trilha:** XAI Aplicada à Redução de Dados em Machine Learning
**Aplicação:** classificação binária de saúde (`0 = Saudável`, `1 = Patologia`)
**Código de referência:** [gerar_artigo_ieee.py](../gerar_artigo_ieee.py) e [artigo_cientifico_ieee.pdf](../artigo_cientifico_ieee.pdf)

> **Objetivo da aula:** Consolidar a articulacao entre fundamentacao teorica, modelagem computacional e redacao academica de alto impacto, estruturando a defesa formal das hipoteses H1 e H2, a documentacao metodologica de limitacoes tecnicas e a formulacao de respostas tecnicas padronizadas perante bancas examinadoras e comites de revisao por pares.

## Campo Didático: Defender Sem Exagerar

Treine a defesa em quatro movimentos: **afirme a hipotese, mostre a evidencia, reconheca a limitacao e explique a proxima verificacao**. Uma resposta forte nao e a mais confiante; e a que distingue claramente dado observado, inferencia estatistica e implicacao pratica.

```text
hipotese -> KPI/figura -> interpretacao -> limitacao -> resposta verificavel
```

Observe se voce consegue explicar o resultado para uma banca tecnica e para um gestor sem trocar precisao por jargao. O erro comum e prometer causalidade, economia ou seguranca clinica a partir de um dataset sintetico. A ponte final e transformar o projeto em uma narrativa cientifica honesta, reproduzivel e auditavel.

---

### Roteiro de dominio

Treine cada resposta com a estrutura **afirmacao, evidencia, limite, proximo teste**. Para H1, mostre preservacao de desempenho e reducao de atributos; para H2, mostre rejeicao de ruido e tempo de selecao. Se um criterio falhar, nao esconda: explique qual hipotese nao foi apoiada e o que sera ajustado.

### Duvidas que esta aula responde

- **Como responder “isso funciona na vida real”?** Diga o que o experimento suporta e o que exige validacao externa.
- **Como defender um resultado negativo?** Mostre o protocolo, a metrica, a incerteza e a decisao que o resultado impede.
- **Economia computacional prova economia hospitalar?** Nao. Ela e um indicador tecnico que precisa ser traduzido com custos reais.
- **Uma boa narrativa pode compensar um experimento fraco?** Nao. Escrita melhora a comunicacao, nao substitui evidencia.

### Regra de explicacao Feynman

Explique sua defesa como uma ponte com placas de peso: cada afirmacao precisa de uma evidencia capaz de suporta-la. Quando o peso excede a evidencia, a ponte quebra em uma pergunta simples da banca.

### A estrutura de uma resposta tecnicamente honesta

Uma defesa forte pode seguir quatro frases:

```text
1. Hipotese: o que esperavamos?
2. Evidencia: qual KPI, tabela ou grafico sustenta a resposta?
3. Limite: em que situacao essa evidencia pode nao se repetir?
4. Proximo teste: qual experimento reduz essa incerteza?
```

Exemplo: “Esperavamos reduzir atributos sem queda de F1 superior a 0,05. Nesta particao sintetica, removemos 75% dos atributos e o F1 caiu cerca de 0,0461, atendendo ao criterio por margem estreita. A selecao levou 14,17 s, contra 3,07 s para RFE, portanto H2 nao foi atendida em custo. Esses resultados nao demonstram desempenho clinico externo.”

### Como evitar promessas indevidas

Troque “prova que” por “sustenta, neste protocolo, a hipotese de”. Troque “variavel causa” por “o modelo atribuiu contribuicao”. Troque “economiza vidas” por “pode reduzir custo operacional, sujeito a validacao clinica”. Precisao de linguagem e parte do rigor tecnico, nao uma fraqueza retorica.

### Duvidas frequentes

- **Reconhecer uma limitacao enfraquece a defesa?** Nao; mostra que o alcance do resultado foi compreendido.
- **Uma hipotese precisa ser confirmada?** Nao. Um resultado negativo tambem informa e pode refutar uma expectativa.
- **Como responder quando nao sabe?** Separe o que os dados mostram do que ainda precisa ser medido.
- **Por que explicar para publicos diferentes?** Uma banca avalia rigor; um gestor precisa entender decisao, custo e risco.

### Fechamento do percurso

O especialista nao e quem memoriza nomes de algoritmos. E quem consegue ligar pergunta, dados, metodo, evidencia, limite e decisao sem saltos escondidos. Essa e a competencia que as 16 camadas constroem em conjunto.

## Cultura, História e Referências

A escrita cientifica tambem tem memoria de erros. O termo HARKing, “Hypothesizing After the Results are Known”, foi discutido por Kerr no artigo [HARKing: Hypothesizing After the Results are Known](https://doi.org/10.1207/S15327965PLI1103_04). A ideia e simples: transformar uma descoberta pos-hoc em hipotese pre-registrada distorce a historia da evidencia. Recursos como o [OSF Registries](https://osf.io/registries/) ajudam a registrar perguntas e criterios antes da analise.

Leia ainda os principios [FAIR para dados cientificos](https://www.go-fair.org/fair-principles/) e veja a pasta `assets/` como uma pequena colecao de artefatos que precisa de nomes, origem, versao e contexto. Uma defesa madura nao esconde incerteza: mostra o que foi medido, o que foi inferido e o que ainda precisa ser testado.

**Pergunta cultural:** qual e a diferenca entre contar uma boa historia e fabricar uma boa historia? A primeira organiza evidencias sem ultrapassa-las; a segunda altera o passado para fazer o resultado parecer inevitavel.

## Recursos de Mídia (Visual e Áudio)

- **Visual local:** tabelas, figuras e documento do artigo em `docs/` e `assets/`.
- **Imagem de apoio:** [Dashboard final comparativo](../assets/dashboard_final_comparativo.png), como síntese visual da defesa.
- **Referencia:** [OSF Registries](https://osf.io/registries/) e [Principios FAIR](https://www.go-fair.org/fair-principles/).
- **Áudio sugerido:** uma banca perguntando “o que exatamente seus dados provam?”.
- **Imagem mental:** ponte com placas de peso: cada afirmacao precisa suportar a evidencia.

## 📊 Elementos de Comunidade e Status

- **Status:** `Defesa preparada` quando a resposta apresentar hipótese, evidência, limite e próximo teste.
- **Debate:** “Reconhecer uma limitacao enfraquece ou fortalece uma tese?”
- **Papel rotativo:** banca, pesquisador, gestor e leitor critico.

## 💡 Engajamento e Conhecimento

- **Atividade:** realizar uma mini-banca de 5 minutos, com réplica baseada em KPI.
- **Produto da aula:** ficha de defesa com H1, H2, evidencia, ressalva e experimento futuro.
- **Conexao profissional:** trocar “prova que” por uma afirmacao proporcional ao desenho experimental.

## Mapa da aula

1. [Subcamada 16.1: O conceito na vida real](#subcamada-161-o-conceito-na-vida-real)
2. [Subcamada 16.2: Desenhando o conceito](#subcamada-162-desenhando-o-conceito)
3. [Subcamada 16.3: Desmistificando a teoria e a notacao formal](#subcamada-163-desmistificando-a-teoria-e-a-notacao-formal)
4. [Subcamada 16.4: Laboratorio ludico no Colab](#subcamada-164-laboratorio-ludico-no-colab)
5. [Subcamada 16.5: O momento serio da nossa aplicacao](#subcamada-165-o-momento-serio-da-nossa-aplicacao)
6. [Subcamada 16.6: Checkpoint de autonomia e fixacao ativa](#subcamada-166-checkpoint-de-autonomia-e-fixacao-ativa)

---

## Subcamada 16.1: O conceito na vida real

### A analogia do diario de bordo entregue ao Almirantado

Considere uma expedicao maritima cientifica designada para cartografar uma rota de navegacao atraves de aguas perigosas e desconhecidas:
- A tripulacao enfrentou tempestades oceanicas de grande escala (o risco de *overfitting* e a dispersao da alta dimensionalidade).
- Navegou entre recifes submersos invisiveis a olho nu (as variaveis de ruido metabolico e colunas redundantes).
- Recorreu a instrumentos opticos e coordenadas astronomicas de precisao (a Teoria dos Jogos do SHAP e a auditoria local do LIME).
- Aliviou a carga util descartando tres quartos do peso morto desnecessario (o Pre-Filtro Estatistico e o shap-select).
- E calibrou a propulsor da embarcacao para a nova massa de deslocamento (a otimizacao com Optuna).

Ao atracar com sucesso no porto de origem, a missao so atinge validade oficial quando o comandante redige o **diario de bordo formal** e o submete a apreciacao do Almirantado:
- Se o diario for redigido com anotacoes imprecisas ou linguagem coloquial, o comando militar descartara o relato como narrativa anedotica.
- Se o diario apresentar registros cronometricos precisos, coordenadas cartograficas reproduziveis e limites claros de navegabilidade, a rota passa a constar dos mapas nauticos oficiais.

O artigo cientifico representa esse diario de bordo formal: ele traduz semanas de computacao e experimentacao em um documento padronizado que permite a qualquer pesquisador independente reproduzir exatamente os mesmos resultados.

### O papel da honestidade metodologica na aceitacao cientifica

Artigos submetidos a periodicos de alto impacto (Qualis A / JCR Q1) frequentemente sofrem rejeicao nao pela ausencia de resultados positivos, mas pela tentativa de camuflar fragilidades:
- Pesquisadores novatos costumam tentar ocultar limitacoes do experimento, temendo que os revisores rejeitem o trabalho.
- Pesquisadores maduros documentam explicitamente as fronteiras da pesquisa: o uso de dados sinteticos controlados, as condicoes sob as quais o modelo logit exige fallback para OLS e as recomendacoes para estudos clinicos futuros com coortes reais.

**A grande sacada:** documentar com clareza as limitacoes metodologicas transforma vulnerabilidades em demonstracoes de maturidade e rigor cientifico.

| Aspecto | Redacao Informal de Divulgacao | Redacao Academica para Periodicos Qualis A |
| :--- | :--- | :--- |
| **Tom textual** | Entusiasta, enfatico, com adjetivos subjetivos | Neutro, analitico, impessoal e estritamente factual |
| **Apresentacao de dados** | Foca apenas nos melhores numeros obtidos | Exibe o quadro comparativo completo (Baseline vs Proposto) |
| **Tratamento de limitacoes** | Omitido para valorizar a solucao | Declarado abertamente com justificativa tecnica de contorno |
| **Defesa de hipoteses** | Afirmacoes genericas de sucesso | Confronto pontual de deltas contra margens $\pm \epsilon$ pre-fixadas |

---

## Subcamada 16.2: Desenhando o conceito

O diagrama abaixo ilustra o roteiro de cinco passos para conversao dos dados experimentais em manuscrito formal:

```text
ROTEIRO DE CONVERSAO EXPERIMENTAL PARA ARTIGO CIENTIFICO
==================================================================================

1. ESTRUTURA BASE (IMRaD)    ---> Abre o gerador oficial do artigo IEEE.
                                  Artigo de ate 3 paginas; relatorio tecnico e complementar.
             │
             ▼
2. EXTRACAO DE DADOS BRUTOS  ---> Executa pipeline_completo.py e coleta metricas
                                  do holdout sintetico (F1, acuracia e tempos).
             │
             ▼
3. INTEGRACAO DE FIGURAS     ---> Insere as 6 figuras de alta resolucao de assets/
                                  (Matriz, ROC, Beeswarm, LIME, Ablacao, Dashboard).
             │
             ▼
4. VERIFICACAO DE HIPOTESES  ---> Confronta os valores da execucao com H1 e H2;
                                  nao os chama de significativos sem teste estatistico.
             │
             ▼
5. DECLARACAO DE LIMITACOES  ---> Documenta as premissas de dados sinteticos
                                  e o fallback OLS na secao de Discussao.
==================================================================================
```

A organizacao tatica de defesa perante arguicoes de revisores e esquematizada no mapa conceitual a seguir:

```text
[ ARGUICAO DO REVISOR ]                   [ RESPOSTA CIENTIFICA EVIDENCIADA ]
"Por que Random Forest e nao               Tabular clinical data: ensembles superam
Deep Learning?"                 ------->  redes neurais em dados tabulares e viabilizam
                                          TreeSHAP exato com tempo polinomial.

"Por que nao usar apenas                   O protocolo compara tres familias e SHAP;
o RFE classico?"                ------->  RFE e o representante wrapper, enquanto
                                          SHAP e LIME oferecem trilhas de auditoria.

"A diferenca de F1 e estatisticamente      Nao foi realizado teste de significancia;
relevante?"                     ------->  ha uma unica particao de holdout sintetico.
                                          A conclusao e descritiva e exploratoria.
```

| Componente de Defesa | Proposito no Exame de Banca | Risco se Despreparado |
| :--- | :--- | :--- |
| **Evidencia de $H_1$** | Comparar atributos e F1 no mesmo holdout | Generalizar a partir de uma unica particao |
| **Evidencia de $H_2$** | Comparar ruido retido e tempo de selecao medido | Tratar um tempo local como custo universal |
| **Artigo IEEE** | Sintetizar pergunta, metodo, resultados e referencias em ate 3 paginas | Exceder o limite ou omitir uma limitacao |

---

## Subcamada 16.3: Desmistificando a teoria e a notacao formal

### A formalizacao dos criterios de avaliacao

A avaliacao de $H_1$ usa os criterios predefinidos abaixo. O resultado de uma particao nao e uma comprovacao universal:

$$\text{Validacao}(H_1) = \mathbf{1}_{\left\{ \left(1 - \frac{p_{\text{enxuto}}}{p_{\text{base}}}\right) \ge 0.50 \right\}} \cdot \mathbf{1}_{\left\{ \Delta F_1 \ge -0.05 \right\}}$$

Para $H_2$, compare a taxa de rejeicao de ruido e o tempo de selecao da proposta com os metodos do benchmark, em especial RFE. Os tempos dependem do hardware e da carga; uma execucao nao demonstra escalabilidade nem significancia estatistica.

### Tratamento formal da quase-separacao e fallback OLS

Na etapa econometrica do `shap-select`, a presenca de atributos SHAP altamente informativos pode provocar o fenomeno da quase-separacao completa (*quasi-complete separation*), no qual a estimativa de maxima verossimilhança da regressao logistica diverge:

$$\lim_{k \to \infty} \hat{\boldsymbol{\beta}}_{\text{Logit}}^{(k)} = \pm \infty, \quad \text{det}(\mathbf{H}) \to 0$$

Quando o ajuste Logit falha nas excecoes previstas, o pipeline informa e tenta OLS como fallback. OLS e Logit nao sao modelos equivalentes; o fallback nao garante estabilidade nem validade dos p-valores se houver colinearidade ou outros problemas:

$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = \left( \boldsymbol{\Phi}^T \boldsymbol{\Phi} \right)^{-1} \boldsymbol{\Phi}^T \mathbf{y}$$

| Simbolo | Significado Formal | Leitura no Projeto |
| :--- | :--- | :--- |
| $\mathbf{1}_{\{A\}}$ | Funcao indicadora do evento logico $A$ | Avaliador binario de confirmacao da meta cientifica |
| $\text{det}(\mathbf{H})$ | Determinante da matriz hessiana de informacao | Pode indicar mau condicionamento; nao e um teste completo de convergencia |
| $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ | Estimador de minimos quadrados | Fallback distinto de Logit, que requer interpretacao cautelosa |
| $p$ | Numero de atributos de entrada | Leia no CSV; shap-select pode reter menos de 10 |
| $F_1$ | Pontuacao harmonica de classificacao | Leia o valor do holdout atual; nao o trate como resultado clinico |

### A ordem correta evita incoerencias de publicacao

Os valores numericos relatados no texto do artigo devem ser extraidos diretamente das execucoes registradas nos logs do `pipeline_completo.py`. Jamais arredonde numeros manualmente de forma heterogenea entre secoes distintas do documento.

---

## Subcamada 16.4: Laboratorio ludico no Colab

Execute o bloco abaixo no Google Colab para gerar uma descricao a partir do CSV atual, sem transformar diferencas em prova de significancia ou aplicabilidade hospitalar:

```python
# =============================================================================
# CAMADA 16: LABORATORIO LUDICO DE REDACAO ACADEMICA
# Demonstracao: Geracao Parametrizada de Paragrafo de Discussao Cientifica
# =============================================================================
import pandas as pd

resultados = pd.read_csv("assets/tabela_comparativa_completa.csv")
base = resultados.loc[resultados["Familia"] == "Referência"].iloc[0]
proposta = resultados.loc[resultados["Metodo"] == "XAI Proposto (shap-select + Optuna)"].iloc[0]
taxa_reducao = (1 - proposta["N_Atributos"] / base["N_Atributos"]) * 100
delta_f1 = proposta["F1_Score"] - base["F1_Score"]
texto_gerado = (
    f"No holdout sintetico desta execucao, a proposta reteve {int(proposta['N_Atributos'])} "
    f"atributos ({taxa_reducao:.1f}% de reducao). O F1 foi {base['F1_Score']:.4f} no baseline "
    f"e {proposta['F1_Score']:.4f} na proposta (delta {delta_f1:+.4f}). "
    "A diferenca e descritiva; uma particao nao demonstra significancia estatistica nem validade clinica."
)

print("PARAGRAFO ACADEMICO FORMATADO (PADRAO QUALIS A):")
print("-" * 76)
print(texto_gerado)
print("-" * 76)
```

> **O que voce deve notar no texto gerado:** os numeros vem do CSV, e as conclusoes ficam restritas ao holdout sintetico observado. Uma unica particao nao sustenta generalizacao clinica ou inferencia de significancia.

**Mini-experimento:** altere o valor de F1 da proposta para que a queda frente ao baseline ultrapasse 0,05. Como a conclusao de H1 deve ser ajustada sem alegar significancia estatistica?

---

## Subcamada 16.5: O momento serio da nossa aplicacao

> Agora aplicamos os conceitos ao benchmark sintetico e aos artefatos experimentais do projeto.

A compilacao do artigo oficial em formato IEEE e realizada pelo script [gerar_artigo_ieee.py](../gerar_artigo_ieee.py). Ele gera o PDF [artigo_cientifico_ieee.pdf](../artigo_cientifico_ieee.pdf), valida o limite de tres paginas e usa os resultados do CSV do pipeline. O arquivo `artigo_xai_reduction.docx` e um rascunho legado e nao e material oficial de entrega.

```bash
# Execute primeiro o pipeline para atualizar os resultados
python pipeline_completo.py
python gerar_artigo_ieee.py
```

```python
import pandas as pd

resultados = pd.read_csv("assets/tabela_comparativa_completa.csv")
proposta = resultados.loc[
    resultados["Metodo"] == "XAI Proposto (shap-select + Optuna)"
].iloc[0]
rfe = resultados.loc[resultados["Metodo"] == "Wrapper (RFE)"].iloc[0]

print(f"Tempo de selecao da proposta: {proposta['Tempo_Selecao_s']:.2f} s")
print(f"Tempo de selecao do RFE: {rfe['Tempo_Selecao_s']:.2f} s")
```

### Tabela oficial de KPIs

> Os valores abaixo sao produzidos pelo codigo, nao devem ser decorados como constantes. Tempo, latencia e ate pequenas variacoes de desempenho dependem do ambiente e da versao das bibliotecas.

| KPI | Como e calculado | Pergunta operacional |
| :--- | :--- | :--- |
| **Criterio H1** | Reducao de atributos >= 50% e delta F1 >= -0,05 no holdout | O resultado desta execucao atende ao criterio predefinido? |
| **Criterio H2** | Rejeicao de ruido e tempo de selecao comparados com os metodos do CSV | A proposta supera os comparadores neste benchmark sintetico? |
| **Fidelidade LIME** | R² do substituto em uma observacao escolhida no holdout | Qual foi o ajuste local, e qual o limite dessa unica observacao? |
| **Conformidade de citacoes** | Conferencia da correspondencia entre citacao e referencia | As fontes sustentam as afirmacoes do artigo? |

### Interpretacao dos resultados e limites de uso

A conclusao da escrita academica coroa o ciclo de desenvolvimento cientifico:

1. **Defesa metodologica:** apresente os numeros observados, o protocolo e a incerteza; uma unica particao sintetica nao estabelece validade clinica.
2. **Trabalho futuro:** dados reais, validacao externa, analise por subgrupos e estudo com especialistas seriam necessarios antes de qualquer proposta de uso em saude.
3. **Comunicacao cientifica:** diferencie resultado medido, hipotese apoiada neste split e implicacao que ainda precisa de estudo.

---

## Subcamada 16.6: Checkpoint de autonomia e fixacao ativa

Explique sem consultar o texto e depois confira sua resposta:

1. **Como a analogia do diario de bordo entregue ao Almirantado sintetiza o papel do artigo cientifico na conclusao de uma pesquisa?**
2. **Qual e a razao pela qual os comites de revisao de periodicos Q1 valorizam a declaracao aberta de limitacoes tecnicas?**
3. **Como voce reporta se os criterios de $H_1$ e $H_2$ foram atendidos sem alegar significancia estatistica?**
4. **Qual e o contra-argumento cientifico exato para a pergunta de por que nao utilizamos Redes Neurais Profundas (Deep Learning) neste experimento?**
5. **Por que o script `pipeline_completo.py` implementa um mecanismo de fallback para OLS durante a regressao multivariada do shap-select?**
6. **Quais sao as quatro dimensoes indispensaveis que devem constar em um paragrafo de discussao cientifica de resultados?**

### Mini-desafio pratico

Preencha a matriz de preparacao para arguicao de banca com os dados de sintese do projeto:

| Pergunta Provocativa da Banca | Evidencia Numerica do Experimento | Conceito Teorico Fundamental |
| :--- | :--- | :--- |
| *"O resultado prova que atributos podem ser descartados em pacientes?"* | Nao; compara metodos em um holdout de dados sinteticos | Validade externa e clinica |
| *"Por que nao adotaram apenas o RFE consagrado?"* | O protocolo compara filter, wrapper, embedded e a proposta SHAP | Benchmark com limite sintetico |
| *"A diferenca de F1 e estatisticamente significativa?"* | Nao foi realizado teste inferencial nem repeticao de splits | Evidencia descritiva e exploratoria |

**Pergunta reflexiva:** o dominio conjunto do codigo, da fundamentacao matematica e da estrutura de comunicacao cientifica confere autonomia plena para conduzir e defender pesquisas avancadas em Inteligencia Artificial?
