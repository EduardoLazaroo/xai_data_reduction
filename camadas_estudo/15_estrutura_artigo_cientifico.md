# Camada 15: A Estrutura de um Artigo Científico — Da Teoria ao Padrão Acadêmico

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 15  
**Contexto Técnico:** [gerar_artigo_word.py](file:///c:/Users/eduar/projetos/xai_data_reduction/gerar_artigo_word.py) e [docs/artigo_xai_reduction.docx](file:///c:/Users/eduar/projetos/xai_data_reduction/docs/artigo_xai_reduction.docx)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> Compreender a arquitetura canônica de um **Artigo Científico em Computação e Inteligência Artificial**. Entender o propósito de cada seção, dominar a diferença crucial entre **"Resultados"** (a descrição objetiva dos dados) e **"Discussão"** (o significado científico e o porquê dos fenômenos), e aprender a conectar código e escrita acadêmica de alto impacto.

---

## 1. O que Estudar em Profundidade?

### 1.1 O Mapa Estrutural do Artigo Científico
O artigo gerado pelo script [gerar_artigo_word.py](file:///c:/Users/eduar/projetos/xai_data_reduction/gerar_artigo_word.py) segue a estrutura internacional IMRaD (*Introduction, Methods, Results, and Discussion*), expandida em 14 seções lógicas:

```
Estrutura Canônica do Artigo
 ├── 1. Introdução & Contextualização
 ├── 2. Problema de Pesquisa & Justificativa
 ├── 3. Objetivos (Geral e Específicos)
 ├── 4. Hipóteses Científicas (H1 e H2)
 ├── 5. Fundamentação Teórica (ML, XAI, SHAP, LIME, Optuna)
 ├── 6. Trabalhos Relacionados
 ├── 7. Metodologia Experimental (Dataset Sintético, Pipeline de 6 Etapas)
 ├── 8. Resultados Numéricos e Gráficos
 ├── 9. Discussão Crítica (O Porquê dos Números)
 ├── 10. Implicações Práticas (Clínicas e Computacionais)
 ├── 11. Limitações do Estudo
 ├── 12. Trabalhos Futuros
 ├── 13. Conclusão
 └── 14. Referências Bibliográficas
```

---

### 1.2 A Diferença Crucial: Resultados vs. Discussão
Um dos erros mais comuns de autores iniciantes é misturar a apresentação de números com a sua interpretação.

| Seção | O que Deve Conter? | Exemplo no Nosso Projeto | O que NÃO Deve Conter? |
| :--- | :--- | :--- | :--- |
| **Resultados** (*Results*) | Relato objetivo, neutro e puramente quantitativo do que foi observado nas tabelas e gráficos. Sem especulações ou opiniões. | *"O modelo Baseline atingiu F1 de 0.8410 com 40 atributos. O modelo reduzido obteve F1 de 0.8600 com 10 atributos e tempo de treino de 110 ms."* | Não tente explicar *por que* o número subiu. Apenas mostre o fato. |
| **Discussão** (*Discussion*) | Interpretação crítica, causal e teórica dos fatos. Conecta os números com as hipóteses e com a literatura científica existente. | *"O aumento de 1.9% no F1-Score do modelo reduzido pode ser atribuído à eliminação dos 20 atributos de ruído metabólico, o que reduziu a variância e mitigou o overfitting..."* | Não repita tabelas inteiras de números brutos sem interpretá-los. |

---

### 1.3 As Duas Hipóteses Centrais do Nosso Estudo
Todo artigo de computação científica é construído para validar ou refutar **Hipóteses Formais**:
- **Hipótese 1 ($H_1$ — Manutenção Preditiva):** É possível eliminar pelo menos 50% dos atributos de um dataset clínico de alta dimensão usando explicabilidade XAI, mantendo o $F_1$-score dentro de uma margem de tolerância clínica de $\pm 0.05$ em relação ao Baseline.
- **Hipótese 2 ($H_2$ — Eficiência Computacional):** O modelo treinado no subconjunto reduzido apresentará uma redução de pelo menos 40% no tempo de treinamento e na latência de inferência em relação ao modelo com todos os atributos.

---

## 2. Por que isso Importa para o Projeto?

Sem essa estrutura canônica, o seu código seria apenas um "experimento solto no computador".  
O padrão de artigo científico é o que transforma o seu trabalho prático em um **ativo acadêmico e profissional publicável** em congressos, periódicos ou como trabalho de conclusão de curso e pós-graduação.

---

## 3. Onde Aparece no Código do Projeto?

No arquivo [gerar_artigo_word.py](file:///c:/Users/eduar/projetos/xai_data_reduction/gerar_artigo_word.py):
- O script automatiza a criação do documento Word (`artigo_xai_reduction.docx`), criando os parágrafos justificados, cabeçalhos hierárquicos e a formatação exata dessas 14 seções estruturais.

---

## 4. Checkpoint de Autonomia do Estudante

Responda antes de passar para a Camada 16:

> [!IMPORTANT]
> 🧠 **Pergunta do Checkpoint:**  
> **Você consegue explicar, sem olhar para as anotações, qual é a diferença entre a seção de "Resultados" e a seção de "Discussão" de um artigo científico, e dar um exemplo prático de uma frase que pertence aos Resultados e uma frase que pertence à Discussão?**
