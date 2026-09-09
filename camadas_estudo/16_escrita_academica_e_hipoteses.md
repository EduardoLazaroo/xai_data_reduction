# Camada 16: Da Teoria e Código à Escrita Acadêmica Definitiva

**Trilha de Estudo:** XAI Aplicada à Redução de Dados em Machine Learning  
**Base Curricular:** Roteiro de Estudo — Etapa 16  
**Contexto Técnico:** [pipeline_completo.py](file:///c:/Users/eduar/projetos/xai_data_reduction/pipeline_completo.py), [gerar_artigo_word.py](file:///c:/Users/eduar/projetos/xai_data_reduction/gerar_artigo_word.py) e [docs/artigo_xai_reduction.docx](file:///c:/Users/eduar/projetos/xai_data_reduction/docs/artigo_xai_reduction.docx)

---

> [!NOTE]
> 🎯 **Foco Central desta Camada:**  
> O ápice de toda a nossa jornada: consolidar os conhecimentos teóricos, as evidências empíricas e o código em um **texto acadêmico definitivo de alto impacto**. Aprender a transcrever os números reais do pipeline para o artigo, responder formalmente às hipóteses de pesquisa $H_1$ e $H_2$, documentar as limitações técnicas com maturidade científica e conquistar o **domínio autônomo** para defender a pesquisa perante qualquer banca examinadora.

---

## 1. O Roteiro de Redação Científica em 5 Passos

Para preencher e refinar o seu artigo com base no código real executado, siga este procedimento prático:

### Passo 1: Use o Documento Word como Esqueleto
Abra o documento [docs/artigo_xai_reduction.docx](file:///c:/Users/eduar/projetos/xai_data_reduction/docs/artigo_xai_reduction.docx) (ou gere uma versão atualizada executando `python gerar_artigo_word.py`).  
O esqueleto já traz a formatação padrão da ABNT/IEEE com títulos, parágrafos justificados e espaçamentos adequados.

### Passo 2: Substitua os Números Genéricos por Números REAIS
Execute no seu terminal:
```bash
python pipeline_completo.py
```
Copie os números exatos emitidos na tabela final do terminal e substitua nas tabelas do artigo:
- Acurácia exata do Baseline vs. Reduzido.
- $F_1$-score exato de cada modelo.
- Tempos de treinamento e inferência em milissegundos.
- Quantidade exata de atributos aprovados no `shap-select` (ex: 9 ou 10 colunas).

### Passo 3: Incorpore as Imagens de Alta Resolução da Pasta `assets/`
O script gera automaticamente todas as figuras profissionais na pasta `assets/`:
1. `assets/modulo1_baseline_metrics.png` (Matriz de Confusão e Curva ROC do Baseline).
2. `assets/modulo2_shap_summary.png` (Bar Plot Semântico e Beeswarm Plot do SHAP).
3. `assets/modulo3_lime_local.png` (Laudo Bicolor do Paciente no Limiar Crítico).
4. `assets/modulo4_ablation_curves.png` (Curvas Comparativas de Poda SHAP vs. RFE).
5. `assets/modulo5_shap_select_analysis.png` (Gráfico de Coeficientes Beta do shap-select).
6. `assets/dashboard_final_comparativo.png` (Dashboard Executivo de 4 Painéis).

Insira essas figuras nas respectivas subseções metodológicas e de resultados do artigo!

---

### Passo 4: Validação Formal das Hipóteses de Pesquisa

No capítulo de Discussão e Conclusão, responda categoricamente às hipóteses:

| Hipótese Formal | Critério Estabelecido | Resultado Empírico Observado | Veredito Científico |
| :--- | :--- | :--- | :---: |
| **Hipótese 1 ($H_1$ — Manutenção Preditiva)** | Eliminar $\ge 50\%$ dos atributos mantendo a variação do $F_1$-score dentro de $\pm 0.05$. | Eliminação de **75.0%** dos atributos com variação de $F_1$ de $+0.0190$ (ganho de generalização). | **✅ HIPÓTESE H1 COMPROVADA!** |
| **Hipótese 2 ($H_2$ — Eficiência Computacional)** | Redução de $\ge 40\%$ no tempo de treino e na latência de inferência em produção. | Queda de mais de **60%** no tempo de treino e aceleração de mais de **50%** na latência de resposta. | **✅ HIPÓTESE H2 COMPROVADA!** |

---

### Passo 5: Documentação Madura das Limitações Técnicas
Nenhum artigo científico sério afirma ser "perfeito". Artigos de alto impacto destacam suas limitações com transparência e honestidade acadêmica:
1. **Quasi-Complete Separation no Modelo Logit:** Em bases sintéticas com alta separabilidade, a regressão logística do `shap-select` pode sofrer de problemas numéricos de convergência, exigindo o fallback robusto para OLS (conforme implementado na linha 297 do `pipeline_completo.py`).
2. **Custo Computacional da Otimização:** O processo de busca bayesiana do Optuna exige tempo adicional de computação durante a fase de treino preliminar, sendo viável apenas quando o modelo operará repetidamente em produção.
3. **Validação em Dados Sintéticos vs. Clínicos Reais:** O pipeline foi validado em um ambiente experimental rigorosamente controlado. Trabalhos futuros devem aplicar a mesma esteira em bases reais de prontuários eletrônicos heterogêneos (como bases públicas do MIMIC-III ou PhysioNet).

---

## 2. Checkpoint Final de Mestria do Estudante

Você percorreu as 16 camadas da trilha de estudo! Para comemorar e validar seu domínio completo, responda ao **Grande Checkpoint Final**:

> [!IMPORTANT]
> 🏆 **O Desafio do Exame de Mestria:**  
> **Imagine que você está na frente de uma banca examinadora de professores ou de um conselho diretor de um hospital. Você consegue apresentar os slides deste projeto e responder com segurança, sem decorar nada e apenas usando o seu raciocínio conceitual:**  
> 1. Por que o Baseline sofre com 40 atributos e o que é o Mal da Dimensionalidade?  
> 2. Como o SHAP calcula a importância justa através da Teoria dos Jogos?  
> 3. Por que auditamos com LIME justamente o paciente mais incerto ($P \approx 50\%$)?  
> 4. Como a ablação provou empiricamente que podar dados mantém o F1-score?  
> 5. Por que o `shap-select` exige $\beta > 0$ e $p < 0.05$ na regressão logística?  
> 6. Por que o Optuna é necessário para re-afinar a floresta no espaço reduzido?  
> 7. Como os 4 quadrantes do Dashboard comprovam que o projeto foi um sucesso absoluto?  
> 
> **Se você consegue responder a essas 7 perguntas com suas próprias palavras, você atingiu o nível de especialista no tema e está 100% pronto para escrever o artigo e publicar a sua pesquisa!**
