# IBM Machine Learning Engineering

<img src="assets/coursera.png">

</br>

Repositório que reúne os projectos desenvolvidos ao longo do curso [**IBM Machine Learning Engineering Professional Certificate**](https://www.coursera.org/professional-certificates/ibm-machine-learning).

Os projectos abrangem diferentes problemas de investigação, incluindo análise económica, previsão, classificação, sistemas de recomendação e aprendizagem por reforço, seguindo boas práticas de preparação de dados, desenvolvimento de modelos e avaliação de desempenho.

**Nota:** Em todos os projectos deste repositório foram usados **apenas dados de Moçambique**, com o objectivo de desenvolver soluções e análises contextualizadas à realidade nacional. Grande parte dos datasets utilizados foi obtida através do projecto [**Kutiva**](https://github.com/HercoZauZau/Kutiva), complementados por outras fontes públicas.

---

## 📌 ToDo

* [x] **Projecto #1** - Análise Exploratória de Dados
* [x] **Projecto #2** - Machine Learning Supervisionado (Regressão)
* [x] **Projecto #3** - Machine Learning Supervisionado (Classificação)
* [ ] **Projecto #4** - Machine Learning Não Supervisionado
* [ ] **Projecto #5** - Deep Learning e Aprendizado por Reforço
* [x] **Projecto #6** - Sistemas de Recomendação

</br>

* [ ] **Artigo #1** - Impacto da Variação dos Preços dos Combustíveis no Preço de Alimentos em Moçambique
* [ ] **Artigo #2** - Análise de Mudanças no Regime Climático Distrital em Moçambique
* [ ] **Artigo #3** - Classificação Automática de Notícias Moçambicanas
* [ ] **Artigo #4** - *por definir*
* [ ] **Artigo #5** - *por definir*
* [ ] **Artigo #6** - Sistema de Recomendação de Imóveis para a Cidade de Maputo

</br>

* [ ] **Diminuir Café** 

---

### #1 - Impacto da Variação dos Preços dos Combustíveis no Preço de Alimentos em Moçambique 🌽 ⛽

**Objectivos:**

* Analisar a relação entre os preços dos combustíveis e do milho.
* Identificar tendências, sazonalidade e padrões temporais.

**Técnicas Utilizadas:**

* Recolha, limpeza e transformação de dados.
* Análise Exploratória de Dados (EDA).
* Testes e avaliação de hipóteses estatísticas.

**Conclusão:**

* Foi identificada uma relação relevante entre os preços dos combustíveis e do milho.
* O impacto não ocorre de forma imediata nem uniforme em todas as províncias.
* Os custos logísticos contribuem para o aumento dos preços dos alimentos.

</br>

### #2 - Detecção de Mudanças no Regime Climático Distrital em Moçambique 🌧️ 🌍

**Objectivos:**

* Analisar a evolução da precipitação distrital em Moçambique entre 1981 e 2025.
* Identificar tendências climáticas e mudanças de regime nas séries temporais de precipitação.
* Detectar quando ocorreram alterações significativas nos padrões históricos de chuva.

**Técnicas Utilizadas:**

* Recolha, limpeza, transformação e integração de dados climáticos e geográficos.
* Análise Exploratória de Dados (EDA).
* Regressão Linear e Regressão Huber.
* Teste de Mann-Kendall para análise de tendências.
* Teste de Pettitt para detecção de pontos de mudança (*Change-Point Detection*).

**Conclusão:**

* A precipitação em Moçambique apresenta forte sazonalidade, com maior concentração de chuvas entre Novembro e Março.
* Não foram encontradas evidências de mudanças significativas no regime pluviométrico nacional quando analisado de forma agregada.
* Foram identificadas pequenas mudanças de regime climático localizadas nos distritos de Ibo, Palma e Mossurize.
* Os resultados mostram que as alterações climáticas relacionadas à precipitação ocorrem de forma heterogénea, afectando algumas regiões mais do que outras.

</br>

### #3 - Classificação Automática de Notícias de Moçambique 📰 🤖

**Objectivos:**

* Desenvolver um modelo capaz de classificar automaticamente notícias moçambicanas por categoria.
* Reduzir o esforço manual na organização e indexação de conteúdos jornalísticos.
* Avaliar o desempenho de diferentes algoritmos de Machine Learning na tarefa de classificação de texto.
* Identificar a abordagem mais adequada para aplicações de categorização automática de notícias em língua portuguesa.

**Técnicas Utilizadas:**

* Recolha, limpeza e pré-processamento de dados textuais.
* Normalização e tratamento de textos.
* Transformação textual utilizando **TF-IDF (Term Frequency–Inverse Document Frequency)**.
* Análise Exploratória de Dados (EDA).
* Treino e avaliação de modelos de classificação supervisionada.

**Conclusão:**

* Os modelos de Machine Learning demonstraram capacidade para classificar automaticamente notícias moçambicanas com elevado nível de precisão.
* A representação textual baseada em **TF-IDF** mostrou-se eficaz na captura dos termos mais relevantes para distinguir diferentes categorias de notícias.

</br>

### #6 - Sistema de Recomendação de Imóveis para a Cidade de Maputo 🏠 📍

**Objectivos:**

* Desenvolver um sistema capaz de recomendar imóveis semelhantes com base nas suas características.
* Facilitar a procura de imóveis relevantes para potenciais compradores e arrendatários.
* Explorar técnicas de agrupamento e recomendação aplicadas ao mercado imobiliário.
* Comparar diferentes abordagens de recomendação para identificar a solução mais adequada.

**Técnicas Utilizadas:**

* Recolha, limpeza e preparação de dados imobiliários.
* Análise Exploratória de Dados (EDA).
* Engenharia de atributos (*Feature Engineering*).
* Normalização e codificação de variáveis categóricas.
* Processamento de texto utilizando **TF-IDF (Term Frequency–Inverse Document Frequency)**.
* Agrupamento de imóveis utilizando **K-Means Clustering**.
* Identificação de imóveis semelhantes através de **Nearest Neighbors**.
* Sistema de recomendação baseado em conteúdo (**Content-Based Filtering**).
* Avaliação através de métricas como Precision@K, Recall@K, F1-Score, nDCG e Cobertura.

**Conclusão:**

* O sistema demonstrou capacidade para recomendar imóveis com características semelhantes aos imóveis de referência.
* A utilização conjunta de atributos estruturados e descrições textuais permitiu capturar melhor as características dos imóveis.
* O modelo baseado em conteúdo apresentou resultados superiores ao baseline aleatório, produzindo recomendações mais relevantes.
* Os resultados demonstram o potencial dos sistemas de recomendação como ferramenta de apoio à procura de imóveis no contexto do mercado imobiliário moçambicano.

---
