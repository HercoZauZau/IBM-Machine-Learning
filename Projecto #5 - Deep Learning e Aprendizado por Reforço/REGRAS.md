# Regras adoptadas — Ntxuva

## 1. Nota sobre variantes

Ntxuva pertence à família Mancala e possui variações locais. Para um sistema de IA é indispensável fixar regras determinísticas e reproduzíveis. Esta implementação usa uma variante computacional documentada abaixo.

## 2. Tabuleiro

- 4 filas × 8 casas (*godi*);
- Jogador 1 controla as duas filas inferiores;
- Jogador 2 controla as duas filas superiores;
- a fila mais próxima do centro é a **fila interna/ataque**;
- a fila exterior é a **fila externa/defesa**;
- cada casa começa com 2 sementes;
- total inicial: 64 sementes.

O código permite alterar o número de colunas, mas toda a configuração padrão, DQN e GUI usam 8.

## 3. Sentido da jogada

A distribuição é feita **no sentido anti-horário**, percorrendo apenas as duas filas pertencentes ao jogador da vez.

## 4. Escolha da casa inicial

### Gula

Se existir pelo menos uma casa com duas ou mais sementes, só essas casas podem iniciar uma jogada.

### Tchonga / singleton

Uma casa com uma única semente só pode ser jogada quando não existe gula. Na formalização usada, a semente única deve cair numa casa vazia.

## 5. Semeadura e relay sowing

1. Retiram-se todas as sementes da casa inicial.
2. Distribui-se uma semente por casa no percurso anti-horário.
3. Se a última semente cair numa casa que já continha sementes, todo o novo conteúdo dessa casa é recolhido e a semeadura continua.
4. A jogada termina quando a última semente cai numa casa que estava vazia.

O motor possui detecção/limite de ciclos para evitar que uma jogada infinita bloqueie uma experiência computacional.

## 6. Captura — Cubá

Uma captura é activada quando:

- a jogada termina numa casa vazia da fila interna do jogador;
- e a casa interna directamente oposta do adversário possui sementes.

Nessa situação são capturadas:

1. as sementes da casa interna oposta;
2. as sementes da casa externa oposta da mesma coluna, se existirem;
3. quando disponíveis, sementes de **duas outras casas do adversário escolhidas pelo jogador**.

A escolha adicional é mantida no modelo computacional como parte da acção. Por isso a acção não é apenas “qual casa jogar”, mas pode ser “qual casa jogar + quais duas casas adicionais capturar”.

## 7. Fim da partida

A partida termina quando o jogador da vez não possui uma jogada válida.

- vence o jogador com mais sementes capturadas;
- se as capturas forem iguais, a implementação declara empate.

## 8. Convenção sobre movimentos cíclicos

Mancala pode produzir relay sowing extremamente longo ou cíclico. O motor detecta repetição do subestado durante uma jogada e exclui esse movimento da lista de acções legais da versão computacional. Existe também um limite de segurança (`max_relay_steps`).

Esta é uma **convenção computacional**, não uma alegação de regra cultural universal.

## 9. Fontes e fidelidade

A referência académica central é Ali, Gimo & Saide (2020), que trata especificamente da variante moçambicana e de um agente MiniMax. A descrição operacional foi confrontada com a página pública `ntxuva.org`, que descreve 4 filas, duas sementes, gula, tchonga, semeadura anti-horária, relay sowing e Cubá.

Como as variantes locais diferem e o texto integral do artigo não é redistribuído com este projecto, esta implementação deve ser entendida como **variante computacional moçambicana documentada**, e não como afirmação de que todas as comunidades de Moçambique usam exactamente estas regras.
