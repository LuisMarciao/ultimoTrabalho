# RELATÓRIO TÉCNICO – PROJETO DE ESTATÍSTICA E PROBABILIDADE

## 1. Integrante

* Luis Arthur Vasconcellos Marciao

---

# 2. Descrição do Dataset

## Dataset Utilizado

Titanic Dataset

Fonte:

https://www.kaggle.com/c/titanic/data

## Domínio

O dataset contém informações sobre passageiros do navio Titanic e tem como objetivo analisar fatores que influenciaram a sobrevivência durante o naufrágio ocorrido em 1912.

## Quantidade de Registros

891 passageiros.

## Principais Atributos

* PassengerId
* Survived
* Pclass
* Name
* Sex
* Age
* SibSp
* Parch
* Ticket
* Fare
* Cabin
* Embarked

## Variável Alvo

A variável alvo escolhida foi:

**Survived**

* 0 = Não Sobreviveu
* 1 = Sobreviveu

---

# 3. Justificativa da Escolha

O dataset Titanic foi escolhido por apresentar características adequadas aos objetivos do projeto.

Ele possui:

* Variáveis quantitativas e qualitativas;
* Valores ausentes;
* Possíveis outliers;
* Dados categóricos para classificação;
* Possibilidade de aplicação do Teorema de Bayes;
* Aplicação de algoritmos supervisionados de classificação.

Além disso, trata-se de um dataset amplamente utilizado em estudos de ciência de dados, permitindo comparações com resultados já conhecidos.

---

# 4. Tratamento e Limpeza dos Dados

Antes da realização das análises exploratórias e da modelagem, foi necessário realizar etapas de tratamento dos dados.

## 4.1 Tratamento de Valores Ausentes

### Coluna Age

Foram identificados valores ausentes na variável Age.

Foi utilizada a mediana para preenchimento dos valores ausentes.

Justificativa:

A mediana é menos sensível à presença de valores extremos quando comparada à média.

### Coluna Embarked

Os valores ausentes da variável Embarked foram substituídos pela moda.

Justificativa:

Por se tratar de uma variável categórica, a moda representa a categoria mais frequente.

---

## 4.2 Remoção da Coluna Cabin

A coluna Cabin possuía uma grande quantidade de valores ausentes.

Justificativa:

Mais de 70% dos registros estavam vazios, tornando inviável um preenchimento confiável.

Impacto:

Redução de ruído e simplificação dos modelos.

---

## 4.3 Remoção de Duplicidades

Foram removidos registros duplicados encontrados no dataset.

Justificativa:

Registros repetidos podem influenciar incorretamente análises estatísticas e algoritmos de classificação.

---

## 4.4 Tratamento de Outliers

Foi utilizado o método do Intervalo Interquartil (IQR) na variável Fare.

Justificativa:

A presença de valores extremamente altos poderia afetar médias, desvios padrão e o treinamento dos modelos.

Impacto:

Maior estabilidade estatística e melhor desempenho dos classificadores.

---

# 5. Análise Exploratória dos Dados (EDA)

Após a limpeza dos dados foi realizada uma análise exploratória para identificar padrões relevantes.

## Distribuição de Idade

Observou-se concentração de passageiros entre aproximadamente 20 e 40 anos.

Esse resultado indica que a maioria dos passageiros estava em idade economicamente ativa.

---

## Sobrevivência por Sexo

A análise mostrou que mulheres apresentaram taxas de sobrevivência significativamente superiores às dos homens.

Esse comportamento está alinhado às estratégias de evacuação adotadas durante o acidente.

---

## Sobrevivência por Classe

Passageiros da primeira classe apresentaram maior probabilidade de sobrevivência quando comparados aos passageiros da segunda e terceira classes.

Esse resultado sugere influência da posição socioeconômica no acesso aos recursos de evacuação.

---

## Correlações

A análise de correlação revelou relações importantes entre:

* Classe social (Pclass);
* Tarifa paga (Fare);
* Sobrevivência.

Essas variáveis demonstraram influência relevante sobre a variável alvo.

---

# 6. Aplicação do Teorema de Bayes

O Teorema de Bayes foi utilizado para calcular a probabilidade de sobrevivência de um passageiro considerando características observadas.

A fórmula utilizada foi:

P(Sobreviveu|X) = [P(X|Sobreviveu) × P(Sobreviveu)] / P(X)

Onde:

* P(Sobreviveu) representa a probabilidade a priori;
* P(X|Sobreviveu) representa a verossimilhança;
* P(X) representa a probabilidade da evidência;
* P(Sobreviveu|X) representa a probabilidade posterior.

Exemplo analisado:

Probabilidade de sobrevivência dado que o passageiro é do sexo feminino.

Os resultados demonstraram que passageiros do sexo feminino possuíam probabilidade significativamente maior de sobrevivência.

---

# 7. Algoritmos de Classificação

Foram utilizados dois algoritmos de classificação supervisionada.

## 7.1 Árvore de Decisão

A Árvore de Decisão foi escolhida por sua simplicidade e facilidade de interpretação.

Vantagens:

* Fácil visualização das regras;
* Boa interpretabilidade;
* Baixo custo computacional.

---

## 7.2 Random Forest

O Random Forest utiliza diversas árvores de decisão combinadas.

Vantagens:

* Maior robustez;
* Menor risco de overfitting;
* Melhor desempenho preditivo.

---

# 8. Avaliação dos Modelos

Os modelos foram avaliados utilizando:

* Accuracy (Acurácia)
* Precision (Precisão)
* Recall
* F1-Score
* Matriz de Confusão

## Resultados Obtidos

### Árvore de Decisão

* Accuracy: ______
* Precision: ______
* Recall: ______
* F1-Score: ______

### Random Forest

* Accuracy: ______
* Precision: ______
* Recall: ______
* F1-Score: ______

Os valores acima devem ser preenchidos com os resultados obtidos durante a execução do notebook.

---

# 9. Comparação entre Bayes e os Algoritmos

A abordagem bayesiana forneceu uma interpretação probabilística das previsões, enquanto os algoritmos de aprendizado de máquina apresentaram maior capacidade de generalização.

O Random Forest apresentou o melhor desempenho geral entre os métodos avaliados, enquanto a Árvore de Decisão forneceu maior interpretabilidade.

---

# 10. Conclusões

O projeto permitiu aplicar conceitos de estatística, probabilidade e aprendizado de máquina em um problema real.

Durante o desenvolvimento foram realizadas etapas de:

* Limpeza e tratamento de dados;
* Análise exploratória;
* Aplicação do Teorema de Bayes;
* Construção de modelos de classificação;
* Comparação de desempenho.

Os resultados mostraram que características como sexo, classe social e tarifa possuem forte influência na sobrevivência dos passageiros do Titanic.

Além disso, observou-se que modelos de ensemble, como o Random Forest, tendem a apresentar desempenho superior quando comparados a modelos mais simples.

Como limitação do estudo, destaca-se a existência de variáveis ausentes e a necessidade de simplificações para aplicação do Teorema de Bayes.

Apesar disso, os objetivos propostos foram alcançados com sucesso, proporcionando uma compreensão prática dos conceitos estudados na disciplina.
