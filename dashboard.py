import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Projeto Titanic",
    page_icon="🚢",
    layout="wide"
)

# ==========================
# CARREGAMENTO DOS DADOS
# ==========================

@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df.drop('Cabin', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)

    IQR = Q3 - Q1

    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    df = df[
        (df['Fare'] >= lim_inf) &
        (df['Fare'] <= lim_sup)
    ]

    return df

df = carregar_dados()

# ==========================
# SIDEBAR
# ==========================

pagina = st.sidebar.radio(
    "Navegação",
    [
        "Análise dos Dados",
        "Classificação Probabilística"
    ]
)

# ==========================
# EDA
# ==========================

if pagina == "Análise dos Dados":

    st.title("🚢 Dashboard Titanic")

    st.header("Informações Gerais")

    col1, col2, col3 = st.columns(3)

    col1.metric("Registros", len(df))
    col2.metric("Variáveis", len(df.columns))
    col3.metric("Taxa de Sobrevivência",
                f"{df['Survived'].mean()*100:.2f}%")

    st.divider()

    st.subheader("Distribuição de Idade")

    fig, ax = plt.subplots()

    sns.histplot(
        df['Age'],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("Distribuição das Idades")

    st.pyplot(fig)

    st.divider()

    st.subheader("Sobreviventes por Sexo")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x='Sex',
        hue='Survived',
        ax=ax
    )

    ax.set_title("Sexo x Sobrevivência")

    st.pyplot(fig)

    st.divider()

    st.subheader("Sobreviventes por Classe")

    fig, ax = plt.subplots()

    sns.countplot(
        data=df,
        x='Pclass',
        hue='Survived',
        ax=ax
    )

    ax.set_title("Classe x Sobrevivência")

    st.pyplot(fig)

    st.divider()

    st.subheader("Mapa de Correlação")

    corr = df.select_dtypes(include=np.number).corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# ==========================
# CLASSIFICAÇÃO
# ==========================

if pagina == "Classificação Probabilística":

    st.title("🎯 Classificação Probabilística")

    df_ml = df.copy()

    df_ml['Sex'] = df_ml['Sex'].map({
        'male': 0,
        'female': 1
    })

    df_ml['Embarked'] = df_ml['Embarked'].map({
        'S': 0,
        'C': 1,
        'Q': 2
    })

    df_ml = df_ml.drop(
        ['PassengerId', 'Name', 'Ticket'],
        axis=1
    )

    X = df_ml.drop('Survived', axis=1)
    y = df_ml['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    arvore = DecisionTreeClassifier(
        random_state=42
    )

    arvore.fit(X_train, y_train)

    floresta = RandomForestClassifier(
        random_state=42
    )

    floresta.fit(X_train, y_train)

    st.sidebar.header("Dados do Passageiro")

    sexo = st.sidebar.selectbox(
        "Sexo",
        ["Masculino", "Feminino"]
    )

    idade = st.sidebar.slider(
        "Idade",
        0,
        80,
        30
    )

    classe = st.sidebar.selectbox(
        "Classe",
        [1, 2, 3]
    )

    tarifa = st.sidebar.slider(
        "Tarifa",
        0.0,
        300.0,
        50.0
    )

    sibsp = st.sidebar.slider(
        "Irmãos/Cônjuge",
        0,
        8,
        0
    )

    parch = st.sidebar.slider(
        "Pais/Filhos",
        0,
        6,
        0
    )

    embarque = st.sidebar.selectbox(
        "Porto",
        ["S", "C", "Q"]
    )

    sexo_num = 1 if sexo == "Feminino" else 0

    porto_num = {
        "S": 0,
        "C": 1,
        "Q": 2
    }[embarque]

    entrada = pd.DataFrame([{
        'Pclass': classe,
        'Sex': sexo_num,
        'Age': idade,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': tarifa,
        'Embarked': porto_num
    }])

    # =====================
    # BAYES
    # =====================

# ====================================
# BAYES COMPLETO
# ====================================

    df_bayes = df.copy()

    # Criando faixas de idade
    df_bayes["FaixaIdade"] = pd.cut(
        df_bayes["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=[
            "Crianca",
            "Adolescente",
            "Adulto",
            "Meia_Idade",
            "Idoso"
        ]
    )

    # Determinar faixa escolhida

    if idade <= 12:
        faixa = "Crianca"
    elif idade <= 18:
        faixa = "Adolescente"
    elif idade <= 35:
        faixa = "Adulto"
    elif idade <= 60:
        faixa = "Meia_Idade"
    else:
        faixa = "Idoso"

    sexo_texto = (
        "female"
        if sexo == "Feminino"
        else "male"
    )

    # Priori

    p_survive = (
        len(df_bayes[df_bayes["Survived"] == 1])
        /
        len(df_bayes)
    )

    p_not_survive = (
        len(df_bayes[df_bayes["Survived"] == 0])
        /
        len(df_bayes)
    )

    # Verossimilhanças

    p_sexo_survive = (
        len(
            df_bayes[
                (df_bayes["Sex"] == sexo_texto)
                &
                (df_bayes["Survived"] == 1)
            ]
        )
        /
        len(df_bayes[df_bayes["Survived"] == 1])
    )

    p_classe_survive = (
        len(
            df_bayes[
                (df_bayes["Pclass"] == classe)
                &
                (df_bayes["Survived"] == 1)
            ]
        )
        /
        len(df_bayes[df_bayes["Survived"] == 1])
    )

    p_faixa_survive = (
        len(
            df_bayes[
                (df_bayes["FaixaIdade"] == faixa)
                &
                (df_bayes["Survived"] == 1)
            ]
        )
        /
        len(df_bayes[df_bayes["Survived"] == 1])
    )

    # Não sobreviveu

    p_sexo_not = (
        len(
            df_bayes[
                (df_bayes["Sex"] == sexo_texto)
                &
                (df_bayes["Survived"] == 0)
            ]
        )
        /
        len(df_bayes[df_bayes["Survived"] == 0])
    )

    p_classe_not = (
        len(
            df_bayes[
                (df_bayes["Pclass"] == classe)
                &
                (df_bayes["Survived"] == 0)
            ]
        )
        /
        len(df_bayes[df_bayes["Survived"] == 0])
    )

    p_faixa_not = (
        len(
            df_bayes[
                (df_bayes["FaixaIdade"] == faixa)
                &
                (df_bayes["Survived"] == 0)
            ]
        )
        /
        len(df_bayes[df_bayes["Survived"] == 0])
    )

    # Naive Bayes

    score_survive = (
        p_survive
        * p_sexo_survive
        * p_classe_survive
        * p_faixa_survive
    )

    score_not = (
        p_not_survive
        * p_sexo_not
        * p_classe_not
        * p_faixa_not
    )

    total = score_survive + score_not

    prob_survive = (
        score_survive / total
    ) * 100

    prob_not = (
        score_not / total
    ) * 100

    pred_arvore = arvore.predict(entrada)[0]
    pred_floresta = floresta.predict(entrada)[0]

    st.subheader("Resultado")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Bayes",
        f"{prob_survive:.2f}%"
    )

    col2.metric(
        "Árvore",
        "Sobreviveu"
        if pred_arvore == 1
        else "Não Sobreviveu"
    )

    col3.metric(
        "Random Forest",
        "Sobreviveu"
        if pred_floresta == 1
        else "Não Sobreviveu"
    )

    # ==========================
    # GRÁFICO DO BAYES
    # ==========================

    st.divider()

    st.subheader("Probabilidades Bayesianas")

    grafico = pd.DataFrame({
        "Classe": [
            "Sobreviveu",
            "Não Sobreviveu"
        ],
        "Probabilidade": [
            prob_survive,
            prob_not
        ]
    })

    st.bar_chart(
        grafico.set_index("Classe")
    )

    # ==========================
    # ACURÁCIA
    # ==========================

    st.divider()

    st.subheader("Acurácia dos Modelos")

    pred_tree = arvore.predict(X_test)
    pred_forest = floresta.predict(X_test)

    acc_tree = accuracy_score(
        y_test,
        pred_tree
    )

    acc_forest = accuracy_score(
        y_test,
        pred_forest
    )

    st.write(
        f"Árvore de Decisão: {acc_tree:.4f}"
    )

    st.write(
        f"Random Forest: {acc_forest:.4f}"
    )