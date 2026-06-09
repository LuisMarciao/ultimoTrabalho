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

    p_survive = (
        df[df['Survived'] == 1].shape[0]
        / df.shape[0]
    )

    sexo_texto = (
        "female"
        if sexo == "Feminino"
        else "male"
    )

    p_sexo_given_survive = (
        len(
            df[
                (df['Sex'] == sexo_texto) &
                (df['Survived'] == 1)
            ]
        )
        /
        len(df[df['Survived'] == 1])
    )

    p_sexo = (
        len(df[df['Sex'] == sexo_texto])
        /
        len(df)
    )

    bayes = (
        p_sexo_given_survive
        * p_survive
    ) / p_sexo

    pred_arvore = arvore.predict(entrada)[0]
    pred_floresta = floresta.predict(entrada)[0]

    st.subheader("Resultado")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Bayes",
        f"{bayes*100:.2f}%"
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