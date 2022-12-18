import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
data = st.session_state['dat']
st.set_page_config(
    page_title="Porovanie modelov",
    page_icon="📈",
    layout="wide",
)

st.sidebar.header("Porovanie modelov")
progress_bar = st.sidebar.progress(75)
with st.sidebar.form("my_form"):
   options = st.multiselect(
    "Vysvetľujúce premenné",
    data.columns.tolist()[:-1],
    data.columns.tolist()[:-1])

   # Every form must have a submit button.
   submitted = st.form_submit_button("Trénovať")
   if submitted:
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(random_state=125)
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test=train_test_split(data[options],data["y"],test_size=0.3, random_state=125)
        rfc.fit(X_train,y_train)
        st.session_state['model'] = rfc
        y_pred = st.session_state['model'].predict(X_test)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)
        st.write("Presnosť modelu", acc)
        st.session_state['prem'] = options


st.write("# 📈 Porovnanie výkonnosti modelov")

tab_1, tab_2 = st.tabs(["Výber najlepšieho modelu","Výkonnosť natrénovaného modelu"])

with tab_1:
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Presnosť modelov pred optimalizáciou")
        df_modely = pd.DataFrame({'Accuracy': [0.908596, 0.901816,0.900812,0.887922,0.878965,0.869591]},index = ["Random Forest","Logistic Regression","K Nearest Neighbor","Support Vector Machine","Decision Tree","Naive Bayes"])
        st.dataframe(df_modely,use_container_width=True)
    with col2:
        st.write("### Presnosť modelov po optimalizácii")
        df_optmodely = pd.DataFrame({'Accuracy': [0.9083451912614046, 0.9001422951368544]},index = ["Random Forest","K Nearest Neighbor"])
        st.dataframe(df_optmodely,use_container_width=True)
    with st.expander("Vysvetlenie"):
        st.markdown("""
            Z výsledkov možno vidieť, že najpresnejší odhad nám ponúka algoritmus **random forest**. Nakoľko je možné presnosť zlepšiť optimalizáciou hyperparametrov, rozhodli sme sa tak urobiť. Avšak nepodarilo sa nám získať lepšiu presnosť ako v prípade využiťa predvolených parametrov. Na tvorbu predikcie sme sa teda rozhodli použiť akgoritmus **random forest** s prednastavenými parametrami.
        """)
with tab_2:
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        fig = px.imshow(confusion_matrix(y_test,y_pred),text_auto=True, title="Konfúzna matica", labels={'x': 'Predikovaná', 'y':'Skutočná'})
        st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")
    except:
        st.error("Najskôr je potrebné model natrénovať !!!")
