import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.set_page_config(
    page_title="Predikcia",
    page_icon="🥠",
    layout="wide",
)

st.sidebar.header("Predikcia")
progress_bar = st.sidebar.progress(100)

try:
    data = st.session_state['dat']
    zoznam = st.session_state['zn']
    rfc = st.session_state['model']
    def get_keys_from_value(premenna, kod_obmeny):
        return [k for k, v in zoznam[str(premenna)].items() if v == kod_obmeny]

    # táto funkcia zistí kod priradený pôvodnej obmene
    def get_value_from_key(premenna, obmena):
        return zoznam[str(premenna)][str(obmena)]

    def zakodovat (vstupne):
        dat = vstupne.copy()
        for col in dat.columns:
            if dat[col].dtypes == "object":
                for index, hodnota in enumerate(dat[col]):
                    dat.loc[index,col] = int(zoznam[col][hodnota])
                dat[col] = dat[col].astype("category")
        return(dat)

    with st.sidebar.form("my_form"):
        dict =  {}
        for x in st.session_state['prem']:
            if data[x].dtypes != "category":
                dict[x] = [st.slider(x,min_value=min(data[x]), max_value=max(data[x]))]
            if data[x].dtypes == "category":
                dict[x] = [st.selectbox(x,zoznam[x])]
        submitted = st.form_submit_button("Odhadnúť")
        if submitted:
            pred = pd.DataFrame(dict)
            predikovana = get_keys_from_value("y",rfc.predict(zakodovat(pred))[0])



    st.write("# 🥠 Predikcia")
    st. write('***')
    st. write('#')
    try:
        st.write("### Vlastnosti osloveného klienta")
        st.dataframe(pred,use_container_width=True)
        if predikovana[0] =="yes":
            st.write("### ✅ Klient so zvolenými vlastnosťami **vloží** peniaze na terminovaný vklad.")
        else:
            st.write("### ❌ Klient so zvolenými vlastnosťami **nevloží** peniaze na terminovaný vklad.")
    except:
        st.error("Najskôr je potrebné zadať vstupné parametre !!!")
except:
        st.error("Najskôr je potrebné natrénovať model !!!")
