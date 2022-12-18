import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from scipy import stats

st.set_page_config(
    page_title="Exploratívna analýza",
    page_icon="📊",
    layout="wide",
)


st.sidebar.header("Exploratívna analýza")
progress_bar = st.sidebar.progress(50)

st.write("# 📊 Exploratívna analýza ")

def nacitaj ():
    return(pd.read_csv(r"campaign.csv",sep = ";"))

data = nacitaj()

st.dataframe(data,use_container_width=True)

#### funkcie
def zobraz_chybajuce(vstup):
    nulove = vstup.isnull().sum()
    chybajuce_udaje=[]
    percenta = []
    for index,stlpec in enumerate(nulove):
        if stlpec != 0:
            chybajuce_udaje.append(nulove.index[index])
            percenta.append(100 * stlpec / data.shape[0])
    tabulka_chabajucich_dat = pd.DataFrame(data=percenta, columns = ["percenta"] , index = chybajuce_udaje ).T
    fig = px.bar(tabulka_chabajucich_dat, barmode="group")
    st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")

def nakresli_box (premenna):
    fig = px.box(data, x=premenna)
    st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")

def nakresli_graf (premenna):
    fig = px.histogram(data, x=premenna,color = "y")
    st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")

tab_1, tab_2 ,tab_3, tab_4, tab_5 = st.tabs(["Odstránenie chýbajúcich pozorovaní",  "Grafická analýza", "Odstránenie odľahlých pozorovaní", "Riešenie multikolinearity","Prekódovanie kategoriálnych premenných"])

with tab_1:
    col1, col2, col3 = st.columns(3)
    with col1:
        zobraz_chybajuce(data)
        with st.expander("Vysvetlenie"):
            st.markdown("""
                        Na grafe uvádzame relatívnu početnosť chýbajúcich údajov. ako možno vidieť premenná "poutcome" obsahuje až 82% chýbajúcich hodnôt. Z tohto dôvodu sme sa rozhodli vylúčiť túto premennú z nášho modelu. Rovnako sme postupovali aj pri premennej "contact"
                        ktorá obshuje takmer 30 percent chýbajúcich hodnôt.
                        """)

    with col2:
        del data["poutcome"]
        del data["contact"]
        zobraz_chybajuce(data)
        with st.expander("Vysvetlenie"):
            st.markdown("""
                        Ako možno vidieť zvyšné premenné obsahujú len malý počet chýbajúcich hodnôt. Premenná "education" približne 4%
                        a premenná "job" približne 0,7 %. Pri týchto premenných sme sa rozhodli odstrániť všetky pozorovania ktoré obsahovali chýbajúce hodnoty.
                        """)
    with col3:
        def odstran_pozorovania():
            global data
            data = data.dropna(subset = data.columns)
        odstran_pozorovania()
        zobraz_chybajuce(data)
        with st.expander("Vysvetlenie"):
            st.markdown("""
                        Na uvedenom grafe možno vidieť že naše dáta už neobsahujú chábajúce údaje v rámci žiadenj z premenných.
                        """)

with tab_2:
    col1, col2 = st.columns(2)
    stlpce = data.columns.tolist()
    with col1:
        for x in stlpce[:round(len(stlpce)/2)]:
            nakresli_graf(x)
    with col2:
        for x in stlpce[round(len(stlpce)/2):]:
            nakresli_graf(x)

with tab_3:
    col1, col2 = st.columns(2)
    with col1:
        st.write("## S odľahlými pozorovaniami")
        for x in data.select_dtypes(include=np.number):
            nakresli_box(x)
    with col2:
        st.write("## Bez odľahlých pozorovaní")
        data=data[(np.abs(stats.zscore(data.select_dtypes(include=np.number)))<3).all(axis=1)]
        for x in data.select_dtypes(include=np.number):
            nakresli_box(x)

with tab_4:

    st.write("## Korelačné matice")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.imshow(data.corr())
        st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")
    with col2:
        del data["pdays"]
        fig = px.imshow(data.corr())
        st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")

    with st.expander("Vysvetlenie"):
        st.markdown("""
                Nakoľko premenná pdays silno korelovala s premennou previous, rozhodli sme sa o jej odstránenie.""")

    st.write("## Variačno inflačný faktor")
    from statsmodels.tools.tools import add_constant
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    def vi_factor():
        vif=pd.DataFrame()#vytvoríme nová tabuľku
        df=add_constant(data[["age","balance","duration","campaign"]])
        vif["VIF"]=[variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
        vif["features"]=df.columns
        return vif
    st.markdown('#')
    st.markdown('#')
    st.dataframe(vi_factor(),use_container_width=True)

    with st.expander("Vysvetlenie"):
        st.markdown("""
                Všetky hodnoty varično inflačného faktora sú nižšie ako 5, respektíve sú veľmi blíze hodnote 1. Z doho dôvodu môžeme konštatovať že medzi premennými sa nevyskytuje korelácia.""")

with tab_5:
    zoznam = {}
    for col in data.columns:
        if data[col].dtypes == "object":
            lst = {}
            for index, prem in enumerate(data[col].unique()):
                lst[prem] = int(index)
            zoznam[col] = lst

    def get_keys_from_value(premenna, kod_obmeny):
        return [k for k, v in zoznam[str(premenna)].items() if v == kod_obmeny]

    # táto funkcia zistí kod priradený pôvodnej obmene
    def get_value_from_key(premenna, obmena):
        return zoznam[str(premenna)][str(obmena)]
    @st.cache
    def zakoduj (vstupne):
        dat = vstupne.copy()
        for col in dat.columns:
            if dat[col].dtypes == "object":
                for index, hodnota in enumerate(dat[col]):
                    dat.loc[index,col] = int(zoznam[col][hodnota])
                dat[col] = dat[col].astype("category")
        return(dat)

    data = data.reset_index(drop = True)
    data = zakoduj(data)

    if 'dat' not in st.session_state:
        st.session_state['dat'] = data

    if 'zn' not in st.session_state:
        st.session_state['zn'] = zoznam
    st.dataframe(st.session_state['dat'],use_container_width=True)
    with st.expander("Vysvetlenie"):
        st.markdown("""Všetky kategoriálne premenné bolo potrebné prekódovať tak, aby s nimi mohol model pracovať. Vyššie uvedený dataframe už je pripravený na využitie v modeloch.""")
