import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Motivácia",
    page_icon="💡",
    layout="wide",
)

st.sidebar.header("Motivácia")
progress_bar = st.sidebar.progress(10)

st.markdown("""
            ### 👨‍💻 Autori
            - Michal Ďurčík
            - Juraj Špánik
            📆 18.12.2022
            """)

tab_1, tab_2, tab_3 = st.tabs(["💡 Motivácia", "📄 Vstupné dáta","🎯 Cieľ modelu"])
with  tab_1:
    st.markdown(
        """$\qquad$ Termínované vklady sú pre banku významným zdrojom príjmov. Termínovaný vklad je peňažná investícia držaná vo finančnej inštitúcii. Vaše peniaze sa investujú za dohodnutú úrokovú sadzbu počas pevne stanoveného času alebo obdobia. Existuje množstvo spôsobov ako môže banka osloviť svojich poteniálnych zákazníkov, ako je napríklad: e-mailový marketing, reklamy, telefonický marketing alebo digitálny marketing.Telefonické marketingové kampane stále zostávajú jedným z najefektívnejších spôsobov ako osloviť ľudí. Vyžadujú si však obrovské investície, pretože sa najímajú veľké call centrá, aby tieto kampane skutočne realizovali. Preto je dôležité vopred identifikovať zákazníkov, u ktorých je najväčšia pravdepodobnosť úspechu, aby sa na nich dalo konkrétne zacieliť prostredníctvom hovoru."""
    )
with tab_2:
    st.markdown(
        """$\qquad$ Údaje sa týkajú priamych marketingových kampaní portugalskej bankovej inštitúcie. Marketingové kampane boli založené na telefonátoch. Často sa vyžadovalo viac kontaktov na toho istého klienta.  V našich vstupných dátach sa nachádzajú nasledovné údaje:
        """)
    st.markdown("""
        - **age** - Vek klienta.
        - **job** - Zamestnanie klienta. ("management", "technician", "entrepreneur", "blue-collar", "retired", "admin.", "services", "self-employed", "unemployed", "housemaid", "student")
        - **marital** - Rodinný stav klienta. ("married","divorced","single")
        - **education** - Vzdelanie klienta. ("secondary","primary","tertiary")
        - **default** - Došlo u klienta k defaultu ? ("yes","no")
        - **balance** - Priemerný ročný zostatok na účte.
        - **housing** - Má klient hypotéku ? ("yes","no")
        - **loan** - Má klient osobnú pôžičku ? ("yes","no")
        - **contact** - Trvanie posledného hovoru v sekundách.
        - **day** - Deň posledného kontaktu v mesiaci.
        - **month** - Mesiac posledného kontaktu v roku.
        - **duration - Trvanie posledného hovoru v sekundách.
        - **campaign** - Počet uskutočnených hovorov s klientom počas tejto kampane.
        - **pdays** - Počet dní, ktoré uplynuli po tom, čo bol klient naposledy kontaktovaný z predchádzajúcej kampane ( -1 znamená, že klient nebol predtým kontaktovaný)
        - **previous** - Počet kontaktov uskutočnených s daným klientom pred touto kampaňou.
        - **poutcome** - Výsledok predchádzajúcej marketingovej kampane („iný“, „neúspech“, „úspech“)
        - **y** - Vložil klient peniaze na terminovaný vklad ? ("yes","no")

    """)
with  tab_3:
    st.markdown(
        """$\qquad$ Ako sme už spomenuli, našim cieľom je odhadnúť profil klienta, ktorý s najväčšou pravdepodobnosťou pozitívne zareaguje na našu kampaň a vloží svoje finančné prostriedky na terminovaný vklad. Keď budeme vedieť profil kliena, ktorý s najväčšou pravdepodobnosťou svoje finančné prostriedky vloží na terminovaný vklad, môžeme efektívnejšie cieliť našu reklamu a tým nepriamo aj znižovať náklady na kampaň. Žiadúce je teda sformulovať taký model, ktorý dokáže najlepšie opísať profil respondenta."""
    )
