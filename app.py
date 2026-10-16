"""
app.py: the website version.

    python -m streamlit run app.py

Tabs on the left, one page per model on the right. Anyone enters their holdings on the
Portfolio tab, hits Run, and every other tab fills in. The pages live in webapp/views.py;
all the math comes from riskmodel/pipeline.py, same as the terminal version (main.py).
"""
import streamlit as st

from webapp import ui, views

st.set_page_config(page_title="Factor Risk Model", page_icon=":material/monitoring:",
                   layout="wide")
ui.apply_style()
st.logo("webapp/logo.svg", size="large")
st.navigation(views.sidebar(), expanded=True).run()
