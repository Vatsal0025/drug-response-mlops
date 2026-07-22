import streamlit as st
import requests
import pandas as pd

API_URL = "http://52.4.155.216:8000/predict"

st.title(
    "Drug-Induced Gene Expression Prediction"
)

cell_type = st.selectbox(

    "Cell Type",

    [

        "B cells",

        "Myeloid cells",

        "NK cells",

        "T cells CD4+",

        "T cells CD8+",

        "T regulatory cells"

    ]

)

sm_name = st.text_input(

    "Drug Name"

)

sm_lincs_id = st.text_input(

    "LINCS ID"

)

smiles = st.text_input(

    "SMILES"

)


num_genes = st.slider(

    "Number of top genes to display",

    min_value=5,

    max_value=100,

    value=20,

    step=5

)

if st.button("Predict"):

    payload = {

        "cell_type": cell_type,

        "sm_name": sm_name,

        "sm_lincs_id": sm_lincs_id,

        "smiles": smiles,

        "top_k": num_genes

    }

    response = requests.post(
        API_URL,
        json=payload
    )

    result = response.json()

    df = pd.DataFrame(

        result["top_genes"]

    )

    st.dataframe(df)