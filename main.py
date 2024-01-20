# To run website: streamlit run main.py
import streamlit as st

st.set_page_config(layout="wide")
st.title("Glove Detection System")

# Columns
col1, col2 = st.columns(2)

with col1:
    selected_glove = st.selectbox(
        label="Select type of glove",
        options=["Select Glove","Glove 1", "Glove 2", "Glove 3", "Glove 4"],
    )

    selected_defect = st.selectbox(
        label="Select type of defect",
        options=["Select Defect", "All","Defect 1", "Defect 2", "Defect 3"],
        disabled= selected_glove == "Select Glove"
    )

with col2:
    # Image Uploader
    uploaded_image = st.file_uploader(
        label="Upload image of glove",
        type=["png", "jpg", "jpeg"],
        disabled= selected_defect == "Select Defect"
    )

    # Detect Defect Button
    st.button(
        label="Detect Defect",
        disabled= uploaded_image is None
    )

    # Display Image
    if uploaded_image is not None:
        st.image(uploaded_image, use_column_width=True)