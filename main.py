# To run website: streamlit run main.py
import streamlit as st
from scripts.cloth.open_seam import detect_open_seam
from scripts.cloth.tearing import detect_tearing
from scripts.cloth.missing_finger import detect_missing_finger
from scripts.polydot.missing_dot import detect_missing_dot
from scripts.polydot.color_inconsistency import detect_color_inconsistency
from scripts.polydot.tearing_poly import detect_tearing_poly
from scripts.plastic.tear import detect_tear
from scripts.plastic.stain import detect_stain
from scripts.plastic.size import detect_inconsistent_size

st.set_page_config(layout="wide")
st.title("Glove Detection System")

# Columns
col1, col2 = st.columns(2)

def glove_defect_dropdown():
    if selected_glove == "Cloth Glove":
        return cloth_glove_defects
    elif selected_glove == "Rubber Glove":
        return rubber_glove_defects
    elif selected_glove == "Poly Dot Glove":
        return poly_dot_glove_defects
    elif selected_glove == "Plastic Glove":
        return plastic_defects
    else:
        return []

def controller():
    # Add Dropdown logic here
    if selected_glove == "Cloth Glove" and selected_defect == "Seam":
        st.image(detect_open_seam(uploaded_image))
    elif selected_glove == "Cloth Glove" and selected_defect == "Tearing":
        st.image(detect_tearing(uploaded_image))
    elif selected_glove == "Cloth Glove" and selected_defect == "Missing Finger":
        st.image(detect_missing_finger(uploaded_image))
    elif selected_glove == "Poly Dot Glove" and selected_defect == "Missing Dot":
        st.image(detect_missing_dot(uploaded_image))
    elif selected_glove == "Poly Dot Glove" and selected_defect == "Color Inconsistency":
        st.image(detect_color_inconsistency(uploaded_image))
    elif selected_glove == "Poly Dot Glove" and selected_defect == "Tearing":
        st.image(detect_tearing_poly(uploaded_image))
    elif selected_glove == "Plastic Glove" and selected_defect == "Tearing":
        st.image(detect_tear(uploaded_image))
    elif selected_glove == "Plastic Glove" and selected_defect == "Stain":
        st.image(detect_stain(uploaded_image))
    elif selected_glove == "Plastic Glove" and selected_defect == "Inconsistent Size":
        st.image(detect_inconsistent_size(uploaded_image))
    else:
        st.text("No defect detected")

# First Column (Left)
with col1:
    # Glove Selection
    gloves = ["Select Glove","Cloth Glove", "Rubber Glove", "Poly Dot Glove", "Plastic Glove"]

    # Glove Defect For Each Glove
    cloth_glove_defects = ["All","Seam", "Tearing", "Missing Finger"]
    rubber_glove_defects = ["All","Defect 1", "Defect 2", "Defect 3"]
    poly_dot_glove_defects = ["All","Missing Dot", "Color Inconsistency", "Tearing"]
    plastic_defects = ["All","Inconsistent Size", "Tearing", "Stain"]

    # Dropdown
    selected_glove = st.selectbox(
        label="Select type of glove",
        options=gloves,
    )

    selected_defect = st.selectbox(
        label="Select type of defect",
        options=glove_defect_dropdown(),
        disabled= selected_glove == "Select Glove"
    )
    
# Second Column (Right)
with col2:
    # Image Uploader
    uploaded_image = st.file_uploader(
        label="Upload image of glove",
        type=["png", "jpg", "jpeg"],
        disabled= selected_defect == "Select Defect",
    )

    # Detect Defect Button
    detect_defect_button = st.button(
        label="Detect Defect",
        disabled= uploaded_image is None
    )

    with st.empty():
        # Display Image
        if uploaded_image is not None:
            controller()
            if detect_defect_button:
                controller()
            else:
                st.image(uploaded_image, use_column_width=True)