import streamlit as st
import pickle

model = pickle.load(open('ML/model.pkl', 'rb'))

def show_predict_page():
    st.title("Predict logS")

    st.write("""### We need some information to predict""")
    #Input variable
    MolLogP = st.number_input("Octanol/water partition coefficient (MolLogP) :")
    MolWt = st.number_input("Molecular Mass (MolWt): ")
    NumRotatableBonds = st.number_input("Number of Rotatable Bonds: ")
    AromaticProportion = st.number_input("Aromatic Proportion: ",0,1)
        
    #Predict code
    
    if st.button("Predict logS"):
        makeprediction = model.predict([[MolLogP, MolWt, NumRotatableBonds,
        AromaticProportion]])
        st.subheader(f"The estimated logS is {makeprediction[0]:.2f}")
       