import streamlit as st
from Ecg import  ECG

ecg = ECG()
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  ecg_user_image_read = ecg.getImage(uploaded_file)
  st.image(ecg_user_image_read)
  ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
  dividing_leads=ecg.DividingLeads(ecg_user_image_read)
  ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
  ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
  ecg_1dsignal = ecg.CombineConvert1Dsignal()
  ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
  ecg_model=ecg.ModelLoad_predict(ecg_final)
  # my_expander5 = st.expander(label='PREDICTION')
  # with my_expander5:
  st.write(ecg_model)
