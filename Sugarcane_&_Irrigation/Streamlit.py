import ls_model as md
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# streamlit run "d:\study\ai\Projects\4-ML Project\2-Predicting Irrigation Need\Streamlit.py"
# aboalababir369@gmail.com

st.title("Predicting_Irrigation_Need_PCA")
st.info("This is app build a machine learning model")


with st.expander("Show train Data"):
               st.write("==Row Data==")
               train= pd.read_csv(r'D:\study\ai\Projects\4-ML Project\2-Predicting Irrigation Need\train.csv')
               train


               st.write('**x_train**')
               x_train= train.drop(['id', 'Irrigation_Need'], axis= 1)
               x_train


               st.write('**y_train**')
               y_train = train['Irrigation_Need']
               y_train

cat_v = []
num_v = []

num_v = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_v = x_train.select_dtypes(include=['object']).columns.tolist()





with st.expander('Data visualization'):
    with st.expander('Pie Chart'):
                for i in cat_v:   
                    fig, ax = plt.subplots()

                    x_train[i].value_counts().plot.pie(autopct='%1.1f%%',ax=ax)

                    ax.set_ylabel("")  
                    ax.set_title(i)  
                    fig.tight_layout()
                    st.pyplot(fig)



    with st.expander("Histograms"):

      for col in num_v:

        fig, ax = plt.subplots()

        ax.hist(x_train[col], bins=20, edgecolor="black", linewidth=1.2)

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        fig.tight_layout()  

        st.pyplot(fig)

        st.markdown("---")  



with st.sidebar:
    st.header('Input Feature')

    Crop_Growth=st.selectbox('Crop_Growth_Stage', ('Sowing', 'Vegetative', 'Flowering', 'Harvest'))
    Mulching=st.selectbox('Mulching_Used', ('Yes', 'No'))

    Rainfall = st.slider('Rainfall_mm', min_value=0.0, max_value=3000.0,value=500.0)

    Wind_Speed = st.slider('Wind_Speed_kmh', min_value=0.0, max_value=50.0, value=10.0)

    Temperature = st.slider('Temperature_C', min_value=0.0, max_value=50.0, value=25.0)

    Soil_Moisture = st.slider('Soil_Moisture', min_value=0.0, max_value=100.0, value=30.0)



    data = {
    'Soil_Moisture': Soil_Moisture,
    'Temperature_C': Temperature,
    'Mulching_Used_Yes': int(Mulching == 'Yes'),
    'Wind_Speed_kmh': Wind_Speed,
    'Crop_Growth_Stage_Flowering': int(Crop_Growth == 'Flowering'),
    'Crop_Growth_Stage_Sowing': int(Crop_Growth == 'Sowing'),
    'Crop_Growth_Stage_Harvest': int(Crop_Growth == 'Harvest'),
    'Crop_Growth_Stage_Vegetative': int(Crop_Growth == 'Vegetative'),
    'Rainfall_mm': Rainfall
}

input_df = pd.DataFrame(data, index=[0])
#     input_irr= pd.concat([input_df, x_train], axis= 0)



with st.expander('Input Feature'):
      st.write("**Input Irrigation**")
      input_df





if st.button("Predict Irrigation Need"):
    input_df = input_df.reindex(columns=md.features, fill_value=0)

    probs = md.model.predict_proba(input_df)
    pred_index = np.argmax(probs, axis=1)[0]

    prediction = md.model.classes_[pred_index]
    confidence = probs[0][pred_index]

    st.success(f"Prediction: {prediction}")
    st.info(f"Confidence: {confidence:.2%}")

    # Probability Table
    st.write("### Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Class": md.model.classes_,
        "Probability": probs[0]
    })

    st.dataframe(prob_df)

    # Bar Chart
    fig, ax = plt.subplots()
    ax.bar(prob_df["Class"], prob_df["Probability"])
    ax.set_title("Class Probabilities")
    ax.set_ylabel("Probability")
    st.pyplot(fig)
