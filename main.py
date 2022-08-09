import streamlit as st  
import numpy as np 
import pandas as pd 
import pickle 

pd.options.display.max_columns = 150 


st.title("Prediction de prix d'une maison")


sqft_living = st.number_input('Surface en squarefeet', key ='sqft_living')
st.write('La surface est de', sqft_living, "squarefeet")


bedrooms = st.selectbox(
    'Nombre de chambres ',
    ('0', '1', '2','3','4','5','6', '7', '8','9','10', '11','12') , key = 'bedrooms')
st.write('Il y a', bedrooms, "chambre(s)")


bathrooms = st.number_input('Nombre de salle de bains', key = 'bathrooms')
st.write('Il y a', bathrooms, "salle de bains")


sqft_lot = st.number_input('Surface du jardin', key = 'sqft_lot')
st.write('La surface du jardin est de ', sqft_lot)


floors = st.number_input("Nombre d'étage", key = 'floors')
st.write('Il y a', floors, 'étage(s)' )


waterfront = st.radio("Possede t-elle une vue sur la mer?" , ('Yes', 'No'),  key = 'waterfront' ) 

if waterfront == 'Yes': 
    waterfront = 1
else: 
    waterfront = 0
    

view =  st.slider('Notez la vue de 0 à 4', 0, 4, 2,  key = 'view')
st.write("La vue est de" , view , "sur 4.")


condition = st.slider("Notez l'état de l'appartement de 1 à 5" , 1, 5, 3,  key = 'condition')
st.write("L'état est de" , condition , "sur 5.")


grade = st.slider(
"Un indice de 1 à 13, où 1-3 est un niveau faible de construction, de design et de la conception des bâtiments, et 11-13 ont un niveau de qualité élevé ", 1, 13, 6,  key = 'grade')
st.write("Le grade est de" , grade , "sur 13.")


sqft_above = st.number_input('Superficie en squarefeet de la surface qui se situe au rez-de-chaussée ainsi que dans les étages',   key = 'sqft_above')
st.write('Cette surface est de', sqft_above)



sqft_basement = st.number_input('Superficie en squarefeet de la surface qui se situe en dessous du rez-de-chaussée',   key = 'sqft_basement')
st.write('Cette surface est de', sqft_basement)


yr_built = st.slider('Année de construction', 1900, 2022, 2000,  key = 'yr_built')
st.write('Le bien a été construit en', yr_built )


yr_renovated = st.slider("Année de rénovation, indiquez 0 si il n'y a jamais eu de rénovation ", 0, 2022, 0,  key = 'yr_renovated')
st.write('Le bien a été rénové en', yr_renovated )


zipcode = st.number_input("Entrez votre code postal",   key = 'zipcode')
st.write('Le code postal est ', zipcode)


lat = st.number_input("Entrez la latitude du bien",   key = 'lat')
st.write('lat:', lat) 

long = st.number_input("Entrez la longitude du bien",    key = 'long')
st.write('long:', long) 


year = st.slider('Année de mise en vente', 2022, 2026, 2022,    key = 'year')

month = st.slider('Mois de mise en vente', 1, 12, 6,  key = 'month')



data = {
    'sqft_living':sqft_living,
    'bedrooms': bedrooms,
    'bathrooms':bathrooms,
    'sqft_lot':sqft_lot,
    'floors':floors,
    'waterfront':waterfront,
    'view':view,
    'condition':condition,
    'grade':grade,
    'sqft_above':sqft_above,
    'sqft_basement':sqft_basement,
    'yr_built':yr_built,
    'yr_renovated':yr_renovated,
    'zipcode':zipcode,
    'lat':lat,
    'long':long,
    'year':year,
    'month':month
}

parametres = pd.DataFrame(data, index=[0])


X = pd.read_csv("/home/kahoul/Bureau/projet_final/df_modelisation.csv")
y = pd.read_csv("/home/kahoul/Bureau/projet_final/df_modelisation_price.csv")


pickle_in = open('my_pipe_lr.pkl', 'rb') 
my_pipe_lr = pickle.load(pickle_in)


if st.button('Estimez le prix de votre bien'):
    print(parametres.info())
    print("-----------------")
    print(data)
    prediction = my_pipe_lr.predict(parametres)

    prix = round(prediction[0][0],2)
    st.write('# Le prix du bien immobilier est:', prix, "$")
