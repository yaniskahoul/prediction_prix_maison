import streamlit as st  
import numpy as np 
import pandas as pd 
import pickle 

pd.options.display.max_columns = 150 


st.title("Prediction de prix d'une maison")


sqft_living = st.number_input('Surface en squarefeet', value = 300,  key ='sqft_living')
st.write('La surface est de', sqft_living, "squarefeet")


bedrooms = st.number_input('Nombre de chambres', value = 1,  key = 'bedrooms')
st.write('Il y a', bedrooms, "chambres")


bathrooms = st.number_input('Nombre de salle de bains', value = 1,  key = 'bathrooms')
st.write('Il y a', bathrooms, "salle de bains")


sqft_lot = st.slider("Surface du jardin" , 0, 2200000, 500,  key = 'sqft_lot')
st.write("La surface du jardin est de " , sqft_lot)



floors = st.number_input("Nombre d'étage", value = 1, key = 'floors')
st.write('Il y a', floors, 'étage(s)' )


waterfront = st.radio("Possede t-elle une vue sur la mer?" , ('Oui', 'Non'),  key = 'waterfront' ) 

if waterfront == 'Yes': 
    waterfront = 1
else: 
    waterfront = 0
    

view =  st.slider('Notez la vue de 0 à 4', 0, 4, 2, key = 'view')
st.write("La vue est de" , view , "sur 4.")


condition = st.slider("Notez l'état de l'appartement de 1 à 5" , 1, 5, 3,  key = 'condition')
st.write("L'état est de" , condition , "sur 5.")


grade = st.slider(
"Un indice de 1 à 13, où 1-3 est un niveau faible de construction, de design et de la conception des bâtiments, et 11-13 ont un niveau de qualité élevé ", 1, 13, 6,  key = 'grade')
st.write("Le grade est de" , grade , "sur 13.")


sqft_above = st.slider("Superficie en squarefeet de la surface qui se situe au rez-de-chaussée ainsi que dans les étages" , 0, 22000, 100, key = 'sqft_above')
st.write('Cette surface est de', sqft_above)


sqft_basement = st.slider('Superficie en squarefeet de la surface qui se situe en dessous du rez-de-chaussée',  0, 22000, 100 , key = 'sqft_basement')
st.write('Cette surface est de', sqft_basement)


yr_built = st.slider('Année de construction', 1900, 2022, 2000,  key = 'yr_built')
st.write('Le bien a été construit en', yr_built )


yr_renovated = st.slider("Année de rénovation, indiquez 0 si il n'y a jamais eu de rénovation ", 0, 2022, 0,  key = 'yr_renovated')
st.write('Le bien a été rénové en', yr_renovated )




zipcode = st.selectbox(
     'Quel est le code postal?',
     (
"98001",
"98002",
"98003",
"98004",
"98005",
"98006",
"98007",
"98008",
"98009",
"98010",
"98011",
"98013",
"98014",
"98015",
"98019",
"98022",
"98023",
"98024",
"98025",
"98027",
"98028",
"98029",
"98030",
"98031",
"98032",
"98033",
"98034",
"98035",
"98038",
"98039",
"98040",
"98041",
"98042",
"98045",
"98047",
"98050",
"98051",
"98052",
"98053",
"98055",
"98056",
"98057",
"98058",
"98059",
"98062",
"98063",
"98064",
"98065",
"98070",
"98071",
"98072",
"98073",
"98074",
"98075",
"98077",
"98083",
"98089",
"98092",
"98093",
"98101",
"98102",
"98103",
"98104",
"98105",
"98106",
"98107",
"98108",
"98109",
"98111",
"98112",
"98113",
"98114",
"98115",
"98116",
"98117",
"98118",
"98119",
"98121",
"98122",
"98124",
"98125",
"98126",
"98127",
"98129",
"98131",
"98133",
"98134",
"98136",
"98138",
"98139",
"98141",
"98144",
"98145",
"98146",
"98148",
"98154",
"98155",
"98158",
"98160",
"98161",
"98164",
"98165",
"98166",
"98168",
"98170",
"98174",
"98175",
"98177",
"98178",
"98181",
"98185",
"98188",
"98190",
"98191",
"98194",
"98195",
"98198",
"98199",
"98224",
"98288"),  key = 'zipcode')
st.write('Le code postal est ', zipcode)





lat = st.number_input("Entrez la latitude du bien", value = 47,  key = 'lat')
st.write('lat:', lat) 

long = st.number_input("Entrez la longitude du bien",  value = -122,  key = 'long')
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


X = pd.read_csv("df_cleaned.csv")
y = pd.read_csv("df_modelisation_price.csv")



parametres['zipcode'] = parametres['zipcode'].astype(str)
parametres['month'] = parametres['month'].astype(str)

pickle_in = open('my_pipe_lr.pkl', 'rb') 
my_pipe_lr = pickle.load(pickle_in)



if st.button('Estimez le prix de votre bien'):
    print(parametres.info())
    print("-----------------")
    print(data)
    prediction = my_pipe_lr.predict(parametres)

    prix = round(prediction[0][0],2)
    st.write('# Le prix du bien immobilier est:', prix, "$")
