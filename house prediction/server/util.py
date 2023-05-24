import json
import pickle
import numpy as np
__locations=None
__data_columns=None
__model=None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index=-1
    
    params = np.zeros(len(__data_columns))
    params[0] = sqft
    params[1] = bath
    params[2] = bhk
    if loc_index>=0:
        params[loc_index] = 1
    
    price = round(__model.predict([params])[0],2)
    return price


def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts... starts")
    global __data_columns
    global __locations

    with open("./server/artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    global __model
    with open("C:/Users/nitis/Desktop/house prediction/server/artifacts/banglore_home_prices_model.pickle", "rb") as f:
        __model=pickle.load(f)
    print("loading saved artifacts...done")

if __name__== '__main__' :
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price("1st Phase JP Nagar",1000, 3, 4))



