import pandas as pd
import pickle
from sklearn.ensemble  import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('Clean Data_pakwheels.csv')

df = df.drop(columns=['Unnamed: 0', 'Location'])

mapping = {
    'Engine Type': {'Petrol': 0, 'Diesel': 1, 'Hybrid':2},
    'Body Type': {'Hatchback': 0, 'Sedan': 1, 'SUV': 2, 'Cross Over': 3, 'Van': 4, 'Mini Van': 5},
    'Color': {
      'Silver': 0,
      'White': 1,
      'Black': 2,
      'Beige': 3,
      'Grey': 4,
      'Brown': 5,
      'Pink': 6,
      'Assembly': 7,
      'Maroon': 8,
      'Burgundy': 9,
      'Gold': 10,
      'Blue': 11,
      'Red': 12,
      'Indigo': 13,
      'Unlisted': 14,
      'Green': 15,
      'Turquoise': 16,
      'Orange': 17,
      'Bronze': 18,
      'Purple': 19,
      'Yellow': 20,
      'Navy': 21,
      'Magenta': 22,
      'Wine': 23
    },
    'Assembly': {'Imported': 0, 'Local': 1},
    'Transmission Type': {'Automatic': 0, 'Manual': 1},
    'Registration Status': {'Un-Registered': 0, 'Registered': 1}
}

for column, map_dict in mapping.items():
    df[column] = df[column].map(map_dict)

df = df.drop(columns=['Company Name', 'Model Name'])

df2 = df[0:5000]
y = df2.pop("Price")
X = df2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

rf = RandomForestRegressor()

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

pickle.dump(rf, open('model_avril.pkl', 'wb'))