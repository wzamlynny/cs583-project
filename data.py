
import pandas as pd
import os

numeric_cols = [
    'Age', 'Breed1', 'Breed2',
    'Gender', 'Color1', 'Color2', 'Color3',
    'MaturitySize', 'FurLength',
    'Vaccinated', 'Dewormed', 'Sterilized',
    'Health', 'Quantity', 'Fee', 'State',
    'VideoAmt', 'PhotoAmt'
]

def load_data(fname):
    return pd.read_csv(fname)

def load_pet_files(regdir):
    """ Extracts all of the files associated with each pet listed
    by the 'PetID' tag.

    regdir - The directory containing the files

    returns a dictionary containing keypairs (k, v) such that v
    matches the regex (k\-.*) where k is the key (a valid PetID).
    """
    pfiles = {}

    # Extract the pet names
    for f in os.listdir(regdir):
        # Extract the name
        n = x[:x.index('-')]
        url = os.path.join(regdir, f)
        if n in pfiles:
            # Add to the entry
            pfiles[n].append(url)
        else:
            # Add a new entry
            pfiles[n] = [url]

    return pfiles
            

def load_train_data():
    # Get the annotations for each pet
    dta = load_data('data/train/train.csv')
     
    # Get the pet pictures
    petpics = load_pet_files('data/train_images/')

    X_num = []
    X_pic = [] # TODO: Load the actual pictures

    Y = []
    
    # Build a single object to store the X values
    X = [X_num, X_pic]
    # Laziness
    if len(X) == 1:
        X = X[0]
    
    for _, row in dta.iterrows():
        # Save the numeric values
        vals = row[numeric_cols]
        X_num.append(vals)
        
        # Save the pictures
        if row['PetID'] in petpics:
            X_pic.append(petpics[row['PetID']])
        else:
            X_pic.append([])
        
        # Save the answer
        Y.append(row['AdoptionSpeed'])

    return X, Y

