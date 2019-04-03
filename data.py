
import os
from PIL import Image
import numpy as np
import pandas as pd

from tqdm import tqdm

numeric_cols = [
    'Age', 'Type', 'Breed1', 'Breed2',
    'Gender', 'Color1', 'Color2', 'Color3',
    'MaturitySize', 'FurLength', 'Vaccinated',
    'Dewormed', 'Sterilized', 'Health',
    'Quantity', 'Fee', 'State',
    'VideoAmt', 'PhotoAmt'
]

one_hot_cols = {
    'Type': 2, 'Breed1': 307, 'Breed2': 307,
    'Gender': 3, 'Color1': 7, 'Color2': 7,
    'Color3': 7, 'MaturitySize': 5,
    'FurLength': 4, 'Vaccinated': 4,
    'Dewormed': 4, 'Sterilized': 4,
    'Health': 4, 'State': 15
}

def one_hot_encode(df, col, num_class, labels=None, remove_original=True):
    ''' Takes in dataframe df and replaces col with num_class columns
        For example, use as follows
        for col, num_class in data.one_hot_cols.items():
            one_hot_encode(train_df, col, num_class)
    '''
    # get the true values from data
    column_values = np.sort(df[col].unique())
    
    if num_class == 2:
        # These can just be boolean
        df[col] = (df[col] == column_values[0]).astype(int)
    else:
        if (num_class != len(column_values)):
            # Issue if the lengths don't match, don't use labels
            labels = None
            
        for i in range(num_class):
            if (i >= len(column_values)):
                break # Index out of bounds
            cur_value = column_values[i]
            if labels:
                # If labels are provided use these for columns names
                df[labels[i]] = (df[col] == cur_value).astype(int)

            else:
                # Otherwise just append id to col name
                df[col+'_'+str(cur_value)] = (df[col] == cur_value).astype(int)
    
        if remove_original:
            # delete original column
            df.drop(col, axis=1, inplace=True)

def load_data(fname):
    return pd.read_csv(fname)

def load_pet_files(regdir):
    """ Extracts all of the files associated with each pet listed
    by the 'PetID' tag.

    regdir - The directory containing the files

    returns a dictionary containing keypairs (k, v) such that v
    matches the regex (k\-.*) where k is the key (a valid PetID).
    """
    if os.path.isfile(os.path.join(regdir + 'picked_pictures.npy')):
        print("Images loaded from existing file")
        return np.load(os.path.join(regdir + 'picked_pictures.npy'))
    pfiles = {}

    # Extract the pet names
    for f in tqdm(os.listdir(regdir), desc='Loading Pet Files'):
        # Extract the name
        n = f[:f.index('-')]
        url = os.path.join(regdir, f)
        img = load_image(url)
        if n in pfiles:
            # Add to the entry
            pfiles[n].append(img)

        else:
            # Add a new entry
            pfiles[n] = [img]

    np.save(os.path.join(regdir + 'picked_pictures.npy'), pfiles)
    return pfiles
            

def load_train_data():
    # Get the annotations for each pet
    dta = load_data('data/train/train.csv')
     
    # Get the pet pictures
    petpics = load_pet_files('data/train_images/')

    # Get the state ids
    states = load_data('data/state_labels.csv')
    states = states['StateID'].tolist()

    X_num = []
    X_pic = []

    Y = []
    
    # Build a single object to store the X values
    X = [X_num, X_pic]
    
    for _, row in dta.iterrows():
        # Save the numeric values
        vals = row[numeric_cols]
        state = [x == row['State'] for x in states]
        assert(sum(state) == 1)
        X_num.append(np.array(list(vals) + state))
        
        # Save the pictures
        if row['PetID'] in petpics:
            X_pic.append(petpics[row['PetID']])
        else:
            X_pic.append([])
        
        # Save the answer
        Y.append(row['AdoptionSpeed'])

    # Laziness
    if len(X) == 1:
        X = np.array(X[0])
    else:
        X = list(map(np.array, X))

    Y = np.array(Y)

    return X, Y

def load_image(img_file, size=64):
    # print('img file:', img_file)
    img = Image.open(img_file)
    img = img.resize((size, size), Image.ANTIALIAS)
    return np.array(img)
