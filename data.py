
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from tqdm import tqdm

# Text parsing imports
import re
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from unicodedata import normalize

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
    'FurLength': 4, 'Vaccinated': 3,
    'Dewormed': 3, 'Sterilized': 3,
    'Health': 4, 'State': 15
}

def one_hot_encode(df, col, num_class=None, labels=None, inplace=False):
    ''' Takes in dataframe df and replaces col with num_class columns
        For example, use as follows
        for col, num_class in data.one_hot_cols.items():
            one_hot_encode(train_df, col, num_class)
    '''
    # get the true values from data
    column_values = np.sort(df[col].dropna().unique())
    if num_class == None:
        num_class = len(column_values)
    if num_class == 2:
        # These can just be boolean
        if inplace:
            # The second value will be True or 1 - keep relative order
            df[col] = (df[col] == column_values[1]).astype(int)
        else:
            return (df[col] == column_values[1]).astype(int)
    else:        
        if labels is not None:
            res = np.zeros((len(df), num_class))
            for i, label in enumerate(labels):
                if inplace:
                    df[col+'_'+str(label)] = (df[col] == label).astype(int)
                else:
                    one_hot = np.zeros(num_class)
                    one_hot[i] = 1
                    res[df[col] == label] = one_hot
        else:
            res = np.zeros((len(df), num_class))
            for i in range(num_class):
                if (i >= len(column_values)):
                    break # Index out of bounds
                cur_value = column_values[i]

                if inplace:
                    df[col+'_'+str(cur_value)] = (df[col] == cur_value).astype(int)
                else:
                    one_hot = np.zeros(num_class)
                    one_hot[i] = 1
                    res[df[col] == cur_value] = one_hot
    
        if inplace:
            # delete original column
            df.drop(col, axis=1, inplace=True)
        else:
            return res

def get_sentiment(df, sentiment_location):
    ''' Parses the text sentiment metadata and adds a few additional
        metrics to the specified dataframe.
    '''
    sentiment_files = glob(sentiment_location + "/*")

    # Add some additional metrics from the sentiment files
    for s_file in sentiment_files:
        pet_id = s_file.split('/')[-1].split('.')[0]
        with open(s_file) as json_file:
            data = json.load(json_file)

            df.loc[df["PetID"] == pet_id, "SentimentMagnitude"] = data['documentSentiment']['magnitude']
            df.loc[df["PetID"] == pet_id, "SentimentScore"] = data['documentSentiment']['score']
            df.loc[df["PetID"] == pet_id, "NumSentences"] = len(data['sentences'])

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

    # Load parsed sentiment
    sentiment = load_data('sentiment_parsed.csv')

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

        # Join sentiment on PetID
        s = sentiment[sentiment['PetID'] == row['PetID']]
        for col in s:
            row[col] = s[col][0]
        
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

def parse_description(train_df, test_df, sequence_length_w = 30):
    train_res = pd.DataFrame(np.zeros((len(train_df),0)))
    test_res = pd.DataFrame(np.zeros((len(test_df),0)))

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(treebank_tag):
        '''Source: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
        Allows lemmatization to work with more different POS
        '''
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # default to noun
            return wordnet.NOUN
        
    def my_lemmatize(text):
        ''' First tags text, determines the pos, and finally lemmatizes
        '''
        text = nltk.pos_tag(text)
        return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in text]

    def preprocess_string(text):
        if type(text) != str:
            return []
        text = normalize('NFD', text).encode('ascii', 'ignore')
        text = text.decode('UTF-8')
        text = str.lower(text)
        text = re.sub('[^a-zA-Z\s]+', '', text)
        text = re.sub(r'\W*\b\w{1,3}\b', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = my_lemmatize(tokens)
        return tokens

    # Process into tokens
    train_res["Tokens"] = [preprocess_string(text) for text in train_df["Description"]]
    test_res["Tokens"] = [preprocess_string(text) for text in test_df["Description"]]

    # Build a dictionary
    corpus = set()
    [corpus.add(word) for tokens in train_res["Tokens"] for word in tokens]
    [corpus.add(word) for tokens in test_res["Tokens"] for word in tokens]
    token_index = {word: i+1 for i, word in enumerate(corpus)}

    # Encode the text - represent each token by its index
    train_res["Sequence"] = [[token_index[token] for token in tokens] for tokens in train_res["Tokens"]]
    test_res["Sequence"] = [[token_index[token] for token in tokens] for tokens in test_res["Tokens"]]

    # Alignment
    train_res['Sequence'] = [np.pad(seq, (0, max(sequence_length_w-len(seq),0)), 'constant')[-sequence_length_w:] for seq in train_res['Sequence']]
    test_res['Sequence'] = [np.pad(seq, (0, max(sequence_length_w-len(seq),0)), 'constant')[-sequence_length_w:] for seq in test_res['Sequence']]

    vocab_len = len(token_index.values())+1
    return train_res['Sequence'], test_res['Sequence'], vocab_len

def parse_breeds(train_df, test_df, breed_labels, weights=[1, 1], pca_len=64, onehot=True):
    type1_breeds_train_df = train_df[train_df['Type'] == 1-onehot] # dogs
    type2_breeds_train_df = train_df[train_df['Type'] == 2-onehot] # cats
    type1_breeds_test_df = test_df[test_df['Type'] == 1-onehot] # dogs
    type2_breeds_test_df = test_df[test_df['Type'] == 2-onehot] # cats

    type1_breed_labels = breed_labels[breed_labels['Type']==1]
    type2_breed_labels = breed_labels[breed_labels['Type']==2]


    # Run One hot encoding on the Breeds
    type1_breed1_train_onehot = one_hot_encode(type1_breeds_train_df, 'Breed1', len(type1_breed_labels), type1_breed_labels['BreedID'])
    type1_breed2_train_onehot = one_hot_encode(type1_breeds_train_df, 'Breed2', len(type1_breed_labels), type1_breed_labels['BreedID'])

    type2_breed1_train_onehot = one_hot_encode(type2_breeds_train_df, 'Breed1', len(type2_breed_labels), type2_breed_labels['BreedID'])
    type2_breed2_train_onehot = one_hot_encode(type2_breeds_train_df, 'Breed2', len(type2_breed_labels), type2_breed_labels['BreedID'])

    type1_breed1_test_onehot = one_hot_encode(type1_breeds_test_df, 'Breed1', len(type1_breed_labels), type1_breed_labels['BreedID'])
    type1_breed2_test_onehot = one_hot_encode(type1_breeds_test_df, 'Breed2', len(type1_breed_labels), type1_breed_labels['BreedID'])

    type2_breed1_test_onehot = one_hot_encode(type2_breeds_test_df, 'Breed1', len(type2_breed_labels), type2_breed_labels['BreedID'])
    type2_breed2_test_onehot = one_hot_encode(type2_breeds_test_df, 'Breed2', len(type2_breed_labels), type2_breed_labels['BreedID'])


    # Weights for combining the one hot encodings
    type1_breeds_train_onehot = weights[0]*type1_breed1_train_onehot + weights[1]*type1_breed2_train_onehot
    type2_breeds_train_onehot = weights[0]*type2_breed1_train_onehot + weights[1]*type2_breed2_train_onehot
    
    type1_breeds_test_onehot = weights[0]*type1_breed1_test_onehot + weights[1]*type1_breed2_test_onehot
    type2_breeds_test_onehot = weights[0]*type2_breed1_test_onehot + weights[1]*type2_breed2_test_onehot

    # PCA
    pca1 = PCA(pca_len)
    pca2 = PCA(pca_len)
    
    pca1.fit(type1_breeds_train_onehot)
    pca2.fit(type2_breeds_train_onehot)

    type1_breeds_train_pca = pca1.transform(type1_breeds_train_onehot)
    type2_breeds_train_pca = pca2.transform(type2_breeds_train_onehot)

    type1_breeds_test_pca = pca1.transform(type1_breeds_test_onehot)
    type2_breeds_test_pca = pca2.transform(type2_breeds_test_onehot)

    return [type1_breeds_train_pca, type2_breeds_train_pca, type1_breeds_test_pca, type2_breeds_test_pca]