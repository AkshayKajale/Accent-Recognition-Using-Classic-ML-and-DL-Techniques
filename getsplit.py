import pandas as pd
from sklearn.model_selection import train_test_split

def filter_df(df):  
    arabic = df[df.native_language == 'arabic']
    mandarin = df[df.native_language == 'mandarin']
    english = df[df.native_language == 'english']
    
    df = arabic.append(english)
    df = df.append(mandarin)  
    
    df.native_language = pd.factorize(df.native_language)[0]
    
    return df

def get_split(df,test_size=0.2):
    return train_test_split(df['language_num'],df['native_language'],test_size=test_size,random_state=1234)

if __name__ == '__main__':
    csv_file = "bio_metadata.csv"
    df = pd.read_csv(csv_file)
    filter_df = filter_df(df)
    print(filter_df)