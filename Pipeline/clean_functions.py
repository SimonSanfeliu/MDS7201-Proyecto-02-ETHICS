import pandas as pd
import numpy as np
import re
import nltk
from nltk import word_tokenize

nltk.download('punkt')
with open('spanish.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
spanish_stopwords = [line.strip() for line in lines]

###########################################################################
def get_dataframe(caso, diferencial):
    # Leemos datasets
    df = pd.read_csv(f'{caso}.csv')
    df_chat = pd.read_csv(f'{caso}_chat.csv')
    
    # Filtramos por diferencial
    df = df[df.df == diferencial]
    
    # Obtenemos los team_id posibles
    team_id_values = df.team_id.unique()
    
    # Filtramos el chat con solo los team_id posibles
    df_chat = df_chat[df_chat.team_id.isin(team_id_values)]
    
    return df, df_chat

# Solo para caso julieta
def delete_extra_phases(dataframe_in):
    titles = [
        'En el último control realizado', # fase1
        'Si la mantención de la beca', # fase2
        'Julieta sabe' # fase3
    ]
    
    mask = dataframe_in.title.str.contains('|'.join(titles))
    dataframe_out = dataframe_in[~mask]
    
    return dataframe_out.reset_index(drop=True)

# Solo para caso julieta
def std_titles(dataframe_in):
    dataframe_out = dataframe_in.copy()
    
    standard = [
        '[Ind1] Julieta en esta situación a la que se ve enfrentada en el control debiera:',
        '[Grup] Julieta en esta situación a la que se ve enfrentada en el control debiera:', 
        '[Ind2] Después de la discusión en grupo, ¿cómo definirías tu posición ahora?'
    ]
    
    titles = []
    etapas = []
    
    for i, title in enumerate(dataframe_out.title):
        if 'Julieta en esta situación' in title and np.isnan(dataframe_out.loc[i, 'team_id']):
            titles.append(standard[0])
            etapas.append('Ind1')
        elif 'Julieta en esta situación' in title:
            titles.append(standard[1])
            etapas.append('Grup')
        elif 'Después de la discusión en grupo' in title:
            titles.append(standard[2])
            etapas.append('Ind2')
        else:
            titles.append(title)
            etapas.append(0)
            
    dataframe_out['title'] = titles
    dataframe_out['etapa'] = etapas
    return dataframe_out

def remove_stopwords(text):
    words = word_tokenize(text, language='spanish')
    filtered_words = [word for word in words if word.lower() not in spanish_stopwords]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_column_name(type_):
    if type_ == 'ind':
        col = 'comment'
    elif type_ == 'chat':
        col = 'message'
    else:
        raise ValueError()
        
    return col

def drop_duplicates(dataframe_in, type_):
    dataframe_out = dataframe_in.copy()
    extra_columns = ['etapa'] if type_ == 'ind' else []
    
    # Ordenamos mensajes por tiempo
    dataframe_out = dataframe_out.sort_values('time')
    
    # Eliminamos repeticiones
    dataframe_out = dataframe_out.drop_duplicates(subset=['user_id', 'seccion'] + extra_columns)
    
    return dataframe_out

def drop_na(dataframe_in, type_):
    dataframe_out = dataframe_in.copy()
    col = get_column_name(type_)
    
    # Eliminamos nulos 
    dataframe_out = dataframe_out.dropna(subset=col)
    
    return dataframe_out

def clean_text(dataframe_in, type_):
    dataframe_out = dataframe_in.copy()
    col = get_column_name(type_)
    
    # Pasamos texto a minúscula
    dataframe_out[col] = dataframe_out[col].str.lower()
    
    # Removemos signos de puntución
    dataframe_out[col] = dataframe_out[col].apply(remove_punctuation)
    
    # Removemos stopwords
    dataframe_out[col] = dataframe_out[col].apply(remove_stopwords)
    
    return dataframe_out 

def clean_data(dataframe_in, type_):
    dataframe_out = dataframe_in.copy()
    col = get_column_name(type_)
    
    # Eliminamos duplicados
    dataframe_out = drop_duplicates(dataframe_out, type_)
    
    # Eliminamos nulos 
    dataframe_out = drop_na(dataframe_out, type_)
    
    return dataframe_out

def valid_data(dataframe_in):
    dataframe_out = dataframe_in.copy()
    
    # Separamos en secciones
    section_dataframe = dataframe_out.groupby('seccion')
    dict_df = {section: dataframe for section, dataframe in section_dataframe}
    
    print("\nSección | Etapa | Cantidad antes de limpieza")
    for section in dict_df.keys():
        value, counts = np.unique(dict_df[section].groupby('user_id').count().id, return_counts=True)
        print(f"Sección {section}:", value, counts)
        
    for section in dict_df.keys():
        mask = dict_df[section].groupby('user_id').id.transform('count') == 3
        dict_df[section] = dict_df[section][mask]
    
    print("\nSección | Etapa | Cantidad después de limpieza")
    for section in dict_df.keys():
        value, counts = np.unique(dict_df[section].groupby('user_id').count().id, return_counts=True)
        print(f"Sección {section}:", value, counts)
    
    # Juntamos todo el dataframe
    dataframe_out = pd.concat(dict_df.values(), ignore_index=True)
    
    return dataframe_out

def clean(caso, diferencial, save=False):
    df, df_chat = get_dataframe(caso, diferencial)

    if caso == 'julieta':
        df = delete_extra_phases(df)
        df = std_titles(df)
    
    print("Registro antes de limpieza:")
    print("\tIndividual:", len(df))
    print("\tChat:", len(df_chat))
    
    # limpieza de texto
    df = clean_text(df, 'ind')
    df_chat = clean_text(df_chat, 'chat')
    
    # Limpieza de datos
    print("\nSeciones disponibles:", df.seccion.unique())
    df = clean_data(df, 'ind')
    df_chat = clean_data(df_chat, 'chat')
    
    # Eliminación de estudiantes sin las tres etapas
    df = valid_data(df)
    
    # Eliminación de chat sin team_id
    df_chat = df_chat[~df_chat.team_id.isna()]
    
    print("\nRegistro después de limpieza:")
    print("\tIndividual:", len(df))
    print("\tChat:", len(df_chat))
    
    # Guardamos los dataframes
    if save:
        df.to_csv(f'clean_{caso}_{diferencial}.csv', index=False)
        df_chat.to_csv(f'clean_{caso}_{diferencial}_chat.csv', index=False)
    
    return df, df_chat