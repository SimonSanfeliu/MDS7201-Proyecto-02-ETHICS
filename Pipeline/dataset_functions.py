from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from clean_functions import clean
import pandas as pd

# Importamos el modelo de Huggings Face para resumir
model_name = "Falconsai/text_summarization"
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

###########################################################################
def summarization(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(input_ids, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def get_team_id(user_id, dataframe_in):
    df_user_id = dataframe_in[dataframe_in.user_id==user_id]
    mask = ~df_user_id['team_id'].isna()
    team_id = df_user_id[mask]['team_id'].values[0]
    return team_id

def fill_team(dataframe_in):
    dataframe_out = dataframe_in.copy()
    without_team = []
    
    for index, row in dataframe_out.iterrows():
        try:
            dataframe_out.at[index, 'team_id'] = get_team_id(row.user_id, dataframe_out)
        except:
            # Estudiantes que no tienen registrado un team_id
            without_team.append(row.user_id)
            
    return dataframe_out, without_team

def get_inds(dataframe_in):
    dataframe_out, without_team = fill_team(dataframe_in)
    dataframe_out = dataframe_out.dropna(subset=['team_id'])
    return dataframe_out

def get_chats(dataframe_in, summarize, save_summary, caso, diferencial):    
    team, chat = [], []
    team_dataframe = dataframe_in.groupby('team_id')

    # Juntamos los mensajes por team_id
    for team_id, dataframe in team_dataframe:
        chat_messages = [f"{row['message']}" for _, row in dataframe.iterrows()]

        team.append(team_id)
        chat.append(" ".join(chat_messages))

    # Obtenemos dataframe con chats y grupos
    data = {
        'team_id': team,
        'message': chat,
    }
    dataframe_out = pd.DataFrame(data)
    
    # Resumimos
    if summarize:
        dataframe_out['summary'] = dataframe_out['message'].apply(summarization)
    
    else:
        dataframe_out['summary'] = dataframe_out['message']

    if save_summary:
        dataframe_out.to_csv(f'chat_resumido_{caso}_{diferencial}.csv', index=False)

    return dataframe_out

def generate_dataset(caso, diferencial, summarize=False, save_clean=False, save_summary=False, save_dataset=False):
    df, df_chat = clean(caso, diferencial, save_clean)
    
    # Procesamos los dataframes
    df = get_inds(df)
    df_chat = get_chats(df_chat, summarize, save_summary, caso, diferencial)
    
    # Combinamos dataframes
    combined_dataframe = pd.merge(df, df_chat, on='team_id', how='left')
    
    # Separamos en estudiantes con chats y sin chats
    users_with_chat = combined_dataframe[~combined_dataframe['message'].isna()]
    user_without_chat = combined_dataframe[combined_dataframe['message'].isna()]

    print(f'\nEstudiantes con chat:', len(users_with_chat) // 3)
    print(f'Estudiantes sin chat:', len(user_without_chat) // 3)

    # Generamos dataset con los estudiantes con chat
    pivot = users_with_chat.pivot(index='user_id', columns='etapa', values=['sel', 'comment'])
    
    # Cambiamos nombres a sel_Ind1, comment_Ind1, etc
    pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]
    
    # Eliminamos user_id
    pivot.reset_index(inplace=True)
    
    # Agregamos user_id, gender, summary
    pivot = pd.merge(pivot, users_with_chat[['user_id', 'gender', 'summary']], on='user_id')
    
    # Eliminamos posibles duplicados
    pivot = pivot.drop_duplicates(ignore_index=True)
    
    # Nos quedamos con las columnas de interés
    pivot = pivot[['user_id', 'gender', 'sel_Ind1', 'sel_Grup','sel_Ind2', 'comment_Ind1', 'comment_Grup','comment_Ind2', 'summary']]
    
    # Eliminamos user_id
    pivot = pivot.drop(columns=['user_id'])

    if save_dataset:
        pivot.to_csv(f'dataset_{caso}_{diferencial}.csv', index=False)
    
    return pivot