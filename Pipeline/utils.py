# Librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import shap 
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize  
from nltk.stem import SnowballStemmer
import re

# Stopwords (https://github.com/xiamx/node-nltk-stopwords/blob/master/data/stopwords/spanish)
with open('spanish.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

spanish_stopwords = [line.strip() for line in lines]

class StemmerTokenizer:
    def __init__(self):
        self.ss = SnowballStemmer('spanish')
    def __call__(self, text):
        text_tok = word_tokenize(text)
        text_tok = [t for t in text_tok if t not in spanish_stopwords]
        return [self.ss.stem(t) for t in text_tok]
    
class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

# PREPROCESAMIENTO DE COLUMNAS CATEGÓRICAS
def cat_cols_preprocessing(dataset, gender=False):
	categorical_columns = ['sel_Ind1', 'sel_Grup','sel_Ind2']
	if gender: genders = [dataset.gender.unique().tolist()]

	transformers = [('sel_etapa', 
		Pipeline([('ordinal_encoder', OrdinalEncoder())]), 
		categorical_columns)
	]

	if gender: transformers.append(('gender', 
		Pipeline([
			('extract', ColumnExtractor(columns=['gender'])),
			('onehot_encoder', OneHotEncoder(sparse_output=False, categories=genders))
		]), 
		['gender']))

	categorical_preprocessing = ColumnTransformer(
	    transformers=transformers,
	    remainder='drop'
	)

	categorical_pipeline = Pipeline([
	    ('categorical_preprocessing', categorical_preprocessing)
	])

	categorical_transformed_data = categorical_pipeline.fit_transform(dataset)

	if gender: 
		gender_column_names = [f'gender:{category}' for category in genders[0]]
		categorical_columns = categorical_columns+gender_column_names

	categorical_transformed_df = pd.DataFrame(categorical_transformed_data, columns=categorical_columns)

	return categorical_transformed_df, categorical_preprocessing, categorical_pipeline

# PREPROCESAMIENTO DE COLUMNAS TEXTUALES
def text_cols_preprocessing(dataset, comment_Ind2=True):
	text_columns = ['summary', 'comment_Ind1', 'comment_Grup']

	if comment_Ind2: text_columns.append('comment_Ind2')

	for col in text_columns:
	    dataset[col] = dataset[col].apply(remove_punctuation)

	bog = CountVectorizer(
	    tokenizer=StemmerTokenizer(),
	    ngram_range=(1,1),
	    token_pattern=None
	)

	if comment_Ind2:
		transformers=[
			('Ind1', bog, 'comment_Ind1'),
		    ('Grup', bog, 'comment_Grup'),
		    ('Ind2', bog, 'comment_Ind2'),
		    ('Chat', bog, 'summary'),
		]

	else:
		transformers=[
			('Ind1', bog, 'comment_Ind1'),
		    ('Grup', bog, 'comment_Grup'),
		    ('Chat', bog, 'summary'),
		]

	# Procesamos cada columna con su propio bag-of-words
	text_preprocessing = ColumnTransformer(
	    transformers=transformers,
	    remainder='drop'
	)

	text_pipeline = Pipeline([
	    ("text_preprocessing", text_preprocessing)
	])

	text_transformed_data = text_pipeline.fit_transform(dataset).toarray()

	feature_names_combined = text_pipeline.named_steps['text_preprocessing'].get_feature_names_out()
	text_transformed_df = pd.DataFrame(text_transformed_data, columns=feature_names_combined)

	return text_transformed_df, text_preprocessing, text_pipeline

# PREPROCESAMIENTO
def preprocessing(dataset, test_size=.2, gender=False, comment_Ind2=True, sel_Ind1=True):
	categorical_transformed_df, categorical_preprocessing, categorical_pipeline = cat_cols_preprocessing(dataset, gender)
	text_transformed_df, text_preprocessing, text_pipeline = text_cols_preprocessing(dataset, comment_Ind2)

	# Concatenamos las columnas categóricas y las columnas de texto
	data = pd.concat([categorical_transformed_df, text_transformed_df], axis=1)

	# Generamos los labels
	data['labels'] = np.sign(data['sel_Ind2'] - data['sel_Ind1'])
	data = data.drop(columns=['sel_Ind2'])

	if not sel_Ind1:
		data = data.drop(columns=['sel_Ind1'])

	# Separamos en train (80%) y test (20%)
	df_train, df_test, y_train, y_test = train_test_split(data, data['labels'], test_size=test_size, stratify=data['labels'], random_state=42)
	df_train, df_test = df_train.drop(columns=['labels']), df_test.drop(columns=['labels'])
	
	return df_train, df_test, y_train, y_test, categorical_preprocessing, text_preprocessing

# MODELOS
def train_models(df_train, df_test, y_train, y_test, grid_rf, grid_xgb):
	# Dummy classifier
	dummy_model = DummyClassifier()
	dummy_model.fit(df_train, y_train)

	y_pred = dummy_model.predict(df_test)
	print('Dummy classifier')
	print(classification_report(y_test, y_pred, zero_division=0.0))

	# Random Forest classifier
	rf_model = RandomForestClassifier(random_state=42)
	grid_search = GridSearchCV(estimator=rf_model, param_grid=grid_rf, cv=5, n_jobs=-1, scoring='accuracy')
	grid_search.fit(df_train, y_train)
	rf_best_model = grid_search.best_estimator_
	rf_best_model.fit(df_train, y_train)

	y_pred = rf_best_model.predict(df_test)

	print('Random Forest classifier')
	best_params = grid_search.best_params_
	print("Best Hyperparameters:", best_params)
	print(classification_report(y_test, y_pred, zero_division=0.0))

	# XG Boost classifier
	label_mapping = {-1: 0, 0: 1, 1: 2}
	y_train_xgb = y_train.map(label_mapping)

	xgb_model = XGBClassifier(random_state=42)
	grid_search = GridSearchCV(estimator=xgb_model, param_grid=grid_xgb, cv=5, n_jobs=-1, scoring='accuracy')
	grid_search.fit(df_train, y_train_xgb)
	xgb_best_model = grid_search.best_estimator_
	xgb_best_model.fit(df_train, y_train_xgb)

	label_mapping = {0: -1, 1: 0, 2: 1}
	y_pred = xgb_best_model.predict(df_test)
	vfunc = np.vectorize(lambda x: label_mapping[x])
	y_pred = vfunc(y_pred)

	print('XGBoost classifier')
	best_params = grid_search.best_params_
	print("Best Hyperparameters:", best_params)
	print(classification_report(y_test, y_pred, zero_division=0.0))

	return dummy_model, rf_best_model, xgb_best_model

def plot_importance_models(rf_best_model, xgb_best_model):
    # RandomForest classifier
    feature_importances = rf_best_model.feature_importances_
    feature_names = rf_best_model.feature_names_in_

    sorted_idx = feature_importances.argsort()[::-1]
    top_number = 20

    top_features = [feature_names[i] for i in sorted_idx[:top_number]][::-1]
    top_importances = feature_importances[sorted_idx][:top_number][::-1]

    plt.title('RandomForest classifier')
    plt.barh(range(len(top_importances)), top_importances)
    plt.yticks(range(len(top_importances)), top_features)
    plt.xlabel('Feature Importance')

    # XGBoost classifier
    types = ['weight', 'cover', 'gain']
    for t in types:
        plot_importance(xgb_best_model, title=f'XGBoost classifier - {t}', importance_type=t, grid=False, max_num_features=20)

    plt.show()

def summary_plot(model, df_train, title):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_train)

    classes = model.classes_
    labels = ['No hay cambio'] * 3
    labels[np.argmax(classes)] = 'Cambio hacia la derecha: No usar información'
    labels[np.argmin(classes)] = 'Cambio hacia la izquierda: Usar información'

    plt.title(title)
    shap.summary_plot(
        shap_values, 
        df_train, 
        plot_type='bar', 
        class_names=labels,
        show=False,
        max_display=10,
        plot_size=0.35
    )

    plt.savefig(f'xgb_summary_plot.pdf', bbox_inches='tight')
    plt.show()

def individual_explanation(idx, model, df_train, title):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_train)
    class_names = {-1: 'Class -1', 0: 'Class 0', 1: 'Class 1'}
    
    for i, class_ in enumerate(model.classes_):
        shap_explanation = shap.Explanation(
            values=shap_values[i][idx, :], 
            base_values=explainer.expected_value[i],
            data=df_train.iloc[idx, :]
        )
        plt.title(title + f' - Clase: {class_}')
        shap.plots.waterfall(shap_explanation, max_display=17, show=False)
        plt.show()