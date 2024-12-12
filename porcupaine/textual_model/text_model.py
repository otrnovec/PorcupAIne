import re
import pandas as pd
import os
import ast
import joblib

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from porcupaine.preprocessing.preprocess_data_original import split_train_val_test
from porcupaine.settings import *
from porcupaine.textual_model.basic_text_model import balance_dataset
from porcupaine.textual_model.demo_contextual_word_embeddings import generate_single_instance_embedding
from porcupaine.textual_model.nn_classes import MultiLSTMBinaryClassifier


def process_embeddings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Converts string representations of lists in specified columns of a DataFrame into actual lists,
    and then expands these lists into separate columns, with each element in the list becoming
    a new feature (column). The original columns are dropped after the expansion.
    :param df: The input DataFrame containing the columns with the embbedings in string representations of lists
    :param cols: A list of column names to be expanded into separate columns.
    :return: The DataFrame with the specified embedding columns expanded into multiple individual columns.
             Each new column represents one element from the original lists. The original embedding columns
             are removed from the resulting DataFrame.

    Example:
    If the input DataFrame `df` has a column 'project_name' with the following values:

    | project_name             | other_column |
    |--------------------------|--------------|
    | [0.1, 0.2, 0.3]          | "A"          |
    | [0.4, 0.5, 0.6]          | "B"          |

    After calling process_embeddings(df, ['project_name']), the result will be:

    | project_name_0 | project_name_1 | project_name_2 | other_column |
    |----------------|----------------|----------------|--------------|
    | 0.1            | 0.2            | 0.3            | "A"          |
    | 0.4            | 0.5            | 0.6            | "B"          |
    """
    for col in cols:
        df.loc[:, col] = df[col].apply(lambda x: ast.literal_eval(x))
        col_expanded = pd.DataFrame(df[col].tolist(), index=df.index)
        col_expanded.columns = [f"{col}_{i}" for i in range(col_expanded.shape[1])]
        df = pd.concat([df.drop(columns=[col]), col_expanded], axis=1)
    return df


def find_best_params(input_data, output_data) -> dict:
    """
        Performs a grid search to find the best hyper parameters for a pipeline consisting
        of MinMaxScaler, SelectKBest, and a RandomForestClassifier.
        :param input_data: Features for model training.
        :param output_data: Target variable for model training.
        :return Best hyperparameters found during the grid search.
    """
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('skb', SelectKBest()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    # Parameters of pipelines can be set using '__' separated parameter names:
    param_grid = {
        "skb__k": [50, 100, 500, 1000],
        "skb__score_func": [chi2, f_classif],
        "classifier__max_depth": [2, 4, 6, 20],
        "classifier__n_estimators": [20, 50, 100, 200, 500]
    }

    # n_jobs=-1 means the work is parallelized - all processors are used
    search = GridSearchCV(pipeline, param_grid, n_jobs=-1)

    search.fit(input_data, output_data)
    return search.best_params_


def get_train_val_data(csv_path: Path, balanced=True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    embedded_df = pd.read_csv(csv_path)
    preprocess_df = pd.read_csv(os.path.join(DATA_DIR, "paro_preprocessed.csv"), usecols=["year"])
    embedded_df["year"] = preprocess_df["year"]
    df_train, df_val, _ = split_train_val_test(embedded_df)

    if balanced:
        df_train = balance_dataset(df_train)

    # embedded_dataset.csv has different structure and needs a little bit more preprocessing
    if re.match(r"embedded_dataset\.csv", str(csv_path)):
        cols = ["project_name", "project_description", "public_interest"]
        X_train = process_embeddings(df_train[cols], cols)
        X_val = process_embeddings(df_val[cols], cols)
    else:
        X_train = df_train.drop(["status", "year"], axis="columns")
        X_val = df_val.drop(["status", "year"], axis="columns")

    y_train = df_train["status"]
    y_val = df_val["status"]

    return X_train, y_train, X_val, y_val


def train_and_save_the_best_model():
    path_to_embeddings = DATA_DIR / "contextual_embeddings.csv"
    X_train, y_train, X_val, y_val = get_train_val_data(path_to_embeddings, balanced=False)
    model = MultiLSTMBinaryClassifier(X_train.shape[1], batch_size=16, epochs=12, lr=0.001)
    model.fit(X_train, y_train)
    joblib.dump(model, MODELS_DIR / 'textual_model.pkl')

# train_and_save_the_best_model()

def predict_text(project_name, project_description, public_interest):
    # predicts on new data
    loaded_pipeline = joblib.load(BASE_DIR / 'model_pipeline.pkl')
    embedding = generate_single_instance_embedding(project_name, project_description, public_interest)
    # print(embedding)
    y_pred = loaded_pipeline.predict_proba(embedding.reshape(1, -1))
    return y_pred


if __name__ == "__main__":
    # path_to_embeddings = DATA_DIR / "contextual_embeddings.csv"
    # X_train, y_train, X_val, y_val = get_train_val_data(path_to_embeddings, balanced=True)
    
    # # print(len(X_val.columns))
    # # print(find_best_params(X_train, y_train))
    
    # # using best params found by GridSearchCV
    # pipeline = Pipeline([
    #         # if we use MinMax instead of Standard scaler there is no problem with negative values while using chi2
    #         # however it performs differently (don't know exactly how...), so be careful:)
    #         ('scaler', MinMaxScaler()),
    #         # ('scaler', StandardScaler()),
    #         ('skb', SelectKBest(score_func=f_classif, k=500)),
    #         ('classifier', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=30))
    # ])
    # pipeline.fit(X_train, y_train)
    
    # joblib.dump(pipeline, BASE_DIR/ 'model_pipeline.pkl')     # saves model
    
    # loaded_pipeline = joblib.load(BASE_DIR / 'model_pipeline.pkl')
    
    # y_pred = loaded_pipeline.predict(X_val)
    
    # print(classification_report(y_val, y_pred))
    # ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
    # plt.show()

    print(predict_text("Odpočinkové lavičky – Brno-Vinohrady 2",
         "Sídlištěm Brno-Vinohrady procházejí dva souběžné dvoupruhové chodníky pro pěší, jejichž povrch byl v minulých letech upravován, ale přesto mají závažný nedostatek - po celé jejich délce téměř schází odpočinkové lavičky pro možnost odpočinku na nich. Stav a vybavenost sídliště Brno-Vinohrady se v uplynulých letech postupně vylepšovaly, Některé jeho nedostatky nejsou vzhledem k poloze vůči městu dost dobře řešitelné nebo jsou značně nákladné (např. napojení cyklostezek na městskou síť nebo parkovací místa), jiné řešení mají, ale nebyla jim věnována žádoucí pozornost. Mezi tyto relativně snadno řešitelné záležitosti patří zejména vybavenost sídliště odpočinkovými lavičkami. Ty jsou na sídlišti sice poměrně hustě rozmístěny, ale bohužel pouze v odpočinkových plochách v parcích, na dětských hřištích apod., ale téměř vůbec ne na dvou hlavních pěších trasách, procházejících sídlištěm. Přitom po těchto chodnících musí projít v podstatě všichni obyvatelé sídliště, jdoucí k lékaři nebo do lékárny, na nákup nebo do jiného zařízení služeb na sídlišti, na úřad či na zastávku MHD apod. Zdravému člověku to nepřijde, ale chodci, mající jakékoli zdravotní potíže, omezující jejich pohyblivost (nemusí jí jenom o problém s nohami, ale i slabší srdce, problém s dechem apod.), v podstatě nemají možnost si (bez odbočení z chodníku do parku nebo na hřiště, tj. mimo směr chůze) na chvíli sednout a odpočinout si před dalším pokračováním ve své chůzi. Doplnění stávajících nových i starých chodníků lavičkami ve všech vhodných místech (mimo profil chodníků, na okraji travnatých ploch) by bylo velmi přívětivým a ohleduplným opatřením vůči všem obyvatelům a návštěvníkům sídliště, nejen vůči seniorům a osobám se zdravotními potížemi. Připojené fotografie dokumentují stav v roce 2021, kdy byl podán první návrh v rámci aktivity Dáme na vás - od té doby k žádné změně k lepšímu nedošlo (v roce 2020 naopak byly lavičky na pěší zóně u parku na Pálavském náměstí odstraněny). Snad je z fotografií zřejmý aktuální stav i možnosti jeho relativně snadného i nenákladného a přitom potřebného řešení.",
         "Předpokládanými uživateli projektu mohou být všichni obyvatelé a návštěvnící sídliště, především pak osoby, mající zdravotní potíže, omezující jejich pohyblivost či dosažitelnou pochůzkovou vzdálenost (bez možnosti krátkodobého odpočinku) při potřebných cestách mezi bydlištěm a zařízeními (obecnou vybaveností) na sídlišti.",
    ))
