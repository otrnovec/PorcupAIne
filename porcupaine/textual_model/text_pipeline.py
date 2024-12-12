import os
import json
import time
from datetime import datetime

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV

from porcupaine.settings import *
from porcupaine.textual_model.nn_classes import MultiLSTMBinaryClassifier, MultiLayerPerceptronNN, HybridCNNLSTM
from porcupaine.textual_model.text_model import get_train_val_data

if __name__ == "__main__":
    """
    Loops over a lot of possible combinations of input data, neural network architectures and their parameters with GridSearchCV.
    Stores the results into json files.
    """

    start = time.time()

    path_to_embeddings = [DATA_DIR / "contextual_embeddings.csv", DATA_DIR / "non_contextual_embeddings.csv"]

    param_grid = {
        'lr': [0.001, 0.003, 0.01],
        'batch_size': [16, 32],
        'epochs': [3, 5, 7, 12],
    }

    for i in range(len(path_to_embeddings)):
        X_train, y_train, X_val, y_val = get_train_val_data(path_to_embeddings[i], balanced=False)

        models = {"mlp1": MultiLayerPerceptronNN(X_train.shape[1]),
                  "lstm1": MultiLSTMBinaryClassifier(X_train.shape[1]),
                  "lstm2_2layers": MultiLSTMBinaryClassifier(X_train.shape[1], lstm_layers=2, dropout_rate=0.3),
                  "lstm3_4layers": MultiLSTMBinaryClassifier(X_train.shape[1], lstm_layers=4, dropout_rate=0.4),
                  "lstm4_2layers_bigger": MultiLSTMBinaryClassifier(X_train.shape[1], lstm_layers=2, dropout_rate=0.4, lstm_hidden_size=64),
                  "lstm5_4layers_bigger": MultiLSTMBinaryClassifier(X_train.shape[1], lstm_layers=4, dropout_rate=0.4, lstm_hidden_size=64),
                  "hyb1": HybridCNNLSTM(),
                  "hyb2_32channels": HybridCNNLSTM(cnn_out_channels=32),
                  "hyb3_32channels_2lstm": HybridCNNLSTM(cnn_out_channels=32, kernel_size=4, lstm_layers=2, dropout_rate=0.4),
                  "hyb4_32channels_4lstm": HybridCNNLSTM(cnn_out_channels=32, kernel_size=4, lstm_layers=4, dropout_rate=0.5),
                  "hyb5_biglstm": HybridCNNLSTM(lstm_hidden_size=64, lstm_layers=4, dropout_rate=0.4),
                  }

        for model in models.values():
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_trained_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_val)
            report = {
                "precision_0": round(precision_score(y_val, y_pred, average="binary", pos_label=0), 4),
                "precision_1": round(precision_score(y_val, y_pred, average="binary", pos_label=1), 4),
                "recall_0": round(recall_score(y_val, y_pred, average="binary", pos_label=0), 4),
                "recall_1": round(recall_score(y_val, y_pred, average="binary", pos_label=1), 4),
                "f1_score_all": round(f1_score(y_val, y_pred), 4),
                "accuracy_all": round(accuracy_score(y_val, y_pred), 4)
            }

            json_output = {
                "embeddings": os.path.basename(path_to_embeddings[i]),
                "custom_parameters": best_model.params if hasattr(best_model, "params") else None,
                "trained_parameters": best_trained_params,
                "results": report
            }

            key = {i for i in models if models[i] == model}     # get the key by the value
            name = f"{key}.json".strip("{'").replace("'}.json", ".json")      # mess to have a nice file name
            name = "con_"+name if i == 0 else "non_"+name
            with open(RESULTS_DIR / name, "w", encoding="utf-8") as file:
                file.write(json.dumps(json_output, indent=4))
                # file.write(str(json_output))

            print(f"Successfully saved results for the {name} model.")
            now = time.time()
            print(datetime.fromtimestamp(now-start).strftime("%H:%M:%S"), "has passed since the beginning of the process.\n")
