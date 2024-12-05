from statistics import median

import pandas as pd

from porcupaine.settings import *
from porcupaine.textual_model.text_model import demo
from porcupaine.numerical_model.process_eval_new_numerical_data import predict_project_success


def compute_porcupaine_score(name: str, description: str, public_interest: str, district: str, category: str, budget: int) -> float:
    """
    Args:
        parameters of the project used to compute the porcupAIne score

    Returns (float): The porcupAIne score, the chance of the project to be successful.
                     Displays in the app after form submission.
    """
    num_inputs = pd.DataFrame({
        'category': category,
        'district': district,
        'budget': budget
    }, index=[0])
    model_path = MODELS_DIR / "numerical_logistic_regression_model.pkl"
    num_score = predict_project_success(num_inputs, model_path)

    text_score = demo(name, description, public_interest)[0][1]     # the function returns the np.array of probabilities for each class, we want the proba of the class 1

    combined_chances = combine_chances(text_score, num_score)

    return round(combined_chances["average"]*100, 2)


def combine_chances(*chances: float):

    # The following function calculates and returns the desired combinations of the provided chances

    average_value = sum(chances) / len(chances)
    median_value = median(chances)
    max_value = max(chances)
    min_value = min(chances)

    return {"average": average_value, "median": median_value, "maximum": max_value, "minimum": min_value}


def print_chances(comb: dict[str, float]):

    # The function accesses the values from combine_chances and prints them

    print(f"Šance na úspěch (avg): {comb ['average']:.2f}%")
    print(f"Šance na úspěch (med): {comb ['median']:.2f}%")
    print(f"Šance na úspěch (max): {comb ['maximum']:.2f}%")
    print(f"Šance na úspěch (min): {comb ['minimum']:.2f}%")


if __name__ == "__main__":
    # Get results from combine_chances function and store them in 'combinations'
    # Example values in brackets
    # combinations = combine_chances(0.5,0.9,0.4,0.99,0.25)

    # Print the results using print_chances function
    # print_chances(combinations)

    name = "Jan"
    description = "Tohle je krátký popis projektu."
    public_interest = "Tohle je krátký popis přínosu projektu."
    district = "Brno-Bohunice"
    category = "Senioři"
    budget = 10000
    print(compute_porcupaine_score(name, description, public_interest, district, category, budget))
