from statistics import median

def combine_chances(*chances: float):

    average_value = sum(chances) / len(chances)
    median_value = median(chances)
    max_value = max(chances)
    min_value = min(chances)

    return {"Average": average_value, "Median": median_value, "Maximum": max_value, "Minimum": min_value}


def print_chances(comb: list[float]): 

    print(f"Šance na úspěch: {comb ['Average']:.2f}%")
    print(f"Šance na úspěch: {median_value:.2f}%")
    print(f"Šance na úspěch: {max_value:.2f}%")
    print(f"Šance na úspěch: {min_value:.2f}%")


if __name__ == "__main__":
    # Get results from combine_chances function and store them in 'combinations'
    combinations = combine_chances(0.5,0.9,0.4)

    # Return the results using return_chances function
    print_chances(combinations)
