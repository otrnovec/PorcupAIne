import random

# Vygenerování seznamu náhodných čísel od 0 do 99. Velikost seznamu může být mezi 15 a 30 čísli. Použití bubble sort algoritmu k seřazení seznamu sestupně.

num_count = random.randint(15, 30)  # Definujeme náhodně velikost seznamu.
# Plníme pozice v seznamu náhodně generovanými čísly a definujeme tak proměnnou random_number jako náš seznam.
random_numbers = [random.randint(0, 99) for _ in range(num_count)] # Náhodné číslo mezi 0 a 99 pro každou pozici v seznamu.
print("Original List:", random_numbers)

# Definice algoritmu Bubble sort.
def bubble_sort_descending(arr):
    n = len(arr)
    
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] < arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]  # přehodí sousední prvky pokud nejsou v žádoucím pořadí (větší, menší).
    return arr

# Aplikace algoritmu na vygenerovaný seznam.
sorted_list = bubble_sort_descending(random_numbers)

# Výsledek.
print("\nSorted List:", sorted_list)
