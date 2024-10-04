import random

def greatings_to_user():
    name = input("Jak se jmenujete?")
    print(f"Ahoj, {name}! Vítej ve 21řádkovém Python skriptu.")

def comparison_of_two_numbers():
    users_num = int(input("Jaké si myslíš číslo?"))
    my_num = random.randint(1, 100)
    print(f"Já si myslím: {my_num}")
    if users_num > my_num:
        diffrence = users_num - my_num
        print(f"Vaše číslo je vetší o {diffrence}")
    elif users_num < my_num:
        diffrence = my_num - users_num
        print(f"Moje číslo je vetší o {diffrence}")
    else:
        print("Myslíme si stejné číslo!")

greatings_to_user()
comparison_of_two_numbers()