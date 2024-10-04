#short script that asks the user for their name and how they are feeling today
print("Enter your name:")

#input() is a function that takes user input from the terminal
terminal_input = input()
print("Hello, " + terminal_input +
      ", how are you today? (Please answer with 'good' or 'bad' only.)")
terminal_input = input()

if terminal_input == 'good':
    print("I'm glad to hear that.")
elif terminal_input == 'bad':
    print("I'm sorry to hear that.")
else:
    print("I'm sorry, I didn't understand that.")

print("Goodbye!")
