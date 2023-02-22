# Custom function to read the two numbers.
def read_numbers():
    num1 = float(input('Enter first number: '))
    num2 = float(input('Enter second number: '))
    return num1, num2

# Simple menu.
print('Welcome to the calculator!')
print('Please choose an operation:')
print('    1. Addition')
print('    2. Subtraction')
print('    3. Multiplication')
print('    4. Division')
print('    5. Exit')

# Main loop.
while True:

    choice = input('\nEnter your choice (1-5): ')

    if choice in ('1', '2', '3', '4', '5'):
        if choice == '1':
            num1, num2 = read_numbers()
            result = num1 + num2
            print(f'Result: {result}')
        
        elif choice == '2':
            num1, num2 = read_numbers()
            result = num1 - num2
            print(f'Result: {result}')
        
        elif choice == '3':
            num1, num2 = read_numbers()
            result = num1 * num2
            print(f'Result: {result}')
        
        elif choice == '4':
            try:
                num1, num2 = read_numbers()
                result = num1 / num2
                print(f'Result: {result}')
            except ZeroDivisionError:
                print("Error: division by zero")
        
        elif choice == '5':
            break
    else:
        print('Invalid Input')