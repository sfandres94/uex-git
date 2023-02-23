# Custom function to read the two numbers.
def read_numbers():
    num1 = float(input('Introduce el primer numero: '))
    num2 = float(input('Introduce el segundo numero: '))
    return num1, num2

# Simple menu.
print('Bienbenido a la calculadora!')
print('Seleccione operacion:')
print('    1. Suma')
print('    2. Resta')
print('    3. Multiplicacion')
print('    4. Division')
print('    5. Salir')

# Main loop.
while True:

    choice = input('\nIntroduce tu eleccion (1-5): ')

    if choice in ('1', '2', '3', '4', '5'):
        if choice == '1':
            num1, num2 = read_numbers()
            result = num1 + num2
            print(f'Resultado: {result}')
        
        elif choice == '2':
            num1, num2 = read_numbers()
            result = num1 - num2
            print(f'Resultado: {result}')
        
        elif choice == '3':
            num1, num2 = read_numbers()
            result = num1 * num2
            print(f'Resultado: {result}')
        
        elif choice == '4':
            try:
                num1, num2 = read_numbers()
                result = num1 / num2
                print(f'Resultado: {result}')
            except ZeroDivisionError:
                print("Error: division entre 0")
        
        elif choice == '5':
            break
    else:
        print('Entrada invalida')
