try:
    broj = float(input("Unesi broj(0.0 - 1.0): "))
    if broj < 0 or broj > 1:
        print("Broj nije u rasponu")
    elif broj >= 0.9:
        print("A")
    elif broj >= 0.8:
        print("B")
    elif broj >= 0.7:
        print("C")
    elif broj >= 0.6:
        print("D")
    elif broj <0.6:
        print("C")
except:
    print("Nije dobar format")