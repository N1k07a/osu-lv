brojevi = []

while True:
    broj = input("Unesi broj ili Done ")
    
    if broj.lower() == "done":
        break

    try:
        broj = float(broj)
        brojevi.append(broj)
    except:
        print("Unos nije broj")

print(len(brojevi))
print(sum(brojevi)/len(brojevi))
print(min(brojevi))
print(max(brojevi))
brojevi.sort()
print(brojevi)