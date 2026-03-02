def total_euro(radni_sat, placeno_h):
    return float(radni_sat)*float(placeno_h)

radni_sat = input("Radni sat: ")
radni_sat = radni_sat.strip("h")
placeno_h = float(input("eura/h: "))
zarada = total_euro(radni_sat, placeno_h)
print(f"Ukupno: {float(radni_sat) * float(placeno_h)} eura")
print(f"Ukupno: {zarada} eura")

