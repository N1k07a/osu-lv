import string

rijecnik = {}

with open("song.txt", "r") as datoeka:
    for line in datoeka:
        rijeci = line.lower()
        rijeci = rijeci.split()
    
    for rijec in rijeci:
        rijec = rijec.strip(",.-/?*;:%&$#()")

        if rijec in rijecnik:
            rijecnik[rijec] += 1
        else:
            rijecnik[rijec] = 1

    jed_rijecnik = {}
    for rijec,broj in rijecnik.items():
        if broj == 1:
            jed_rijecnik[rijec] = 1

    print(len(jed_rijecnik))
    print(jed_rijecnik)