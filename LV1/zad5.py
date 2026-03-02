ham_words_total = 0
ham_count = 0

spam_words_total = 0
spam_count = 0
spam_with_exclamation = 0


with open("SMSSpamCollection.txt", "r", encoding="utf-8") as file:
    for line in file:
       parts = line.strip()
       parts = parts.split(maxsplit=1)
            
        
       label = parts[0].lower()
       text = parts[1]
 
       words = text.split()
       num_words = len(words)
       
       if label == "ham":
            ham_count += 1            
            ham_words_total += num_words
       elif label == "spam":
            spam_count += 1            
            spam_words_total += num_words 
                
            if text.endswith("!"):
                spam_with_exclamation += 1


avg_ham = ham_words_total / ham_count if ham_count > 0 else 0
avg_spam = spam_words_total / spam_count if spam_count > 0 else 0

print(f"Prosječan broj riječi u 'ham' porukama: {avg_ham:.2f}")
print(f"Prosječan broj riječi u 'spam' porukama: {avg_spam:.2f}")
    
print(f"Broj spam poruka koje završavaju uskličnikom: {spam_with_exclamation}")
