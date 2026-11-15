
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Given Input sentence
sentence = (
    " LeBron James as the NBA player of the United States "
    "in Laker and won the NBA Final MVP in 2020."
)

# Process the sentence using spaCy
doc = nlp(sentence)

# Print the detected named entities
print("Named Entity Recognition Results:")
print("-" * 70)

for ent in doc.ents:
    print(f"Entity: {ent.text}")
    print(f"Label: {ent.label_}")
    print(f"Start Position: {ent.start_char}")
    print(f"End Position: {ent.end_char}")
    print("-" * 70)
