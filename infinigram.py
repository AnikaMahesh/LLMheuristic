import requests
import json
import nltk
from nltk.tag import pos_tag
import numpy as np
import pandas as pd
import csv
import spacy
# Define the row of data
def download_nltk_data():
    try:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        print("Error downloading NLTK data. Please run the following commands in Python console:")
        print("import nltk")
        print("nltk.download('averaged_perceptron_tagger')")
        return False
    return True

def query_infinigram_counts(query : str):
    payload = {
        'index': 'v4_rpj_llama_s4',
        'query_type': 'count',
        'query': query,
    }
    result = requests.post('https://api.infini-gram.io/', json=payload).json()
    count = result['count']
    return count
#print(query_infinigram_counts("Tom Cruise"))  # unit test

def parse_sentence(query : str):
    # Try to use NLTK for part-of-speech tagging
    if download_nltk_data():
        try:
            # Split the sentence into words using string methods
            words = query.split()
            
            # Get part of speech tags for each word
            pos_tags = pos_tag(words)
            
            # List of pronouns to exclude
            pronouns = {'i', 'me', 'my', 'mine', 'myself',
                        'you', 'your', 'yours', 'yourself', 'yourselves',
                        'he', 'him', 'his', 'himself',
                        'she', 'her', 'hers', 'herself',
                        'it', 'its', 'itself',
                        'we', 'us', 'our', 'ours', 'ourselves',
                        'they', 'them', 'their', 'theirs', 'themselves'}
            
            # Extract nouns (NN, NNS, NNP, NNPS) that are not pronouns
            nouns = []
            for word, tag in pos_tags:
                # Check if the word is a noun (NN, NNS, NNP, NNPS) and not a pronoun
                if tag.startswith('NN') and word.lower() not in pronouns:
                    nouns.append(word)
            
            return nouns
        except Exception as e:
            print("Error using NLTK for part-of-speech tagging. Falling back to basic noun detection.")
    
    # Fallback method: simple noun detection without NLTK
    words = query.split()
    # List of common pronouns to exclude
    pronouns = {'i', 'me', 'my', 'mine', 'myself',
                'you', 'your', 'yours', 'yourself', 'yourselves',
                'he', 'him', 'his', 'himself',
                'she', 'her', 'hers', 'herself',
                'it', 'its', 'itself',
                'we', 'us', 'our', 'ours', 'ourselves',
                'they', 'them', 'their', 'theirs', 'themselves',
                'who', 'whom', 'whose', 'which', 'what', 'where', 'when', 'why', 'how'}
    
    # Simple heuristic: words that start with capital letters and aren't pronouns
    # This is a basic fallback that works for proper nouns
    #nouns = [word.lower() for word in words if word[0].isupper() and word.lower() not in pronouns]
    nouns = [word for word in words if word not in pronouns]
    
    return " ".join(nouns)

def get_nouns_using_spacy(query):
    # first run python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)
    words = [token.text for token in doc if token.pos_ in ("PROPN") and token.pos_ != "PRON"]
    maxcount = 10000000000000000000000000000000000000000000
    word = ""
    for i in words:
        count = query_infinigram_counts(i)
        if count < maxcount:
            maxcount = count
            print(word)
            word = i
    return word

def import_ds(path: str):
    """
    Import a CSV file and iterate through its rows.
    
    Args:
        path (str): Path to the CSV file
    """
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(path)
        row = ['row index', 'query', 'count']
        # Iterate through each row
        with open('results2.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
            for index, row in df.iterrows():
                problem = row['problem']
                answer = row['answer']
                #parsed = parse_sentence(" ".join([problem,answer]))
                parsed1 = get_nouns_using_spacy(problem)
                parsed = " AND ".join([parsed1,answer])
                parsed.replace("?", "")
                parsed.replace(".", "")
                parsed.replace("!", "")
                count = query_infinigram_counts(parsed)
                print(" ".join([problem,answer]), "    ", parsed, "    ",count)
                writer.writerow([index,parsed,count])
    except Exception as e:
        print(f"Error reading CSV file: {e}")

#parsed = parse_sentence("Who is Tom Cruise's mother") 
path = "simple_qa_test_set.csv"
import_ds(path)



