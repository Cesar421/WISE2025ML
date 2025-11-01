
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk", "pandas"])
import nltk
import pandas as pd
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from collections import Counter

class ExtractFeatures:
    # Initialize with dataframe, here I will 
    # transform data frame into a list of words and sentences
    def __init__(self, dataframe):
        self.dataframe = dataframe
        text = " ".join(dataframe['text'].astype(str))
        self.Word = nltk.word_tokenize(text)
        self.Sentences = nltk.sent_tokenize(text)
        print("Features extracted successfully. \n")
    def NumberOfSentences(self):
        return len(self.Sentences)
    def AverageWordLength(self):
        total_length = sum(len(word) for word in self.Word)
        average_length = total_length / len(self.Word) if self.Word else 0
        return average_length
    def NumberOfWords(self):
        return len(self.Word)
    def AverageWordsPerSentence(self):
        total_words = self.NumberOfWords()
        total_Sentences = self.NumberOfSentences()
        AveragWordsPersentence = total_words / total_Sentences
        return(AveragWordsPersentence)
    def MostCommonWords(self, n):
        words_lower = [word.lower() for word in self.Word]
        MostCommonWords = Counter(words_lower).most_common(n)
        return "\n".join([f"Word: '{tupla[0]}' - Frequency: {tupla[1]}" for tupla in MostCommonWords])
    def PrintInformation(self, n=10):
        print(f"1. Total Words: {self.NumberOfWords()}")
        print(f"2. Average Word Length: {self.AverageWordLength():.2f}")
        print(f"3. Total Sentences: {self.NumberOfSentences()}")
        print(f"4. Average Words Per Sentence: {self.AverageWordsPerSentence():.2f}")
        print(f"5. Most {n} commons Words: \n\n{self.MostCommonWords(n)}")

# Import TSV file
df = pd.read_csv('dataset.tsv', sep='\t')
ExtractDfFeatures = ExtractFeatures(df)
ExtractDfFeatures.PrintInformation(n=15)