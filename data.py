import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import re
import random

def load_data():
    df = pd.read_excel("C:\\Users\\Agnes\\Desktop\\talproject\\88milSMS_88522.ods", engine="odf")
    sentences = df.iloc[1:, 4].dropna().astype(str).tolist()

    cleaned_sentences = []
    for sentence in sentences:
        sentence = re.sub(r"<[A-Z]{3}_\d+>", "", sentence)
        sentence = re.sub(r'\d+', '<NUM>', sentence)
        sentence = sentence.lower()
        sentence = re.sub(r"(?<![a-zA-Z])-|-(?![a-zA-Z])", "", sentence)
        sentence = re.sub(r"[^\w\s\-]", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        cleaned_sentences.append(sentence)

    return cleaned_sentences

cleaned_sentences = load_data()
random.shuffle(cleaned_sentences)
split_index = int(len(cleaned_sentences) * 0.9)
train_data = cleaned_sentences[:split_index]
test_data = cleaned_sentences[split_index:]


if __name__ == "__main__":
    print(f"Total: {len(cleaned_sentences)} sentences")
    print(cleaned_sentences[:5])
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")



    