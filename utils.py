import contractions
import re
from decimal import Decimal
from num2words import num2words
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud


nlp = spacy.load("en_core_web_sm")

custom_stop_words = ['and', 'the', 'in', 'to', 'of', 'for', 'on', 'with', 'as', 'an', 'by', 'at']

def clean_text(text):
    text = contractions.fix(text)
    currency_symbols = {
        '$': 'dollars ',
        '€': 'euros ',
        '£': 'sterling ',
        '-': ' '
    }
    for symbol, word in currency_symbols.items():
        text = text.replace(symbol, word)

    pattern = r'[\(\[]?-?\d{1,3}(?:,\d{3})*(?:[.,]\d+)?(?:e-?\d+)?%?[\)\]]?(?![\d])'

    def convert_to_words(match):
        try:
            number = Decimal(match.group())
            return num2words(number)  
        except Exception as e:
            return match.group()

    text = re.sub(pattern, convert_to_words, text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'[^\w\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower() 
    text = ' '.join(word for word in re.findall(r'\b\w+\b', text) if word.lower() not in custom_stop_words)

    return text

def lemmatize_text(text):
  doc = nlp(text)
  lemmatized_text = ' '.join([token.lemma_ for token in doc])
  return lemmatized_text

def spacy_sentences(nlp, text):
    sentences = []
    for sent in nlp(text).sents:
        sentences.append(sent)
    return sentences

def explore(token, children=None, level=0, order=None):
    if children is None:
        children = []
    if order is None:
        order = token.idx
    for child in token.children:
        children.append((child, level, child.idx < order))
        explore(child, children=children, level=level+1, order=order)
    return children


def search_adjectives(nlp_text, nouns=None):
    nouns_map = dict([(x, []) for x in nlp_text if x.pos_ in ['NOUN', 'PROPN']])
    if nouns is None:
        nouns = nouns_map.keys()
    else:
        pass
    for noun in nouns:
        subtree = explore(noun)
        subnouns = [(x, l) for x, l, _ in subtree if x.pos_ in ['NOUN', 'PROPN']]
        for token, level, left in subtree:
            if token.pos_ == 'ADJ' and len([(n, l) for n, l in subnouns if l < level]) == 0:
                try:
                    nouns_map[noun].append(token)
                except KeyError:
                    pass
    return nouns_map


def verb_adjectives(text, adjective_map, be_only=True):
    if be_only:
        verbs = [x for x in text if x.lemma_ == 'be']
    else:
        verbs = [x for x in text if x.pos_ in {'AUX', 'VERB'}]
    for verb in verbs:
        subtokens = explore(verb)
        subject = [(x) for x, level, left in subtokens if left and x.dep_ == 'nsubj']
        if len(subject) > 0:
            subj = subject[0]
            for candidate, level, left in subtokens:
                if not left:
                    if candidate.pos_ == 'ADJ' and level == 0:
                        try:
                            adjective_map[subj].append(candidate)
                        except KeyError:
                            pass
                    elif candidate.dep_ in ['dobj', 'attr', 'conj']:
                        adj = search_adjectives(text, nouns=[candidate])
                        try:
                            adjective_map[subj] += adj[candidate]
                        except KeyError:
                            pass
                    else:
                        pass


def find_first_not_phrase(text):
    doc = nlp(text)
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if "not" in sent_text:
            return sent_text
    return None


def find_aspect_lines(data, features, target_class, aspects):
    aspect_lines = {}
    selected_reviews = set()

    for feature in features:
        for index, row in data.iterrows():
            if row['rating'] == target_class and all(token in row['review'] for token in feature):
                if row['category'] in aspects:
                    aspect = row['category']
                    if aspect not in aspect_lines:
                        if row['review'] not in selected_reviews:
                            aspect_lines[aspect] = row
                            selected_reviews.add(row['review'])

        if len(aspect_lines) == len(aspects):
            break

    return aspect_lines


import numpy as np

wes_palette = ['#ABDDDE', '#D69C4E', '#036C9A', '#E6A0C4', '#ECCBAE']

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return np.random.choice(wes_palette)

def plot_wordcloud_and_feature_importance(top_features_class, class_number, custom_palette=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')    
    top_features_text = ' '.join([str(feature) for feature in top_features_class.index])
    
    # Word cloud with custom color function
    wordcloud = WordCloud(width=400, height=400, background_color='white', color_func=color_func).generate(top_features_text)

    axes[0].imshow(wordcloud, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title(f'Word Cloud of Top 10 Contributing Words for Class {class_number}')
    
    if custom_palette:
        top_features_class.plot(kind='bar', y=str(class_number), ax=axes[1], color=custom_palette)
    else:
        top_features_class.plot(kind='bar', y=str(class_number), ax=axes[1], color='blue')
    
    axes[1].set_xlabel('Feature')
    axes[1].set_ylabel('Importance Score')
    axes[1].set_title(f'Top Features for Class {class_number}')
    
    plt.tight_layout()
    image_format = 'jpg' 
    image_name = f'top_{class_number}.jpg'
    plt.savefig(image_name, format=image_format, dpi=1200)

    plt.show()
    plt.close()



