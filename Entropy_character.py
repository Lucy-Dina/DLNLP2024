"""
Calculating the Average Information Entropy of Chinese by character basis
"""
import os
import jieba
import math
import time
import re
import logging
from collections import Counter

# ========= Preprocess =========
# Get the path to all novel files
def getFileNames(file_dir):
    List_txt = []
    for root, dirs, files in os.walk(file_dir):         # ergodic( root: absolute paths; dirs: subfolders name; files: files name)
        for file in files:
            if os.path.splitext(file)[1] == '.txt':     # [1]:Extension; [0]: name
                List_txt.append(os.path.join(root, file))
    return List_txt

path_list = getFileNames(r"D:\Desktop\zzzzzz\2024_spring\Deep_NLP\Homework1\jyxstxtqj_downcc.com")

# Storing corpus
corpus = []
for path in path_list:
    with open(path, "r", encoding="ANSI") as file:
        text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in file][3:]
        # \u3000: Hexadecimal Unicode encoding of full width spaces; \t: Tab
        # [3:]: takes the list created in the previous step and excludes the first three elements (rows) from the list
        corpus += text

# Regex process (replace and count)
regex_str = ".*?([^\u4E00-\u9FA5]).*?"
Non_CN = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
#Non_CN = u'[^a-zA-Z0-9\u4e00-\u9fa5]'
symbol = []
for j in range(len(corpus)):
    corpus[j] = re.sub(Non_CN, "", corpus[j])      # Search and replace parts of a string that match regex expressions
    symbol += re.findall(regex_str, corpus[j])      # Returns all strings in the string that match the pattern (in the form of an array)
count_ = Counter(symbol)        # A dictionary subclass used to calculate the number of occurrences of an element
count_symbol = count_.most_common()     # Calculate the frequency of elements appearing in the list (return a ranked tuple list)

# Indentify and remove noise
noise_symbol = []
for eve_tuple in count_symbol:
    if eve_tuple[1] < 200:      # eve_tuple[0]: elements; eve_tuple[1]: frequency
        noise_symbol.append(eve_tuple[0])
noise_number = 0
for line in corpus:
    for noise in noise_symbol:
        line.replace(noise, "")
        noise_number += 1

print("Completed noise data replacement points：", noise_number)
print("Substituted noise symbols：")
for i in range(len(noise_symbol)):
    print(noise_symbol[i], end=" ")
    if i % 50 == 0:
        print()

# Generate preprocessed text
with open("Preprocessed_text.txt", "w", encoding="utf-8") as f:
    for line in corpus:
        if len(line) > 1:
            print(line, file=f)
# Generate final corpus
with open("Preprocessed_text.txt", "r", encoding="utf-8") as f:
    corpus = [eve.strip("\n") for eve in f]

# ========= Calculate the Average Information Entropy of Chinese =========
# 1-gram
# Word frequency count
token = []
for para in corpus:
    token += [char for char in para]        # word segmentation by character
token_num = len(token)
ct1 = Counter(token)
vocab1 = ct1.most_common()

# Calculate entropy
entropy_1gram = sum([-(eve[1]/token_num)*math.log((eve[1]/token_num),2) for eve in vocab1])
print("Total number of words in the corpus：", token_num, " ", "The number of different words：", len(vocab1))
print("The top 10 most frequently occurring 1-gram words: ", vocab1[:10])
print("entropy_1-gram:", entropy_1gram)

# 2-gram
# Segmentation and combine 2 gram
def Combine2Gram(cutword_list):
    if len(cutword_list) == 1:
        return []
    res = []
    for i in range(len(cutword_list)-1):
        res.append(cutword_list[i] + "s" + cutword_list[i+1])       # Separate with "s"
    return res
token_2gram = []
for para in corpus:
    cutword_list = [char for char in para]
    token_2gram += Combine2Gram(cutword_list)

# Word frequency count
token_2gram_num = len(token_2gram)
ct2 = Counter(token_2gram)
vocab2 = ct2.most_common()
same_1st_word = [eve.split("s")[0] for eve in token_2gram]
# Extract the first element of each string in token_2gram (the string part before the split point "s")
assert token_2gram_num == len(same_1st_word)        #If the value is true, continue execution
ct2_1st = Counter(same_1st_word)
vocab2_1st = dict(ct2_1st.most_common())

# Calculate entropy
entropy_2gram = 0
for eve in vocab2:
    p_xy = eve[1]/token_2gram_num
    first_word = eve[0].split("s")[0]
    entropy_2gram += -p_xy*math.log(eve[1]/vocab2_1st[first_word], 2)
print("Total number of words in the corpus：", token_2gram_num, " ", "The number of different words：", len(vocab2))
print("The top 10 most frequently occurring 2-gram words: ", vocab2[:10])
print("entropy_2-gram:", entropy_2gram)

# 3-gram
# Segmentation and combine 3 gram
def Combine3Gram(cutword_list):
    if len(cutword_list) <= 2:
        return []
    res = []
    for i in range(len(cutword_list)-2):
        res.append(cutword_list[i] + cutword_list[i+1] + "s" + cutword_list[i+2] )
    return res
token_3gram = []
for para in corpus:
    cutword_list = [char for char in para]
    token_3gram += Combine3Gram(cutword_list)

# Word frequency count
token_3gram_num = len(token_3gram)
ct3 = Counter(token_3gram)
vocab3 = ct3.most_common()
same_2st_word = [eve.split("s")[0] for eve in token_3gram]
# Extract the first two elements of each string in token_2gram (the string part before the split point "s")
assert token_3gram_num == len(same_2st_word)
ct3_2st = Counter(same_2st_word)
vocab3_2st = dict(ct3_2st.most_common())

# Calculate entropy
entropy_3gram = 0
for eve in vocab3:
    p_xyz = eve[1]/token_3gram_num
    first_2word = eve[0].split("s")[0]
    entropy_3gram += -p_xyz*math.log(eve[1]/vocab3_2st[first_2word], 2)
print("Total number of words in the corpus：", token_3gram_num, " ", "The number of different words：", len(vocab3))
print("The top 10 most frequently occurring 3-gram words: ", vocab3[:10])
print("entropy_3-gram:", entropy_3gram)


