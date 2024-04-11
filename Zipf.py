"""
Verifying Zipf's Law through Chinese Language Corpora
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import jieba

# Return all absolute paths of files in the directory
def getFileNames(file_dir):
    List_txt = []
    for root, dirs, files in os.walk(file_dir):         # ergodic( root: absolute paths; dirs: subfolders name; files: files name)
        for file in files:
            if os.path.splitext(file)[1] == '.txt':     # [1]:Extension; [0]: name
                List_txt.append(os.path.join(root, file))
    return List_txt


# Get all the words and store them in a dictionary format
def getChineseTerms(files_path):
    tmp = {}
    for i in range(len(files_path)):        # Number of files
        filename = files_path[i]
        with open(filename, 'rb') as f:     # 'rb': Binary read
            mytext = f.read().decode('gb18030')     # gb18030: Chinese Encode
            mytext = " ".join(jieba.cut(mytext))    # Chinese word segmentation -> get specific words (separate with blanks)
            myword = [i for i in mytext.strip().split() if len(i) >= 2]
            # strip(): Default deletion of spaces and line breaks at the beginning and end of the current string
            # split(): Default splitting by spaces (Space deleted)
            for j in myword:
                tmp[j] = tmp.get(j, 0) + 1
    return tmp


# Figures
def showFigures(res):
    ranks = []
    freqs = []
    for rank, value in enumerate(tmp1):  # Simultaneously obtaining indexes (default from 0) and values
        ranks.append(rank + 1)
        freqs.append(value[1])
        rank += 1

    plt.plot(ranks, freqs)
    plt.xlabel('Word frequency', fontsize=14, fontweight='bold')
    plt.ylabel('Word ranking', fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.show()
    plt.loglog(ranks, freqs)
    plt.xlabel('Word frequency(log)', fontsize=14, fontweight='bold')
    plt.ylabel('Word ranking(log)', fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    files_path = getFileNames(r'D:\Desktop\zzzzzz\2024_spring\Deep_NLP\Homework1\jyxstxtqj_downcc.com')
    tmp = getChineseTerms(files_path)
    tmp1 = sorted(tmp.items(), key=lambda x: x[1], reverse=True)        # Sort all iterable objects
    # reverse=True: Descending order
    # key=lambda x: x[1], sorting basis is lambda function. [1] is dimension
    print(tmp1)
    showFigures(tmp1)