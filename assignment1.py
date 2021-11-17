# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from numpy import dot
from numpy.linalg import norm 
from sklearn.feature_extraction.text import TfidfTransformer
from itertools import combinations
from collections import defaultdict as dd
import json 
import re
import os
import seaborn as sns
from nltk.corpus import stopwords

# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'


def cosin_sim(v1, v2):
    return dot(v1, v2)/(norm(v1)*norm(v2))

def finding_max(lst):
    max_num = 0
    for score in lst:
        scores = score.split('-')
        local = 0
        for number in scores:
            if int(scores[0]) <= 99 and int(scores[1]) <= 99:
                local += int(number)
        if local > max_num:
            max_num = local
    return max_num

def count(wordlst, word):
    count_words = 0
    for letters in wordlst:
        if letters == word:
            count_words += 1
    return count_words

def task1():
    with open(datafilepath, encoding = 'UTF-8') as json_data:
        data = json.load(json_data)
    return sorted(data['teams_codes'])

    
def task2():
    with open(datafilepath, encoding = 'UTF-8') as json_data:
        data = json.load(json_data)
    team_dic = {}
    team_dic['team_code'] = [team['club_code'] for team in data['clubs']]
    team_dic['goals_scored_by_team'] = [team['goals_scored'] for team in data['clubs']]
    team_dic['goals_scored_agaisnt_team'] = [team['goals_conceded'] for team in data['clubs']]

    df = pd.DataFrame.from_dict(team_dic)
    df.sort_values(by = 'team_code', inplace = True)
    df.set_index("team_code", inplace = True)
    df.to_csv('task2.csv')
    return df
      
def task3():
    team_score = {}
    team_score['filename'] = []
    team_score['total_goals'] = []
    for file in os.listdir(articlespath):
        with open(f'data/football/{file}') as f:
            txt = f.read()
            answer = []
            max_num = -999
            answer = re.findall('\d+-\d+', txt)
            if not answer:
                max_num = 0
            if answer:
                max_num = finding_max(answer)
            team_score['filename'].append(file)
            team_score['total_goals'].append(max_num)
    df = pd.DataFrame.from_dict(team_score).sort_values(by = 'filename')
    df.set_index("filename", inplace = True)
    df.to_csv('task3.csv')
    return team_score

def task4():
    new = pd.DataFrame(task3())[['total_goals']].describe()
    IQR = (new.loc['75%'] - new.loc['25%'])* 1.5
    lower_bound = new.loc['25%'] - IQR
    upper_bound = new.loc['75%'] + IQR
    upper_bound = float(upper_bound)
    df = np.array(task3()['total_goals'])
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (12,8))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(df)
    mask = df > upper_bound
    ax.scatter([1]*len(df[mask]), df[mask], 
               marker='x', s=20, color='r', label = 'outliers')
    plt.title("Distribution of total goals per article", size = 24)
    plt.ylabel("Total number of goals", size = 14)
    plt.plot(np.linspace(0,2, 100), [upper_bound]*100, '--b', label = 'lower threshold')
    plt.plot(np.linspace(0,2, 100), [lower_bound]*100, '--g', label = 'upper threshold')
    plt.xlabel("Articles", size = 14)
    plt.xlim(0,2)
    plt.legend(fontsize = 10)
    plt.savefig('task4.png')

import re
import os
def task5():
    with open(datafilepath, encoding = 'UTF-8') as json_data:
        data = json.load(json_data)
    teams = {}
    for team in data['participating_clubs']:
        count_team = 0
        for file in os.listdir(articlespath):
            with open(f'data/football/{file}') as f:
                txt = f.read()
            answer = re.findall(team, txt)
            if answer:
                count_team += 1
        teams[team] = count_team
    df = pd.DataFrame.from_dict(teams, orient = 'index')
    df = df.rename(columns = {0: 'number_of_mentions'})
    df.index.name = 'club_name'
    plt.figure(figsize = (20, 16))
    plt.bar(np.linspace(0,30,20), df['number_of_mentions'], width = 1, color = 'red')
    plt.xticks(np.linspace(0,30,20), df.index, rotation = 90, size = 14)
    plt.title("Frequency each club is mentioned by the media", size = 30)
    plt.ylabel("Number of mentiones", size = 20)
    plt.xlabel("Club Names", size = 20)
    plt.grid(True, axis  = 'y')
    plt.tight_layout()
    plt.savefig('task5.png')
    df.to_csv("task5.csv")
    return df
    
    
def task6():
    with open(datafilepath, encoding = 'UTF-8') as json_data:
        data = json.load(json_data)
    teams = {}
    for team in data['participating_clubs']:
        count_team = 0
        for file in os.listdir(articlespath):
            with open(f'data/football/{file}') as f:
                txt = f.read()
            answer = re.findall(team, txt)
            if answer:
                count_team += 1
        teams[team] = count_team


    teams2 = {}
    clubs = data['participating_clubs']  
    for i in range(len(clubs)):
        for j in range(len(clubs)):
            mutual = 0
            for file in os.listdir(articlespath):
                with open(f'data/football/{file}') as f:
                    txt = f.read()
                answer1 = re.findall(clubs[i], txt)
                answer2 = re.findall(clubs[j], txt)
                if answer1 and answer2:
                    mutual += 1
            teams2[str(clubs[i])+' '+ str(clubs[j])] = mutual
    val = []
    for ele in teams2.values():
        val.append(ele)
    answer = []
    for i in range (len(clubs)):
        for j in range(len(clubs)):
            sim1 = teams[clubs[i]]
            sim2 = teams[clubs[j]]
            sim3 = val[20*i+j]
            if sim1 == 0 and sim2 == 0:
                answer.append(1)
            else: 
                answer.append(2*sim3/(sim1+sim2))
    plt.figure(figsize = (12,8))
    df = pd.DataFrame(np.array(answer).reshape(20,-1))
    df.columns = clubs
    df.index = clubs
    sns.heatmap(df)
    plt.title("Similarity score for each pair of clubs", size = 24)
    plt.xlabel('Club_Names', size = 20)
    plt.ylabel('Club_Names', size = 20)
    plt.savefig('task6.png')

    
def task7():
    # First part of task5
    with open(datafilepath, encoding = 'UTF-8') as json_data:
        data = json.load(json_data)
    teams = {}
    for team in data['participating_clubs']:
        count_team = 0
        for file in os.listdir(articlespath):
            with open(f'data/football/{file}') as f:
                txt = f.read()
            answer = re.findall(team, txt)
            if answer:
                count_team += 1
        teams[team] = count_team
    task5 = pd.DataFrame.from_dict(teams, orient = 'index')
    task5 = task5.rename(columns = {0: 'number_of_mentions'})
    from scipy.stats import pearsonr
    team_dic = {}
    team_dic['name'] = [team['name'] for team in data['clubs']]
    team_dic['code'] = [team['club_code'] for team in data['clubs']]
    team_dic['goals_scored_by_team'] = [team['goals_scored'] for team in data['clubs']]
    plt.figure(figsize = (12,8))
    df = pd.DataFrame.from_dict(team_dic)
    df.sort_values(by = 'name', inplace = True)
    coef = pearsonr(task5['number_of_mentions'], df['goals_scored_by_team'])[0]
    for i in range(len(df['goals_scored_by_team'])):    
        plt.scatter(task5['number_of_mentions'][i], df['goals_scored_by_team'][i])
        if i == 18:
            plt.text(task5['number_of_mentions'][i] - 1.5, task2()['goals_scored_by_team'][i] + 0.2, task2()['goals_scored_by_team'].index[i])
        else:
            plt.text(task5['number_of_mentions'][i] - 1.5, df['goals_scored_by_team'][i] - 0.4, df['code'][i])
    plt.title(f"Relationship between team's performance and team's media coverage ($p$ = {coef:.3f})", size = 20)
    plt.xlabel("Mentions on newspapers", size = 20)
    plt.ylabel("Goals_scored_by_team", size = 20)
    plt.xticks(np.arange(0,100, 10))
    plt.savefig('task7.png')
    plt.tight_layout()

def task8(filename):
    with open(filename) as f:
        txt = f.read()
    txt = re.sub("[\.\?!,:;\-_/\[\]\(\)\d*'\"]", ' ', txt)
    txt = re.sub("[\t\n]", ' ', txt)
    txt = txt.lower()
    wordList = nltk.word_tokenize(txt)
    stopWords = set(stopwords.words('english')) # Meaningless words
    filteredwords = [w for w in wordList if w not in stopWords] # Filtering
    filteredwords = [ele for ele in filteredwords if len(ele) > 1]
    return filteredwords

def task9():
    os.listdir(articlespath)
    filepaths = []
    for ele in os.listdir(articlespath):
        filepath = f'data/football/{ele}'
        filepaths.append(filepath)
    
    master_set = set()
    for file in filepaths:
        wordlist = task8(file)
        for word in wordlist:
            if word not in master_set:
                master_set.add(word)
    master_lst = []
    set_lst = {}
    for file in filepaths:
        wordlist = task8(file)
        
        indivi_lst = [count(wordlist, word) if word in wordlist else 0 for word in master_set]
        set_lst[file[-7:]] = indivi_lst
    for key in set_lst:
        master_lst.append(set_lst[key])
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(master_lst)
    for i in range(len(set_lst)):
        set_lst[list(set_lst.keys())[i]] = tfidf.toarray()[i]
    new = []
    for pair in combinations(set_lst, 2):
        score = cosin_sim(set_lst[pair[0]], set_lst[pair[1]])
        new.append((score, pair[0], pair[1]))
    lst = sorted(new, reverse = True)[:10]
    dic_task9 = dd(list)
    for ele in lst:
        dic_task9['article1'].append(ele[1])
        dic_task9['article2'].append(ele[2])
        dic_task9['similarity'].append(ele[0])
    df = pd.DataFrame.from_dict(dic_task9)
    df.set_index("article1", inplace = True)
    df.to_csv('task9.csv')


