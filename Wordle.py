#This is a Wordle solver

import random
import matplotlib.pyplot as plt
import numpy as np
import math

"""We take the results of the last guesses and use this to suggest a random word which is consistent with all these results.
    We assume that the corpus is already consistent with the first n-1 results."""
"""The results consists an array of length 5 with numbers 0, 1, 2 where 2 is right-letter right place and 1
is the right letter in the wrong place. The corpus is a list of five letter words. words is last word tried"""
def suggest_word(word, results, corpus, distribution):
    #print(len(word), word)
    for i in range(5):
        if results[i] == 0:
            corpus, distribution = remove_wrong(corpus, word[i], distribution)
        if results[i] == 1:
            corpus, distribution = remove_no_letter(corpus, word[i], distribution)
            corpus, distribution = remove_wrong_place(corpus, word[i], i, distribution)
        if results[i] == 2:
            corpus, distribution = remove_not_green(corpus, word[i], i, distribution)
    #print(type(distribution), sum(distribution), corpus)
    distribution = distribution/sum(distribution)
    return corpus, distribution


"""This function removes a word if doesn't have a particular letter"""   #yellow
def remove_no_letter(corpus, letter, distribution):
    new_corpus = []
    new_distribution = []
    for i in range(len(corpus)):
        word = corpus[i]
        if letter in word:
            new_corpus.append(word)
            new_distribution.append(distribution[i])
    #print(len(corpus))
    return new_corpus.copy(), new_distribution.copy()

"""This removes words which do not have a letter in a given place"""  #green
def remove_not_green(corpus, letter, index, distribution):
    new_corpus = []
    new_distribution = []
    for i in range(len(corpus)):
        word = corpus[i]
        if word[index] == letter:
            new_corpus.append(word)
            new_distribution.append(distribution[i])
    return new_corpus.copy(), new_distribution.copy()

"""This removes words which have a letter in a known wrong place"""  #yellow
def remove_wrong_place(corpus, letter, index, distribution):
    new_corpus = []
    new_distribution = []
    for i in range(len(corpus)):
        word = corpus[i]
        if word[index] != letter:
            new_corpus.append(word)
            new_distribution.append(distribution[i])
    return new_corpus.copy(), new_distribution.copy()

"""This removes words which contain a known wrong letter"""   #blank
def remove_wrong(corpus, letter, distribution):
    new_corpus = []
    new_distribution = []
    for i in range(len(corpus)):
        word = corpus[i]
        if letter not in word:
            new_corpus.append(word)
            new_distribution.append(distribution[i])
    return new_corpus.copy(), new_distribution.copy()


lines = []
with open("./words.txt") as f:
    lines = f.readlines()

for i in range(len(lines)):
    lines[i] = lines[i].strip()


"""This method takes a true word and a guess and outputs the list of 0s, 1s and 2s"""
def wordle(true_word, guess_word):
    out = [0,0,0,0,0]
    for i in range(5):
        if true_word[i] == guess_word[i]:
            out[i] = 2
        if true_word[i] != guess_word[i]:
            if guess_word[i] in true_word:
                out[i] = 1
            else:
                out[i] = 0
    return out

def play(corpus, true_word, distribution=None):
    orig_corpus = corpus.copy()
    guess_indices = []
    #if distribution == None:
    #    distribution = np.ones(len(corpus))/len(corpus)
    rand_index = np.random.choice(np.arange(0, len(corpus)), p=distribution)
    guess = corpus[rand_index]
    #guess = "cares"
    guess_indices.append(rand_index)
    out = wordle(true_word, guess)
    turn = 1
    #print(guess, out, true_word, turn)
    while sum(out) < 10:
        corpus, distribution = suggest_word(guess,out, corpus, distribution)
        #rand_index = random.randint(0, len(corpus)-1)
        #distribution = np.ones(len(corpus)) / len(corpus)
        rand_index = np.random.choice(np.arange(0, len(corpus)), p=distribution)
        guess = corpus[rand_index]
        guess_indices.append(orig_corpus.index(guess))
        out = wordle(true_word, guess)
        turn += 1
        #print(guess, out, true_word, turn)
    #print("--------------")
    return turn, guess_indices

"""
num_turns = []
for j in range(1000):
    true_index = random.randint(0, len(lines) - 1)
    distribution = np.ones(len(lines))/len(lines)
    #turn, guess_indices = play(lines, lines[true_index], distribution)
    turn, _ = play(lines, "proxy", distribution)
    num_turns.append(turn)
print(sum(num_turns)/1000)
plt.hist(num_turns, bins=[1,2,3,4,5,6,7])
plt.show()"""


"""
#play(lines, "domes")
num_iters = 500000
number_of_turns = []
for iter in range(num_iters):
    if iter % 500 == 0:
        print(iter)
    eps = 0.00002
    distribution = np.ones(len(lines))/len(lines)
    true_index = random.randint(0, len(lines) - 1)
    turns1, indices1 = play(lines, lines[true_index], distribution)
    turns2, indices2 = play(lines, lines[true_index], distribution)
    if turns1 < turns2:
        for i in range(len(indices2)):
            distribution[indices2[i]] -= eps/turns2
        for i in range(len(indices1)):
            distribution[indices1[i]] += eps/turns1
    elif turns2 < turns1:
        for i in range(len(indices2)):
            distribution[indices2[i]] += eps/turns2
        for i in range(len(indices1)):
            distribution[indices1[i]] -= eps/turns1
    number_of_turns.append(turns1)
    number_of_turns.append(turns2)"""

"""This function tests what the average number of turns needed is given some distribution"""
def test_performance(corpus, distribution, num_ters=1000):
    total_turns = 0
    turn_array = []
    for iter in range(num_ters):
        if iter % 500 == 0:
            print(iter)
        true_index = random.randint(0, len(corpus) - 1)
        turns, _ = play(lines, corpus[true_index], distribution)
        turn_array.append(turns)
        total_turns += turns
    return total_turns/num_ters#, np.var(turn_array)


"""This function takes each initial guess and calculates the size of the resulting corpus after one
    round"""
def initial_choices(corpus, iters=100):
    distribution = np.ones(len(corpus)) / len(corpus)
    means = []
    stds = []
    for i in range(len(corpus)):
        initial_word = corpus[i]
        corp_size = 0
        corp_size2 = 0
        print(initial_word)
        for j in range(iters):
            rand_index = random.randint(0, len(corpus)-1)
            true_word = corpus[rand_index]
            out = wordle(true_word, initial_word)
            new_corpus, _ = suggest_word(initial_word, out, corpus, distribution)
            corp_size += len(new_corpus)
            corp_size2 += len(new_corpus)*len(new_corpus)
        corp_var = corp_size2/iters - (corp_size/iters)**2
        means.append(corp_size/iters)
        stds.append(math.sqrt(corp_var))
    return means, stds


def suggest_best(corpus):
    distribution = np.ones(len(corpus)) / len(corpus)
    corpus_lengths = []
    for i in range(len(corpus)):
        sugg_word = corpus[i]
        corp_len = 0
        for j in range(len(corpus)):
            true_word = corpus[j]
            out = wordle(true_word, sugg_word)
            new_corpus, _ = suggest_word(sugg_word, out, corpus, distribution)
            corp_len += len(new_corpus)
        corpus_lengths.append(corp_len)
    ind_max = np.argmin(np.array(corpus_lengths))
    return corpus[ind_max]

#best_word = suggest_best(lines)
#print("best word is: " + best_word)

def play_best(corpus, true_word, distribution=None):
    orig_corpus = corpus.copy()
    guess_indices = []
    rand_index = np.random.choice(np.arange(0, len(corpus)), p=distribution)
    guess = corpus[rand_index]
    #guess = "cares"
    guess_indices.append(rand_index)
    out = wordle(true_word, guess)
    turn = 1
    print(guess, out, true_word, turn)
    while sum(out) < 10:
        corpus, distribution = suggest_word(guess,out, corpus, distribution)
        if len(corpus) < 300:
            guess = suggest_best(corpus)
        else:
            guess = corpus[np.random.choice(np.arange(0, len(corpus)), p=distribution)]
        guess_indices.append(orig_corpus.index(guess))
        out = wordle(true_word, guess)
        turn += 1
        print(guess, out, true_word, turn)
    print("--------------")
    return turn, guess_indices

distribution = np.ones(len(lines))/len(lines)
total_turns = 0
for i in range(20):
    rand_index = np.random.choice(np.arange(0, len(lines)), p=distribution)
    true = lines[rand_index]
    turns, _ = play_best(lines, true, distribution)
    total_turns += turns
print(total_turns/20)


"""
distribution = np.ones(len(lines))/len(lines)
out = wordle("proxy", "raise")
lines2, distribution = suggest_word("raise", out, lines, distribution)

best_word = suggest_best(lines2)
print("best next word is: " + best_word)

out = wordle("proxy", best_word)
lines3, distribution = suggest_word(best_word, out, lines2, distribution)
best_word = suggest_best(lines3)
print("best next word is: " + best_word)

out = wordle("proxy", best_word)
lines4, distribution = suggest_word(best_word, out, lines3, distribution)
best_word = suggest_best(lines4)
print("best next word is: " + best_word)

out = wordle("proxy", best_word)
lines5, distribution = suggest_word(best_word, out, lines4, distribution)
best_word = suggest_best(lines5)
print("best next word is: " + best_word)"""

"""
distribution = np.ones(len(lines))/len(lines)
lines2, distribution = suggest_word("cared", [0,0,1,0,0], lines, distribution)
print("---------------")

lines3, distribution = suggest_word("orbit", [1,2,0,0,0], lines2, distribution)
print(lines3)

lines4, distribution = suggest_word("wrong", [0,2,2,0,0], lines3, distribution)
print(lines4)
rand_index = random.randint(0, len(lines4) - 1)
guess = lines4[rand_index]
print(guess)"""

"""
means, _ = initial_choices(lines, iters=100)
means = np.array(means)
lines = np.array(lines)
sort_indices = means.argsort()
means = means[sort_indices]
lines = lines[sort_indices]
with open("./initial_word_means2.txt", 'w') as f:
    for i in range(len(lines)):
        print(lines[i], means[i])
        f.write(lines[i] + "   ")
        f.write(str(means[i]))
        f.write("\n")
#trained_performance = test_performance(lines, distribution, 1000)
#uniform_distribution = np.ones(len(lines)) / len(lines)
#uniform_performance = test_performance(lines, uniform_distribution, 1000)
#print(uniform_performance, trained_performance)
"""
"""
turns = []
for j in range(len(lines)):
    num_turns = 0
    for i in range(100):
        num_turns += play(lines, lines[j])
    turns.append(num_turns/10)

plt.hist(turns) #, bins = [1,2,3,4,5,6,7,8])
plt.show()

turns = np.asarray(turns)
index = np.argmax(turns)
print(lines[index])"""

#print(turns)
#plt.hist(turns, bins = [1,2,3,4,5,6,7,8])
#plt.show()"""

#distribution = np.ones(len(lines))/len(lines)
#word_test = suggest_word('cured', [0,0,1,1,0], lines, distribution)
#print(word_test)

"""
lines2, distribution = suggest_word("cured", [0,0,1,1,0], lines, distribution)
print("---------------")
lines3, distribution = suggest_word("raise", [1,0,2,1,2], lines2, distribution)

rand_index = random.randint(0, len(lines3) - 1)
guess = lines3[rand_index]
print(guess)"""


"""This is a harder generalisation of the same game -- we get a 2(green) is you get right thing in right place as before
    you get a yellow(1) if the letter in this place of the true word is somewhere in your word. You get a blank (0) if the
    letter in this position appears nowhere in your word"""
def reverdle(true_word, guess_word):
    out = [0,0,0,0,0]
    for i in range(5):
        if true_word[i] == guess_word[i]:
            out[i] = 2
        if guess_word[i] != true_word[i]:
            if true_word[i] in guess_word:
                out[i] = 1
            else:
                out[i] = 0
    return out

#out = reverdle("bated", "hated")
#print(out)

#a = wordle("beach", "blade")
#print(a)

"""This solver works by picking from a corpus according to some weightings rather than randomly uniformly."""


















