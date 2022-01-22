#This is a Wordle solver

import random
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse

"""We take the results of the last guesses and use this to suggest a random word which is consistent with all these results.
    We assume that the corpus is already consistent with the first n-1 results."""

"""The results consists an array of length 5 with numbers 0, 1, 2 where 2 is right-letter right place and 1
    is the right letter in the wrong place. The corpus is a list of five letter words. 'word' is last word tried.
    It returns a corpus consisting of the words of the original corpus consistent with the results."""
def valid_corpus(word, results, corpus, distribution):
    for i in range(5):
        if results[i] == 0:
            corpus, distribution = remove_wrong(corpus, word[i], distribution)
        if results[i] == 1:
            corpus, distribution = remove_no_letter(corpus, word[i], distribution)
            corpus, distribution = remove_wrong_place(corpus, word[i], i, distribution)
        if results[i] == 2:
            corpus, distribution = remove_not_green(corpus, word[i], i, distribution)
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

"""Return textfile of words as a list"""
def return_corpus(filename="./words.txt"):
    with open(filename) as f:
        lines = f.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    return lines

"""Plays a game using corpus given a true word and a corpus. Outputs the number of turns taken and the indices of the guesses
    made in the corpus"""
def play(corpus, true_word, distribution=None):
    orig_corpus = corpus.copy()
    guess_indices = []
    rand_index = np.random.choice(np.arange(0, len(corpus)), p=distribution)
    guess = corpus[rand_index]
    guess_indices.append(rand_index)
    out = wordle(true_word, guess)
    turn = 1
    while sum(out) < 10:
        corpus, distribution = valid_corpus(guess,out, corpus, distribution)
        rand_index = np.random.choice(np.arange(0, len(corpus)), p=distribution)
        guess = corpus[rand_index]
        guess_indices.append(orig_corpus.index(guess))
        out = wordle(true_word, guess)
        turn += 1
    return turn, guess_indices

"""Here we chosoe the first word at random after which we pick that which will, on average minimise the size of the resulting
    corpus."""
def play_greedy(corpus, true_word, distribution=None):
    orig_corpus = corpus.copy()
    guess_indices = []
    rand_index = np.random.choice(np.arange(0, len(corpus)), p=distribution)
    #guess = corpus[rand_index]
    guess = "cares"
    guess_indices.append(rand_index)
    out = wordle(true_word, guess)
    turn = 1
    while sum(out) < 10:
        corpus, distribution = valid_corpus(guess,out, corpus, distribution)
        if len(corpus) < 300:
            guess = suggest_greedy(corpus)
        else:
            guess = corpus[np.random.choice(np.arange(0, len(corpus)), p=distribution)]
        guess_indices.append(orig_corpus.index(guess))
        out = wordle(true_word, guess)
        turn += 1
    return turn, guess_indices

"""This function tests what the average number of turns needed is given some distribution of words"""
def test_performance(corpus, distribution, num_ters=1000):
    total_turns = 0
    for iter in range(num_ters):
        if iter % 500 == 0:
            print(iter)
        true_index = random.randint(0, len(corpus) - 1)
        turns, _ = play(lines, corpus[true_index], distribution)
        total_turns += turns
    return total_turns/num_ters#, np.var(turn_array)

"""This function tests what the average number of turns needed is given we play the greedy way"""
def test_greedy_performance(corpus, num_ters=100):
    distribution = np.ones(len(corpus)) / len(corpus)
    total_turns = 0
    for i in range(num_ters):
        rand_index = np.random.choice(np.arange(0, len(corpus)), p=distribution)
        true = corpus[rand_index]
        turns, _ = play_greedy(corpus, true, distribution)
        total_turns += turns
    return total_turns/num_ters

"""Suggest a word which minimises the expected size of the residual corpus"""
def suggest_greedy(corpus):
    distribution = np.ones(len(corpus)) / len(corpus)
    corpus_lengths = []
    for i in range(len(corpus)):
        sugg_word = corpus[i]
        corp_len = 0
        for j in range(len(corpus)):
            true_word = corpus[j]
            out = wordle(true_word, sugg_word)
            new_corpus, _ = valid_corpus(sugg_word, out, corpus, distribution)
            corp_len += len(new_corpus)
        corpus_lengths.append(corp_len)
    ind_min = np.argmin(np.array(corpus_lengths))
    return corpus[ind_min]

"""This function takes each initial guess and calculates the size of the resulting corpus after one
    round"""
def initial_choices(corpus):
    distribution = np.ones(len(corpus)) / len(corpus)
    means = []
    stds = []
    for i in range(len(corpus)):
        initial_word = corpus[i]
        corp_size = 0
        corp_size2 = 0
        print(corpus[i])
        for j in range(len(corpus)):
            rand_index = random.randint(0, len(corpus)-1)
            true_word = corpus[rand_index]
            out = wordle(true_word, initial_word)
            new_corpus, _ = valid_corpus(initial_word, out, corpus, distribution)
            corp_size += len(new_corpus)
            corp_size2 += len(new_corpus)*len(new_corpus)
        corp_var = corp_size2/len(corpus) - (corp_size/len(corpus))**2
        means.append(corp_size/len(corpus))
        stds.append(math.sqrt(corp_var))
    return means, stds


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


def main():
    parser = argparse.ArgumentParser(description='Arguments for Wordle')
    parser.add_argument('--true-word', type=str, default="hello", help='input the true word for the model to guess (default: hello)')
    args = parser.parse_args()

    lines = return_corpus()
    distribution = np.ones(len(lines))/len(lines)
    #out = wordle("robot", "thief")
    out = [1,0,1,1,0]
    lines2, distribution = valid_corpus("chink", out, lines, distribution)

    #best_word = suggest_greedy(lines2)
    #print("best next word is: " + best_word)
    print(lines2)

    out = [0,2,2,2,2]
    #out = wordle("robot", "court")
    lines3, distribution = valid_corpus("mince", out, lines2, distribution)

    #best_word = suggest_greedy(lines3)
    #print("best next word is: " + best_word)
    print(lines3)

    out = wordle("robot", "roofy")
    lines4, distribution = valid_corpus("roofy", out, lines3, distribution)

    print(lines4)

    distribution = np.ones(len(lines)) / len(lines)
    total_turns = 0
    for i in range(1000):
        turns, guess_indices = play(lines, true_word=args.true_word, distribution=distribution)
        total_turns += turns
    print("Mean turns taken for " + str(args.true_word) + " using random strategy is: " + str(total_turns/1000))


    distribution = np.ones(len(lines)) / len(lines)
    turns, guess_indices = play_greedy(lines, true_word=args.true_word, distribution=distribution)
    print("-----------")
    for j in range(len(guess_indices)):
        print(lines[guess_indices[j]])
    print("-----------")
    print("Mean turns taken for " + str(args.true_word) + " using greedy stategy is: " + str(turns))


    means, _ = initial_choices(lines)
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

    """
    distribution = np.ones(len(lines))/len(lines)
    out = wordle("proxy", "raise")
    lines2, distribution = valid_corpus("raise", out, lines, distribution)

    best_word = suggest_best(lines2)
    print("best next word is: " + best_word)

    out = wordle("proxy", best_word)
    lines3, distribution = valid_corpus(best_word, out, lines2, distribution)
    best_word = suggest_best(lines3)
    print("best next word is: " + best_word)

    out = wordle("proxy", best_word)
    lines4, distribution = valid_corpus(best_word, out, lines3, distribution)
    best_word = suggest_best(lines4)
    print("best next word is: " + best_word)

    out = wordle("proxy", best_word)
    lines5, distribution = valid_corpus(best_word, out, lines4, distribution)
    best_word = suggest_best(lines5)
    print("best next word is: " + best_word)"""


if __name__ == '__main__':
    main()













