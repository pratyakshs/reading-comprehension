# plot histograms of question, context, answer lengths in training data

import os
import matplotlib.pyplot as plt

data_dir = 'assignment4/data/'
squad_dir = 'squad/'

# histogram of question length
def question_len_histogram(filename, outfile):
    with open(os.path.join(data_dir, squad_dir, 'train.question'), 'r') as f:
        lengths = [len(line.split()) for line in f.read().splitlines()]
    plt.clf()
    n, bins, patches = plt.hist(lengths, 50, normed=1)
    plt.xlabel('length of question')
    plt.ylabel('frequency')
    plt.title('question length histogram')
    plt.grid(True)
    plt.savefig(outfile)


# histogram of context length
def context_len_histogram(filename, outfile):
    with open(os.path.join(data_dir, squad_dir, 'train.context'), 'r') as f:
        lengths = [len(line.split()) for line in f.read().splitlines()]
    plt.clf()
    n, bins, patches = plt.hist(lengths, 50, normed=1)
    plt.xlabel('length of context')
    plt.ylabel('frequency')
    plt.title('context length histogram')
    plt.grid(True)
    plt.savefig(outfile)


# histogram of answer length
def answer_len_histogram(filename, outfile):
    with open(os.path.join(data_dir, squad_dir, 'train.answer'), 'r') as f:
        lengths = [len(line.split()) for line in f.read().splitlines()]
    plt.clf()
    n, bins, patches = plt.hist(lengths, 50, normed=1)
    plt.xlabel('length of answer')
    plt.ylabel('frequency')
    plt.title('answer length histogram')
    plt.grid(True)
    plt.savefig(outfile)



if __name__ == '__main__':
    question_len_histogram('train.question', 'qlen-hist.pdf')
    context_len_histogram('train.context', 'clen-hist.pdf')
    answer_len_histogram('train.answer', 'alen-hist.pdf')
