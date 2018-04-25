import random
import math
import numpy

# Outputs a random integer, according to a multinomial
# distribution specified by probs.
def rand_multinomial(probs):
    # Make sure probs sum to 1
    assert(abs(sum(probs) - 1.0) < 1e-5)
    rand = random.random()
    for index, prob in enumerate(probs):
        if rand < prob:
            return index
        else:
            rand -= prob
    return 0

# Outputs a random key, according to a (key,prob)
# iterator. For a probability dictionary
# d = {"A": 0.9, "C": 0.1}
# call using rand_multinomial_iter(d.items())
def rand_multinomial_iter(iterator):
    rand = random.random()
    for key, prob in iterator:
        if rand < prob:
            return key
        else:
            rand -= prob
    return 0

class HMM():

    def __init__(self):
        self.num_states = 2
        self.prior = [0.5, 0.5]
        self.transition = [[0.999, 0.001], [0.01, 0.99]]
        # self.transition = [[0.5, 0.5], [0.4, 0.6]]
        self.emission = [{"A": 0.291, "T": 0.291, "C": 0.209, "G": 0.209}, #L
                         {"A": 0.169, "T": 0.169, "C": 0.331, "G": 0.331}] #H

    # Generates a sequence of states and characters from
    # the HMM model.
    # - length: Length of output sequence
    def sample(self, length):
        sequence = []
        states = []
        rand = random.random()
        cur_state = rand_multinomial(self.prior)
        for i in range(length):
            states.append(cur_state)
            char = rand_multinomial_iter(self.emission[cur_state].items())
            sequence.append(char)
            cur_state = rand_multinomial(self.transition[cur_state])
        return sequence, states

    # Generates a emission sequence given a sequence of states
    def generate_sequence(self, states):
        sequence = []
        for state in states:
            char = rand_multinomial_iter(self.emission[state].items())
            sequence.append(char)
        return sequence

    # Computes the (natural) log probability of sequence given a sequence of states.
    def logprob(self, sequence, states):
        ###########################################
        result = [];
        initial = self.emission[states[0]][sequence[0]]
        result.append(math.log(self.prior[0]) + math.log(initial))

        for i in range(1, len(sequence)):
            em = self.emission[states[i]][sequence[i]]
            trns = self.transition[states[i-1]][states[i]]
            prev = result[i-1]

            result.append(math.log(trns) + math.log(em) + prev)

        return result[len(result)-1]
        ###########################################

    # Outputs the most likely sequence of states given an emission sequence
    # - sequence: String with characters [A,C,T,G]
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def viterbi(self, sequence):
        ###########################################
        l = len(sequence)

        initial0 = self.emission[0][sequence[0]]
        initial1 = self.emission[1][sequence[0]]

        row0 = numpy.empty([self.num_states, l])
        row1 = numpy.empty([self.num_states, l])

        row0[0, 0] = math.log(self.prior[0]) + math.log(initial0)
        row0[1, 0] = math.log(self.prior[1]) + math.log(initial1)
        row1[0, 0] = 0
        row1[1, 0] = 0

        for i in range(1, l):
            for j in range(0, self.num_states):

                prev1 = row0[0][i-1]
                prev2 = row0[1][i-1]

                emission = self.emission[j][sequence[i]]

                trns1 = self.transition[0][j]
                trns2 = self.transition[1][j]

                low = math.log(trns1) + prev1
                high = math.log(trns2) + prev2
                row0[j, i] = max(low, high) + math.log(emission)
                row1[j, i] = numpy.argmax([low, high])

        states = numpy.empty(l, int)
        # finds which state has a higher prob
        states[l-1] = row0[:, l-1].argmax()

        for j in range(l-1, 0, -1):
            states[j-1] = row1[states[j], j]

        return states.tolist()
        # End your code

def read_sequence(filename):
    with open(filename, "r") as f:
        return f.read().strip()

def write_sequence(filename, sequence):
    with open(filename, "w") as f:
        f.write("".join(sequence))

def write_output(filename, logprob, states):
    with open(filename, "w") as f:
        f.write(repr(logprob))
        f.write("\n")
        for state in range(2):
            f.write(str(states.count(state)))
            f.write("\n")
        f.write("".join(map(str, states)))
        f.write("\n")

hmm = HMM()

sequence = read_sequence("small.txt")
viterbi = hmm.viterbi(sequence)
logprob = hmm.logprob(sequence, viterbi)
write_output("my_small_output.txt", logprob, viterbi)

# sequence = read_sequence("ecoli.txt")
# viterbi = hmm.viterbi(sequence)
# logprob = hmm.logprob(sequence, viterbi)
# write_output("ecoli_output.txt", logprob, viterbi)