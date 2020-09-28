import argparse
import os
from math import log
import string
from string import digits
table = str.maketrans(dict.fromkeys(string.punctuation))
table1 = str.maketrans("", "", digits)

from enum import Enum

tuning_var = 1

class MessageType(Enum):
    REGULAR = 1,
    SPAM = 2

## Bigram class: represent one bigram consisting of two tokens (words)
class Bigram():
    def __init__(self):
        self.token1 = None
        self.token2 = None
    
    def __hash__(self):
        return hash((self.token1, self.token2))

    def __eq__(self, other):
        return (self.token1, self.token2) == (other.token1, other.token2)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)

class Counter():

    def __init__(self):
        self.counter_regular = 0
        self.counter_spam = 0

    def increment_counter(self, message_type):
        """
        Increment a word's frequency count by one, depending on whether it occurred in a regular or spam message.

        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            self.counter_regular += 1
        else:
            self.counter_spam += 1

class Bayespam():

    def __init__(self):
        self.regular_list = None
        self.spam_list = None
        self.vocab = {}

    def list_dirs(self, path):
        """
        Creates a list of both the regular and spam messages in the given file path.

        :param path: File path of the directory containing either the training or test set
        :return: None
        """
        # Check if the directory containing the data exists
        if not os.path.exists(path):
            print("Error: directory %s does not exist." % path)
            exit()

        regular_path = os.path.join(path, 'regular')
        spam_path = os.path.join(path, 'spam')

        # Create a list of the absolute file paths for each regular message
        # Throws an error if no directory named 'regular' exists in the data folder
        try:
            self.regular_list = [os.path.join(regular_path, msg) for msg in os.listdir(regular_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'regular'." % path)
            exit()

        # Create a list of the absolute file paths for each spam message
        # Throws an error if no directory named 'spam' exists in the data folder
        try:
            self.spam_list = [os.path.join(spam_path, msg) for msg in os.listdir(spam_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'spam'." % path)
            exit()

    ## Remove puctation (table), digits (table1), and words with 3 or less letters, change all upper to lower case letters
    def clean_vocab(self, token):
        token = token.translate(table)
        token = token.translate(table1)
        token = token.lower()
        if len(token) > 3 : return token


    def read_messages(self, message_type):
        """
        Parse all messages in either the 'regular' or 'spam' directory. Each token is stored in the vocabulary,
        together with a frequency count of its occurrences in both message types.
        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            message_list = self.regular_list
        elif message_type == MessageType.SPAM:
            message_list = self.spam_list
        else:
            message_list = []
            print("Error: input parameter message_type should be MessageType.REGULAR or MessageType.SPAM")
            exit()

        for msg in message_list:
            try:
                # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
                f = open(msg, 'r', encoding='latin1')

                # Loop through each line in the message
                for line in f:
                    # Split the string on the space character, resulting in a list of tokens
                    split_line = line.split(" ")
                    # Loop through the tokens
                    for idx in range(len(split_line) - 1):
                        ## create a bigram from two consecutivve tokens
                        bigram = Bigram()
                        bigram.token1 = split_line[idx]
                        bigram.token2 = split_line[idx + 1]

                        ## Remove unwanted characters from the tokens
                        bigram.token1 = self.clean_vocab(bigram.token1)
                        bigram.token2 = self.clean_vocab(bigram.token2)

                        ## Exclude bigrams containing None values and set the according counter 
                        if bigram.token1 != None and bigram.token2 != None:
                            if bigram in self.vocab.keys():
                                # If the token is already in the vocab, retrieve its counter
                                counter = self.vocab[bigram]
                            else:
                                # Else: initialize a new counter
                                counter = Counter()

                            # Increment the token's counter by one and store in the vocab
                            counter.increment_counter(message_type)
                            self.vocab[bigram] = counter
            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()

    ## Remove bigrams that occur less than four times from the dicitonary
    def delete_low_frequency_bigrams(self):
        for bigram in list(self.vocab):
            if self.vocab.get(bigram).counter_regular < 4 and self.vocab.get(bigram).counter_spam < 4:
                del self.vocab[bigram]

    ## Calculate a priori probabilities 
    def apriori(self):
        n_messages_regular = len(self.regular_list)  
        n_messages_spam = len(self.spam_list) 
        n_messages_total = n_messages_regular + n_messages_spam
        prob_regular = log(n_messages_regular / n_messages_total)
        prob_spam = log(n_messages_spam / n_messages_total)
        return prob_regular, prob_spam

    ## Create a dictionary containing the bigrams' probabilities of occurring given the message type 
    def conditional_word(self):
        conditional_dict = {}
        n_words_regular = 0
        n_words_spam = 0
        for bigram in self.vocab:
            n_words_regular += self.vocab.get(bigram).counter_regular
            n_words_spam += self.vocab.get(bigram).counter_spam

        ## Create a small valued probability to fall back on when it would otherwise be zero
        fallback_prob = log(tuning_var / (n_words_regular + n_words_spam))

        ## Calculate the probabilites of the words in the dicitonary given the type of message
        ## Add into a new dictionary
        for bigram in self.vocab:
            p_array = [0, 0]
            if (self.vocab.get(bigram).counter_regular == 0):
                p_word_given_regular = fallback_prob
            else:
                p_word_given_regular = log(self.vocab.get(bigram).counter_regular / n_words_regular)
            p_array[0] = p_word_given_regular

            if (self.vocab.get(bigram).counter_spam == 0):
                p_word_given_spam = fallback_prob
            else:
                p_word_given_spam = log(self.vocab.get(bigram).counter_spam / n_words_spam)
            p_array[1] = p_word_given_spam
            
            conditional_dict[bigram] = p_array
        return conditional_dict

    ## Classify messages to return a confusion matrix for the given message type
    def posterior(self, message_type, apriori_regular, apriori_spam, conditional_dict):

        alpha_regular = 0
        alpha_spam = 0
        p_regular_given_msg = alpha_regular + apriori_regular
        p_spam_given_msg = alpha_spam + apriori_spam

        true_regular = 0
        true_spam = 0
        false_regular = 0
        false_spam = 0

        ## Set a flag to represent the actual type of the message
        regular = None
        if message_type == MessageType.REGULAR:
            message_list = self.regular_list
            regular = True
        elif message_type == MessageType.SPAM:
            message_list = self.spam_list
            regular = False
        else:
            message_list = []
            print("Error: input parameter message_type should be MessageType.REGULAR or MessageType.SPAM")
            exit()

        for msg in message_list:
            try:
                # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
                f = open(msg, 'r', encoding='latin1')

                # Loop through each line in the message
                for line in f:
                    # Split the string on the space character, resulting in a list of tokens
                    split_line = line.split(" ")
                    # Loop through the tokens
                    for idx in range(len(split_line) - 1):
                        bigram = Bigram()
                        bigram.token1 = split_line[idx]
                        bigram.token2 = split_line[idx + 1]

                        ## Sum the conditional probabilites of the words in the messages
                        ## (bigrams not in the dictionary will be ignored)
                        if bigram in conditional_dict.keys():
                            p_regular_given_msg += conditional_dict.get(bigram)[0]
                            p_spam_given_msg += conditional_dict.get(bigram)[1]

                ## Compare the classification result to the actual type of the message
                ## Increase the counter accordingly for the confusion matrix
                if (p_regular_given_msg > p_spam_given_msg and regular == True):
                    true_regular += 1
                elif (p_regular_given_msg < p_spam_given_msg and regular == True):
                    false_spam += 1
                elif (p_regular_given_msg < p_spam_given_msg and regular == False):
                    true_spam += 1
                elif (p_regular_given_msg > p_spam_given_msg and regular == False):
                    false_regular += 1


            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()
        
        return true_regular, false_regular, true_spam, false_spam


    def print_vocab(self):
        """
        Print each word in the vocabulary, plus the amount of times it occurs in regular and spam messages.

        :return: None
        """
        for bigram, counter in self.vocab.items():
            # repr(word) makes sure that special characters such as \t (tab) and \n (newline) are printed.
            print("%s | %s | In regular: %d | In spam: %d" % (repr(bigram.token1), repr(bigram.token2), counter.counter_regular, counter.counter_spam))

    def write_vocab(self, destination_fp, sort_by_freq=False):
        """
        Writes the current vocabulary to a separate .txt file for easier inspection.

        :param destination_fp: Destination file path of the vocabulary file
        :param sort_by_freq: Set to True to sort the vocab by total frequency (descending order)
        :return: None
        """

        if sort_by_freq:
            vocab = sorted(self.vocab.items(), key=lambda x: x[1].counter_regular + x[1].counter_spam, reverse=True)
            vocab = {x[0]: x[1] for x in vocab}
        else:
            vocab = self.vocab

        try:
            f = open(destination_fp, 'w', encoding="latin1")

            for bigram, counter in vocab.items():
                # repr(word) makes sure that special  characters such as \t (tab) and \n (newline) are printed.
                f.write("%s | %s | In regular: %d | In spam: %d\n" % (repr(bigram.token1), repr(bigram.token2), counter.counter_regular, counter.counter_spam),)

            f.close()
        except Exception as e:
            print("An error occurred while writing the vocab to a file: ", e)


def main():
    # We require the file paths of the training and test sets as input arguments (in that order)
    # The argparse library helps us cleanly parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str,
                        help='File path of the directory containing the training data')
    parser.add_argument('test_path', type=str,
                        help='File path of the directory containing the test data')
    args = parser.parse_args()

    # Read the file path of the folder containing the training set from the input arguments
    train_path = args.train_path

    # Initialize a Bayespam object
    bayespam = Bayespam()
    # Initialize a list of the regular and spam message locations in the training folder
    bayespam.list_dirs(train_path)

    # Parse the messages in the regular message directory
    bayespam.read_messages(MessageType.REGULAR)
    # Parse the messages in the spam message directory
    bayespam.read_messages(MessageType.SPAM)

    ## Reduce the dictionary to the most common bigrams
    bayespam.delete_low_frequency_bigrams()

    ## Calculate a priori
    apriori_regular, apriori_spam = bayespam.apriori()
    print('apriorispam, regular: ', apriori_spam, apriori_regular)

    ## Create a dictionary containing the probabilities of bigrams given the message type
    conditional_word = bayespam.conditional_word()

    ## Read the file path of the folder containing the training set form the input arguments
    test_path = args.test_path
    ## Reset the message lists
    bayespam.regular_list = None
    bayespam.spam_list = None
    ## Initialize a list of the regular and spam message locations in the test folder
    bayespam.list_dirs(test_path)

    ## Calculate confusion matrices for the two message types and add them into one
    confusion_matrix1 = bayespam.posterior(MessageType.REGULAR, apriori_regular, apriori_spam, conditional_word)
    confusion_matrix2 = bayespam.posterior(MessageType.SPAM, apriori_regular, apriori_spam, conditional_word)
    confusion_matrix = []
    for confusion_matrix1, confusion_matrix2 in zip(confusion_matrix1, confusion_matrix2):
        confusion_matrix.append(confusion_matrix1 + confusion_matrix2)
    print("confusion_matrix: ", confusion_matrix)

    ## Calculate performance values
    number_messages = len(bayespam.regular_list) + len(bayespam.spam_list)
    accuracy = (confusion_matrix[0] + confusion_matrix[2]) / number_messages
    # sensitivity
    true_positive = confusion_matrix[0] / len(bayespam.regular_list)
    # specificity
    true_negative = confusion_matrix[2] / len(bayespam.spam_list)
    print("accuracy: ", accuracy, "\nsensitivity: ", true_positive, "\nspecificity: ", true_negative)


    # bayespam.print_vocab()
    # bayespam.write_vocab("vocab.txt")

    print("N regular messages: ", len(bayespam.regular_list))
    print("N spam messages: ", len(bayespam.spam_list))

    """
    Now, implement the follow code yourselves:
    1) A priori class probabilities must be computed from the number of regular and spam messages
    2) The vocabulary must be clean: punctuation and digits must be removed, case insensitive
    3) Conditional probabilities must be computed for every word
    4) Zero probabilities must be replaced by a small estimated value
    5) Bayes rule must be applied on new messages, followed by argmax classification
    6) Errors must be computed on the test set (FAR = false accept rate (misses), FRR = false reject rate (false alarms))
    7) Improve the code and the performance (speed, accuracy)
    
    """

if __name__ == "__main__":
    main()