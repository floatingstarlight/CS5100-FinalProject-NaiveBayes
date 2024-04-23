from collections import Counter
from math import log2

#This is the class dealing with the major Naive Bayes algorithm
#The algorithm implementation is revised based on Colvin (2014)'s code
#It includes a train function to calculate p(c) and P(w|c)
#It includes a test function to calculate p(c|w)

class NaiveBayesClassifier:
    # class list: [-1, 0, 1]
    # vocab_Size: to optimize, we calculate P(Feature|Class) for the 5000 most frequent words 
    # The rest words are combiend into 1 additonal feature called as one feature called Out-of-Vocab words(OOV)
    # smoothing factor used in laplace smoothing
    def __init__(self, C:list, vocab_size:int=5000, smoothing:float=0.05):
        self.class_list = C
        self.vocab_size = vocab_size
        self.smoothing = smoothing

    def train(self, data_list):
        # Add words into big list
        big_list = []
        for i in range(len(data_list)):
            big_list.extend(data_list[i][0])

        # counts: dictionary-like object that keep track of frequency of words in big list
        word_counts = Counter(big_list)
        
		# get a list of tuples(a word and its frequency count), sorted in descending order of frequency;
		# The V list size equals to vocab_size: 5000;
        self.V_most_frequent = [tup[0] for tup in word_counts.most_common(self.vocab_size)]
        
        total_documents = len(data_list)

		#To save log P(c)
        self.logpriors = []

		#To save words of one specific class into seperate lists
        self.class_doc = []

		#To save P(w|c) for each of the words(total: 5001 words-most frequent 5000 words + OOV)
        self.log_likelihood = [[] for _ in range(self.vocab_size + 1)] 

        # Iterate over three classes(-1, 0 ,1)
        for i in range(len(self.class_list)):
            c = self.class_list[i]
            print('Training %d' % c)

            # Calculate P(c) = number of documents in class c / all documents
            class_count = 0
            self.class_doc.append([])
            for sample in data_list:
                if sample[1] == c:
                    class_count += 1
                    self.class_doc[i].extend(sample[0])
            self.logpriors.append(log2(class_count/ total_documents))

			# Calculate P(w|c): number of occurrences of feature wi in documents in class c / the number of occurrences of all words in class c.
			# get words frequency for the current class
            class_word_counts = Counter(self.class_doc[i])

            # Using lapalce smoothing to calculate P(w|c): count(w, c) + smoothing_factor / sum of count(w,c) + |V=5000| * smoothing factor
			# Calculate sum of count(w,c) + |V| * smoothing factor
            occurrences = sum(class_word_counts.values()) + self.smoothing * (self.vocab_size + 1)

			# get counts for every word in V_most frequent 5000 words list; calculate p(w|c), and save into logLikelihood list
            for j in range(self.vocab_size): 
                current_word = self.V_most_frequent[j]
                words_frequency = class_word_counts[current_word]
                self.log_likelihood[j].append(log2((words_frequency + self.smoothing) / occurrences))

			# OOV words' probability: sum of probabilities of all the OOV words in that class.
            # get sum of OOV(out of vocabulary) word counts for OOV(out 0f 5000 frequency dict)
            oov_word_count = 0
            oov_word_count = sum([count for word, count in class_word_counts.most_common() if word not in self.V_most_frequent])
            self.log_likelihood[self.vocab_size].append(log2((oov_word_count+ self.smoothing) / occurrences))

    # get probability of each class, choose argmax class as prediction
    def test(self, testdoc):
        max_prob, max_c = float('-inf'), None
        for class_index, class_label in enumerate(self.class_list):
            class_prob = self.logpriors[class_index]
            for word in testdoc:
                if word in self.V_most_frequent:
                    word_index = self.V_most_frequent.index(word) if word in self.V_most_frequent else self.vocab_size
                    class_prob += self.log_likelihood[word_index][class_index]
            if class_prob > max_prob:
                max_prob = class_prob
                max_class = class_label
        return max_class

