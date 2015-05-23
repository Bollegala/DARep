#! /usr/bin/python -u
"""
Learn word representations for domain adaptation.
"""


import numpy
import sys
import cPickle as pickle


class WordRepresentationLearner:

    def __init__(self, D):
        self.D = D  # Dimensionality of the representations
        pass


    def load_words(self, pivots_fname, nonpivots_fname, source_train_fname, target_train_fname):
        """
        Load all related files. 
        """
        self.pivots = self.load_pivots(pivots_fname, "P")
        self.src_nonpivots = self.load_pivots(nonpivots_fname, "S")
        self.tgt_nonpivots = self.load_pivots(nonpivots_fname, "T")
        print "Total no. of pivots =", len(self.pivots)
        print "Total no. of source nonpivots =", len(self.src_nonpivots)
        print "Total no. of target nonpivots =", len(self.tgt_nonpivots)
        self.src_instances, self.src_count = self.load_train_data(source_train_fname)
        self.tgt_instances, self.tgt_count = self.load_train_data(target_train_fname)
        print "Total no. of source train instances = %d (cases = %d)" % (self.src_count, len(self.src_instances))
        print "Total no. of target train instances = %d (cases = %d)" % (self.tgt_count, len(self.tgt_instances))
        pass


    def train(self, epohs=1, l=1.0, alpha=0.001):
        """
        Perform training
        """
        self.squared_gradient_initialization()
        S_N = len(self.src_instances) 
        T_N = len(self.tgt_instances)
        u = numpy.ones(self.D, dtype=numpy.float)

        # Model parameters.
        self.epohs = epohs
        self.l = l 
        self.alpha = alpha
       
        for t in range(epohs):
            total_train_errors = 0
            count = 0
            print "\rSource round =", t
            for (i, (pivot, positives, negatives)) in enumerate(self.src_instances):
                print "\r%d: %2.2f" % (i, float(100 * i) / float(S_N)),
                for pos in positives:
                    for neg in negatives:
                        score = numpy.dot(self.c_S[pivot], (self.w_S[pos] - self.w_S[neg]))
                        count += 1
                        if score < 1:
                            total_train_errors += 1
                            # dL/dwS
                            grad_w_S_pos = (-self.c_S[pivot])
                            self.w_S[pos] -= alpha * (1.0 / numpy.sqrt(u + self.grad_w_S[pos])) * grad_w_S_pos 
                            self.grad_w_S[pos] += grad_w_S_pos ** 2
                    
                            # dL/dw*S
                            grad_w_S_neg = self.c_S[pivot]
                            self.w_S[neg] -= alpha * (1.0 / numpy.sqrt(u + self.grad_w_S[neg])) * grad_w_S_neg
                            self.grad_w_S[neg] += grad_w_S_neg ** 2
                          
                            # dL/dcS
                            grad_c_S = self.w_S[neg] - self.w_S[pos] + (l * (self.c_S[pivot] - self.c_T[pivot]))
                            self.c_S[pivot] -= alpha * (1.0 / numpy.sqrt(u + self.grad_c_S[pivot])) * grad_c_S
                            self.grad_c_S[pivot] += grad_c_S ** 2
                            
                        else:
                            grad_c_S = l * (self.c_S[pivot] - self.c_T[pivot])
                            self.c_S[pivot] -= alpha * (1.0 / numpy.sqrt(u + self.grad_c_S[pivot])) * grad_c_S
                            self.grad_c_S[pivot] += grad_c_S ** 2
                            
            print "\n Source error rate =", float(100 * total_train_errors) / float(count)

            total_train_errors = 0
            count = 0
            print "\rTarget round =", t
            for (i, (pivot, positives, negatives)) in enumerate(self.tgt_instances):
                print "\r %d: %2.2f" % (i, float(100 * i) / float(T_N)),
                for pos in positives:
                    for neg in negatives:
                        score = numpy.dot(self.c_T[pivot], (self.w_T[pos] - self.w_T[neg]))
                        count += 1
                        if score < 1:
                            total_train_errors += 1
                            # dL/dwT
                            grad_w_T_pos = (-self.c_T[pivot])
                            self.w_T[pos] -= alpha * (1.0 / numpy.sqrt(u + self.grad_w_T[pos])) * grad_w_T_pos 
                            self.grad_w_T[pos] += grad_w_T_pos ** 2
                            
                            # dL/dw*T
                            grad_w_T_neg = self.c_T[pivot]
                            self.w_T[neg] -= alpha * (1.0 / numpy.sqrt(u + self.grad_w_T[neg])) * grad_w_T_neg
                            self.grad_w_T[neg] += grad_w_T_neg ** 2
                           
                            # dL/dcT
                            grad_c_T = self.w_T[neg] - self.w_T[pos] + (l * (self.c_T[pivot] - self.c_S[pivot]))
                            self.c_T[pivot] -= alpha * (1.0 / numpy.sqrt(u + self.grad_c_T[pivot])) * grad_c_T
                            self.grad_c_T[pivot] += grad_c_T ** 2
                            
                        else:
                            grad_c_T = l * (self.c_T[pivot] - self.c_S[pivot])
                            self.c_T[pivot] -= alpha * (1.0 / numpy.sqrt(u + self.grad_c_T[pivot])) * grad_c_T
                            self.grad_c_T[pivot] += grad_c_T ** 2

            print "\n Traget error rate =", float(100 * total_train_errors) / float(count)            
        pass


    def normalize(self, x):
        return (x / numpy.linalg.norm(x))     

    def squared_gradient_initialization(self):
        """
        Initialize the squared totals for gradients used by AdaGrad
        """
        self.grad_w_S = {}
        self.grad_c_S = {}
        self.grad_w_T = {}
        self.grad_c_T = {}
        for w in self.pivots:
            self.grad_c_S[w] = numpy.zeros(self.D, dtype=numpy.float)
            self.grad_c_T[w] = numpy.zeros(self.D, dtype=numpy.float)
        for w in self.src_nonpivots:
            self.grad_w_S[w] = numpy.zeros(self.D, dtype=numpy.float)
        for w in self.tgt_nonpivots:
            self.grad_w_T[w] = numpy.zeros(self.D, dtype=numpy.float)
        pass


    def random_initialization(self):
        """
        Randomly initialize all word representations. 
        """
        self.w_S = {}
        self.c_S = {}
        self.w_T = {}
        self.c_T = {}
        for w in self.pivots:
            x = self.get_unit_random_vect()
            self.c_S[w] = x
            self.c_T[w] = x
        for w in self.src_nonpivots:
            self.w_S[w] = self.get_unit_random_vect()
        for w in self.tgt_nonpivots:
            self.w_T [w]= self.get_unit_random_vect()
        pass


    # def save_model(self, fname):
    #     """
    #     Save the current word representations to a file.
    #     """
    #     para = {"epohs":self.epohs, "l":self.l, "alpha":self.alpha}
    #     model = {"w_S":self.w_S, "c_S":self.c_S, 
    #             "w_T":self.w_T, "c_T":self.c_T, "parameters":para}
    #     with open(fname, "wb") as model_file:
    #         pickle.dump(model, model_file)
    #     pass

    def save_model(self, fname):
        """
        Save the current word representations to a file.
        """
        para = {"epohs":self.epohs, "l":self.l, "alpha":self.alpha}
        model = {"w_S":self.w_S, "c_S":self.c_S, 
                "w_T":self.w_T, "c_T":self.c_T, "parameters":para}
        with open(fname, "wb") as model_file:
            pickle.dump(model, model_file)
        pass


    def get_unit_random_vect(self):
        """
        Returns a unit L2 length random vector sampled uniformly from [0, 1]
        """
        x = numpy.random.uniform(low=-1.0, high=1.0, size=self.D)
        return x / numpy.linalg.norm(x)


    def load_train_data(self, train_fname):
        """
        Load source or target train instances to train the skip-gram model
        """
        instances = []
        count = 0
        with open(train_fname) as train_file:
            for line in train_file:
                p = line.strip().split('\t')
                pivot = p[0]
                positives = p[1].split(',')
                negatives = p[2].split(',')
                instances.append((pivot, positives, negatives))
                count += len(positives) + len(negatives)
        return (instances, count)


    def load_pivots(self, fname, domain):
        """
        Returns a list of pivots (or non-pivots). 
        """
        L = []
        with open(fname) as F:
            for line in F:
                p = line.strip().split()
                d = p[2].strip()
                w = p[1].strip()
                if domain == d:
                    L.append(w)
                else:
                    ValueError
        return L



def process(source, target, D):
    print "Source =", source, "Target =", target
    print "Dimensionality =", D
    base_path = "../work/%s-%s/" % (source, target)
    WR = WordRepresentationLearner(D)
    WR.load_words(base_path+"pivots", base_path+"nonpivots", base_path+"source.train", base_path+"target.train")
    WR.random_initialization()
    WR.train(epohs=10, l=1.0, alpha=0.1)
    WR.save_model(base_path+"model.e=10.d=300.l=1")
    pass


def batch_process(source):
    """
    Calls all targets with the given source.
    """
    domains = ["books", "electronics", "dvd", "kitchen"]
    D = 300
    print source
    for target in domains:
        if target != source:
            process(source, target, D)
    pass


if __name__ == '__main__':
    process("books", "electronics", 300)
    pass

