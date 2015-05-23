#! /usr/bin/python -u

"""
Create training instances for learning word representations. 
"""

import sys
import math
import numpy
import scipy.io as sio 
from collections import defaultdict, OrderedDict
import string
import re


class FEATURE_GENERATOR:

    def __init__(self):
        # How many instances should be taken as training data.
        self.load_stop_words("./stopWords.txt")
        pass
    

    def load_stop_words(self, stopwords_fname):
        """
        Read the list of stop words and store in a dictionary.
        """
        with open(stopwords_fname) as F:
            self.stopWords = set(map(string.strip, F.readlines()))
        pass
    

    def is_stop_word(self, word):
        """
        If the word is listed as a stop word in self.stopWords,
        then returns True. Otherwise, returns False.
        """
        return (word in self.stopWords)
        

    def get_tokens(self, line):
        """
        Return a list of dictionaries, where each dictionary has keys
        lemma (lemmatized word), infl (inflections if any), and pos (the POS tag).
        The elements in the list are ordered according to their
        appearance in the sentence.
        """
        table = string.maketrans("", "")
        elements = line.strip().lower().translate(table, string.punctuation).split()
        return elements
    

    def get_rating_from_label(self, label):
        """
        Set the rating using the label.
        """
        if label == "positive":
            return "positive"
        elif label == "negative":
            return "negative"
        elif label == "unlabeled":
            return None
        pass


    def get_text(self, fname):
        """
        Extract text from each file. This is a list of lines. 
        """
        lines = [] #List of feature vectors.
        F = open(fname)
        line = F.readline()
        inReview = False
        count = 0
        tokens = []
        while line:
            if line.startswith('^^ <?xml version="1.0"?>'):
                line = F.readline()
                continue
            if line.startswith("<review>"):
                inReview = True
                tokens = []
                line = F.readline()
                continue
            if inReview and line.startswith("<rating>"):
                # Do not uncomment the following line even if you are not
                # using get_rating_from_score because we must skip the rating line.
                ratingStr = F.readline()
                line = F.readline() #skipping the </rating>
                continue
            if inReview and line.startswith("<Text>"):
                while line:
                    if line.startswith("</Text>"):
                        break
                    if len(line) > 1 and not line.startswith("<Text>"):
                        lines.append(line.strip())
                    line = F.readline()                    
            if inReview and line.startswith("</review>"):
                inReview = False
            line = F.readline()
        # write the final lines if we have not seen </review> at the end.
        if inReview:
            count += 1
        F.close()
        return lines
    

    def process_file(self, fname, label=None):
        """
        Open the file fname, generate all the features and return
        as a list of feature vectors.
        """
        feature_vectors = [] #List of feature vectors.
        F = open(fname)
        line = F.readline()
        inReview = False
        count = 0
        tokens = []
        while line:
            if line.startswith('^^ <?xml version="1.0"?>'):
                line = F.readline()
                continue
            if line.startswith("<review>"):
                inReview = True
                tokens = []
                line = F.readline()
                continue
            if inReview and line.startswith("<rating>"):
                # Do not uncomment the following line even if you are not
                # using get_rating_from_score because we must skip the rating line.
                ratingStr = F.readline()
                line = F.readline() #skipping the </rating>
                continue
            if inReview and line.startswith("<Text>"):
                while line:
                    if line.startswith("</Text>"):
                        break
                    if len(line) > 1 and not line.startswith("<Text>"):
                        curTokens = self.get_tokens(line.strip())
                        if curTokens:
                            tokens.extend(curTokens)
                    line = F.readline()                    
            if inReview and line.startswith("</review>"):
                inReview = False
                # generate feature vector from tokens.
                # Do not use rating related features to avoid overfitting.
                fv = self.get_features(tokens, rating=None)
                feature_vectors.append((label, fv))
                tokens = []
                count += 1
            line = F.readline()
        # write the final lines if we have not seen </review> at the end.
        if inReview:
            count += 1
        F.close()
        print fname, len(feature_vectors)
        return feature_vectors
    

    def get_features(self, tokens, rating=None):
        """
        Create a feature vector from the tokens and return it.
        """
        fv = defaultdict(int)
        # generate unigram features
        for token in tokens:
            if not self.is_stop_word(token):
                fv[token] += 1
                
        # generate bigram features.
        for i in range(len(tokens) - 1):
            bigram = "%s__%s" % (tokens[i], tokens[i+1])
            if not (self.is_stop_word(tokens[i]) and self.is_stop_word(tokens[i+1])):
                fv[bigram] += 1

        # if not__x is a bigram feature then remove x if x exists in the feature vector
        remove = set()
        for feat in fv:
            if feat.startswith("not__"):
                head = feat.split("__")[1].strip()
                if head in fv:
                    #print feat, head
                    remove.add(head)
        for feat in remove:
            #print "Removing", feat
            del(fv[feat])
        return fv
    pass


def extract_text(domain):
    FeatGen = FEATURE_GENERATOR()
    lines = []
    for (mode, label) in [("train", "positive"), ("train", "negative"), ("train", "unlabeled"),
                            ("test", "positive"), ("test", "negative")]:
        fname = "../data/%s-data/%s/%s.review" % (mode, domain, label)
        lines.extend(FeatGen.get_text(fname))

    # clean and write lines to a file
    table = string.maketrans("", "")
    with open("../work/%s/all.text" % domain, 'w') as F:
        for line in lines:            
            elements = re.sub("\d+", "NUM", line.strip().lower().translate(table, string.punctuation)).split()
            words = []
            for word in elements:
                if word not in FeatGen.stopWords:
                    words.append(word)
            n = len(words)
            i = 0
            while(i < n):
                if words[i] == "not":
                    if i < (n - 1):
                        F.write("not__%s" % words[i+1])
                        i += 1
                    else:
                        F.write("%s" % word)                        
                else:
                    F.write("%s" % words[i])
                if i == (n - 1):
                    F.write("\n")
                else:
                    F.write(" ")
                i += 1
    pass


def generateFeatureVectors(domain):
    """
    Create feature vectors for each review in the domain. 
    """
    extract_text(domain)
    FeatGen = FEATURE_GENERATOR()
    for (mode, label) in [("train", "positive"), ("train", "negative"), ("train", "unlabeled"),
                            ("test", "positive"), ("test", "negative")]:
        fname = "../data/%s-data/%s/%s.review" % (mode, domain, label)
        fvects = FeatGen.process_file(fname, label)
        writeFeatureVectorsToFile(fvects, "../work/%s/%s.%s" % (domain, mode, label))   
    pass


def writeFeatureVectorsToFile(fvects, fname):
    """
    Write each feature vector in fvects in a single line in fname. 
    """
    F = open(fname, 'w')
    for e in fvects:
        for w in e[1].keys():
            F.write("%s " % w)
        F.write("\n")
    F.close()
    pass


def getCounts(S, M, fname):
    """
    Get the feature co-occurrences in the file fname and append 
    those to the dictionary M. We only consider features in S.
    """
    count = 0
    F = open(fname)
    for line in F:
        count += 1
        #if count > 1000:
        #   break
        allP = line.strip().split()
        p = []
        for w in allP:
            if w in S:
                if w not in p:
                    p.append(w) 
        n = len(p)
        for i in range(0,n):
            for j in range(i + 1, n):
                pair = (p[i], p[j])
                rpair = (p[j], p[i])
                if pair in M:
                    M[pair] += 1
                elif rpair in M:
                    M[rpair] += 1
                else:
                    M[pair] = 1
    F.close()
    pass


def getVocab(S, fname):
    """
    Get the frequency of each feature in the file named fname. 
    """
    F = open(fname)
    for line in F:
        p = line.strip().split()
        for w in p:
            S[w] = S.get(w, 0) + 1
    F.close()
    pass


def selectTh(h, t):
    """
    Select all elements of the dictionary h with frequency greater than t. 
    """
    p = {}
    for (key, val) in h.iteritems():
        if val > t:
            p[key] = val
    del(h)
    return p


def getVal(x, y, M):
    """
    Returns the value of the element (x,y) in M.
    """
    if (x,y) in M:
        return M[(x,y)] 
    elif (y,x) in M:
        return M[(y,x)]
    else:
        return 0
    pass


def select_pivots_nonpivots(source, target):
    """
    Read the unlabeled data (test and train) for both source and the target domains. 
    Compute the full co-occurrence matrix. Drop co-occurrence pairs with a specified
    minimum threshold. For a feature w, compute its score(w),
    score(w) = pmi(w, source_domain_label) + pmi(w, target_domain_label)
    and sort the features in the descending order of their scores. 
    """
    # Parameters
    if source == "dvd":
        SourceFreqTh = 50
    else:
        SourceFreqTh = 50

    if target == "dvd":
        TargetFreqTh = 50
    else:
        TargetFreqTh = 50

    coocTh = 1
    noPivots = 500
    noSourceSpecific = 500
    noTargetSpecific = 500

    print "Source = %s, Target = %s" % (source, target)

    # Get the set of source domain features.
    S = {}
    getVocab(S, "../work/%s/all.text" % source)
    print "Total source features =", len(S)
    # Remove source domain features with total frequency less than SourceFreqTh
    S = selectTh(S, SourceFreqTh)
    print "After thresholding at %d we have = %d" % (SourceFreqTh, len(S))

    # Get the set of target domain features.
    T = {}
    getVocab(T, "../work/%s/all.text" % target)
    print "Total target features =", len(T)
    # Remove target domain features with total frequency less than TargetFreqTh
    T = selectTh(T, TargetFreqTh)
    print "After thresholding at %d we have = %d" % (TargetFreqTh, len(T))  

    # Compute the co-occurrences of features in reviews
    src_M = {}
    tgt_M = {}
    print "Creating source co-occurrence matrix ...",
    getCounts(S, src_M, "../work/%s/all.text" % source)
    print "Done."
    print "Creating target co-occurrence matrix ...",
    getCounts(T, tgt_M, "../work/%s/all.text" % target)
    print "Done."

    # Compute the intersection of source and target domain features.
    pivots = set(S.keys()).intersection(set(T.keys()))
    print "Total no. of pivots =", len(pivots)

    labels = {'books':'book', "electronics":"electronic", "dvd":"dvd", "kitchen":"kitchen"}
    # Compute PMI scores for pivots.
    C = {}
    src_N = sum(S.values())
    tgt_N = sum(T.values())
    i = 0
    for pivot in pivots:
        C[pivot] = 0.0
        i += 1
        src_val = getVal(pivot, labels[source], src_M)  
        tgt_val = getVal(pivot, labels[target], tgt_M)  
        if src_val > coocTh and tgt_val > coocTh:
            C[pivot] += getPMI(tgt_val, T[labels[target]], T[pivot], tgt_N) + getPMI(src_val, S[labels[source]], S[pivot], src_N) 
        if i % 500 == 0:
            print "%d: pivot = %s, MI = %.4g" % (i, pivot, C[pivot])
    pivotList = C.items()
    pivotList.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    # write pivots to a file.
    pivotsFile = open("../work/%s-%s/pivots" % (source, target), 'w')
    DI = []
    for (i, (w, v)) in enumerate(pivotList[:noPivots]):
        pivotsFile.write("%d %s P %s\n" % (i+1, w, str(v))) 
        DI.append(w)
    pivotsFile.close()

    # Select source specific features
    srcSpecificFeatures = {}
    featCount = 0
    for w in S:
        if w in DI:
            continue
        featCount += 1
        if w not in T:
            val = getVal(w, labels[source], src_M)
            if (val > coocTh):
                srcSpecificFeatures[w] = getPMI(val, S[labels[source]], S[w], src_N)
        print "SS %d of %d %s %f" % (featCount, len(S), w, srcSpecificFeatures.get(w, 0))

    # Select target specific features
    tgtSpecificFeatures = {}
    featCount = 0
    for w in T:
        if w in DI:
            continue
        featCount += 1
        if w not in S:
            val = getVal(w, labels[target], tgt_M)
            if (val > coocTh):
                tgtSpecificFeatures[w] = getPMI(val, T[labels[target]], T[w], tgt_N)
        print "TS %d of %d %s %f" % (featCount, len(T), w, tgtSpecificFeatures.get(w, 0))

    srcList = srcSpecificFeatures.items()
    srcList.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    tgtList = tgtSpecificFeatures.items()
    tgtList.sort(lambda x, y: -1 if x[1] > y[1] else 1)

    # Domain specific feature list.
    DSFile = open("../work/%s-%s/nonpivots" % (source, target), 'w')
    DS = []
    count = 0
    for (f,v) in srcList[:noSourceSpecific]:
        count += 1
        DS.append(f)
        DSFile.write("%d %s S %s\n" % (count, f, str(v)))
    for (f,v) in tgtList[:noTargetSpecific]:
        count += 1
        DS.append(f)
        DSFile.write("%d %s T %s\n" % (count, f, str(v)))
    DSFile.close() 
    print "Total nonpivots =", count
    pass


def getPMI(n, x, y, N):
    """
    Compute the weighted PMI value. 
    """
    pmi =  math.log((float(n) * float(N)) / (float(x) * float(y)))
    norm = -math.log(float(n) / float(N))
    res = pmi / norm
    return res


def create_corpus(source, target):
    """
    Merge the two all.text files so that we can run glove later.
    """
    corpus_file = open("../work/%s-%s/merged.txt" % (source, target), 'w')
    source_file = open("../work/%s/all.text" % source)
    for line in source_file:
        corpus_file.write("%s" % line)
    source_file.close()
    target_file = open("../work/%s/all.text" % target)
    for line in target_file:
        corpus_file.write("%s" % line)
    target_file.close()
    corpus_file.close()
    pass


def generateAll():
    """
    Generate matrices for all pairs of domains. 
    """
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source == target:
                continue
            #select_pivots_nonpivots(source, target)
            #create_train_data(source, target)
            create_corpus(source, target)
    pass


def get_average_words(fname):
    """
    Compute the average number of words per line in a file
    """
    with open(fname) as F:
        count = 0
        n = 0
        for line in F:
            n += 1
            count += len(line.strip().split()) 
        print "Average no. of words per line =", (float(count) / float(n)), count, n
    pass


def load_pivots(fname, domain=None):
    """
    Returns a list of pivots (or non-pivots). 
    """
    L = []
    with open(fname) as F:
        for line in F:
            p = line.strip().split()
            d = p[2].strip()
            w = p[1].strip()
            if domain is not None:
                if domain == d:
                    L.append(w)
            else:
                L.append(w)
    return L


def get_unigram_distribution(domain, pivots):
    """
    Get the probability distribution of unigrams for 
    the given domain. We will sample from this distribution. 
    """
    h = OrderedDict()
    with open("../work/%s/all.text" % domain) as txt_file:
        for line in txt_file:
            for w in line.strip().split():
                if w in pivots:
                    h[w] = h.get(w, 0.0) + 1.0
    power = 0.75
    for w in h:
        h[w] = h[w] ** power 
    total = sum(h.values())
    for w in h:
        h[w] = h[w] / total
    return h


def create_train_data(source, target):
    """
    Select positive training data from contexts where pivots occur,
    and randomly sample from the unigram distribution over pivots 
    to select negative training instances. 
    """
    print source, target
    k = 5  # ratio between negative vs. positive instances 
    pivots = load_pivots("../work/%s-%s/pivots" % (source, target))
    source_nonpivots = load_pivots("../work/%s-%s/nonpivots" % (source, target), domain="S")
    target_nonpivots = load_pivots("../work/%s-%s/nonpivots" % (source, target), domain="T")
    print "Total no. of pivots =", len(pivots)
    print "Total no. of source nonpivots =", len(source_nonpivots)
    print "Total no. of target nonpivots =", len(target_nonpivots)
    source_dist = get_unigram_distribution(source, source_nonpivots)
    target_dist = get_unigram_distribution(target, target_nonpivots)
    extract_contexts(source, pivots, source_nonpivots, "../work/%s-%s/source.train" % (source, target), source_dist, k)
    extract_contexts(target, pivots, target_nonpivots, "../work/%s-%s/target.train" % (source, target), target_dist, k)
    pass


def get_sample(h, n):
    """
    Get a sample of n elements from the distribution specified by h. 
    """
    return numpy.random.choice(h.keys(), size=n, p=h.values())


def extract_contexts(domain, pivots, nonpivots, train_fname, source_dist, k):
    """
    Extract positive and negative instances for a domain. 
    """
    window_size = 5
    vocab = set(pivots).union(set(nonpivots))
    contexts_file = open("../work/%s/all.text" % domain)
    train_file = open(train_fname, 'w')
    for line in contexts_file:
        words = line.strip().split()
        L = filter(lambda x: x in vocab, words)
        for pivot in pivots:
            if pivot in L:
                pivot_ind = L.index(pivot)
                contexts = []
                if pivot_ind != -1:
                    for c in L[max(0, pivot_ind - window_size): min(len(L), pivot_ind + window_size)]:
                        if c not in contexts and c not in pivots:
                            contexts.append(c)
                if len(contexts) > 0:
                    # negative sampling
                    negatives = []
                    while len(negatives) == 0:
                        negatives = filter(lambda x: x not in words, get_sample(source_dist, (k * len(contexts))))
                    train_file.write("%s\t%s\t%s\t\n" % (pivot, ",".join(contexts), ",".join(negatives)))
    contexts_file.close()
    train_file.close()
    pass



if __name__ == "__main__":
    #get_average_words(sys.argv[1].strip())
    #generateFeatureVectors("books")
    #generateFeatureVectors("dvd")
    #generateFeatureVectors("electronics")
    #generateFeatureVectors("kitchen")
    generateAll()
    #select_pivots_nonpivots("kitchen", "books")
    #create_train_data("kitchen", "books")
    pass

