"""
Perform cross-domain sentiment classification using word embeddings.
"""

import subprocess
import numpy
import sklearn.preprocessing
from wordreps import WordReps
import sys


def train_LBFGS(train_file, model_file):
    """
    Train lbfgs on train file. and evaluate on test file.
    Read the output file and return the classification accuracy.
    """
    retcode = subprocess.call("classias-train -tb -a lbfgs.logistic -pc1=0 -pc2=1 -m %s %s > /dev/null"  %
                            (model_file, train_file), shell=True)
    return retcode


def test_LBFGS(test_file, model_file):
    """
    Evaluate on the test file.
    Read the output file and return the classification accuracy.
    """
    output = "../work/output"
    subprocess.call("cat %s | classias-tag -m %s -t > %s" % (test_file, model_file, output), shell=True)
    F = open(output)
    for line in F:
        if line.startswith("Accuracy"):
            p = line.strip().split()
            accuracy = float(p[1])
    F.close()
    return accuracy


def load_model(fname):
    """
    Load the word representations learnt using the C++ version.
    """
    w_S = {}
    c_S = {}
    w_T = {}
    c_T = {}
    with open(fname) as F:
        for line in F:
            p = line.strip().split()
            if p[1] == 'cS':
                c_S[p[0]] = numpy.array(map(float, p[2:]))
            if p[1] == 'cT':
                c_T[p[0]] = numpy.array(map(float, p[2:]))
            if p[1] == 'wS':
                w_S[p[0]] = numpy.array(map(float, p[2:]))
            if p[1] == 'wT':
                w_T[p[0]] = numpy.array(map(float, p[2:]))
    return (w_S, c_S, w_T, c_T)


def load_reviews(fname, label, vocab):
    """
    Load reviews from a file where each line is a review.
    """
    h = []
    with open(fname) as F:
        for line in F:
            v = {}
            for token in line.strip().split():
                if vocab is None:
                    #v[token] = v.get(token, 0) + 1
                    v[token] = 1
                else:
                    if token in vocab:
                        #v[token] = v.get(token, 0) + 1
                        v[token] = 1
            h.append({"label":label, "fvect":v})
    return h


def write_feat_file(fname, vects):
    """
    Write the features to file. 
    """
    with open(fname, 'w') as feat_file:
        for v in vects:
            feat_file.write("%d " % v["label"])
            for (feat, fval) in v["fvect"].items():
                feat_file.write("%s:%f " % (feat, fval))
            feat_file.write("\n")
    pass


def neighbour_feats(x, c_T, w_T, cands, feats):
    """
    Compute the centroid of the target test review x using c_S and c_T.
    Let this centroid be r. Compute the cosine similarity between r and
    all the features in the cands. Append the cands with positive similarity
    scores as features.
    """
    r = numpy.zeros(cands.shape[1], dtype=float)
    y = {"label":x["label"], "fvect":{}}
    for (feat, fval) in x["fvect"].items():
        y["fvect"][feat] = fval
        if feat in c_T:
            r += (c_T[feat] * fval)
        elif feat in w_T:
            r += (w_T[feat] * fval)
    if len(x["fvect"]) != 0:
        r = (1.0 / float(len(x["fvect"]))) * r
    rnorm = numpy.linalg.norm(r)
    if rnorm != 0:
        r = r / rnorm
    scores = numpy.dot(cands, r)
    score_norm = numpy.linalg.norm(scores)
    if score_norm != 0:
        scores = scores / score_norm
    for (i, feat) in enumerate(feats):
        if scores[i] > 0:
            y["fvect"][feat] = y["fvect"].get(feat, 0) + scores[i]
    return y


def append_dim_feats(vects, c_S, c_T):
    """
    Add each dimension as a feature from source and target
    pivot representations.
    """
    L = []
    for h in vects:
        v = {"label":h["label"], "fvect":{}}
        for (feat, fval) in h["fvect"].items():
            v["fvect"][feat] = fval 
            if feat in c_S:
                for i in range(0, len(c_S[feat])):
                    v["fvect"]["%s_cS_%d" % (feat, i)] = (c_S[feat][i] * fval) 
                    pass
            if feat in c_T:
                for i in range(0, len(c_T[feat])):
                    v["fvect"]["%s_cT_%d" % (feat, i)] = (c_T[feat][i] * fval)
                    pass
        L.append(v)
    return L


def append_dim_feats_all(vects, c_S, c_T, w_S, w_T):
    """
    Add each dimension as a feature from source and target
    pivot representations. We add features for all words, not only pivots.
    """
    L = []
    for h in vects:
        v = {"label":h["label"], "fvect":{}}
        for (feat, fval) in h["fvect"].items():
            v["fvect"][feat] = fval * 1

            # if feat in w_S:
            #     for i in range(0, len(w_S[feat])):
            #         v["fvect"]["%s+%d" % (feat, i)] = (w_S[feat][i] * fval)
            #         pass
            if feat in w_T:
                for i in range(0, len(w_T[feat])):
                    v["fvect"]["%s+%d" % (feat, i)] = (w_T[feat][i] * fval)
                    pass
            
            # if feat in c_S:
            #     for i in range(0, len(c_S[feat])):
            #         v["fvect"]["%s+cS+%d" % (feat, i)] = (c_S[feat][i] * fval)
            #         pass

            if feat in c_T:
                for i in range(0, len(c_T[feat])):
                    v["fvect"]["%s+cT+%d" % (feat, i)] = (c_T[feat][i] * fval)
                    pass
        L.append(v)
    return L


def append_wordreps(vects, WR):
    """
    Use word reps.
    """
    L = []
    for h in vects:
        v = {"label":h["label"], "fvect":{}}
        for (feat, fval) in h["fvect"].items():
            v["fvect"][feat] = fval * 1
            if feat in WR.vects:
                x = WR.vects[feat]            
                for i in range(0, len(x)):
                    v["fvect"]["%s+cS+%d" % (feat, i)] = (x[i] * fval)
        L.append(v)
    return L


def normalize(d):
    """
    Normalize the word vectors to unit L2 length.
    """
    D = len(d.values()[0])
    N = len(d)
    A = numpy.zeros((N,D), dtype=numpy.float64)
    words = []
    count = 0
    for w in d:
        words.append(w)
        A[count,:] = d[w]
        count += 1
    A = sklearn.preprocessing.scale(A)
    for (i,w) in enumerate(words):
        d[w] = A[i,:]
    return d


def load_pivots(fname, domain):
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


def get_vocab(w_S, c_S, w_T, c_T):
    vocab = set()
    for w in w_S:
        vocab.add(w)
    for w in c_S:
        vocab.add(w)
    for w in w_T:
        vocab.add(w)
    for w in c_T:
        vocab.add(w)
    return vocab


def no_adapt_baseline(source, target):
    """
    Perform cross-domain sentiment classification. 
    """
    base_path = "../work/%s-%s/" % (source, target)
    pivots = set(load_pivots(base_path + "/pivots", "P"))
    src_nonpivots = set(load_pivots(base_path + "/nonpivots", "S"))
    tgt_nonpivots = set(load_pivots(base_path + "/nonpivots", "T"))
    vocab = set()
    vocab = vocab.union(pivots)
    vocab = vocab.union(src_nonpivots)
    vocab = vocab.union(tgt_nonpivots)

    pos_source = load_reviews("../work/%s/train.positive" % source, 1, vocab)
    neg_source = load_reviews("../work/%s/train.negative" % source, -1, vocab)
    pos_target = load_reviews("../work/%s/test.positive" % target, 1, vocab)
    neg_target = load_reviews("../work/%s/test.negative" % target, -1, vocab)
    source_vects = pos_source[:]
    source_vects.extend(neg_source)
    target_vects = pos_target[:]
    target_vects.extend(neg_target)
    train_file = "../work/train_file"
    test_file = "../work/test_file"
    model_file = "../work/model.classias"
    print "Total train instances =", len(source_vects)
    print "Total test instances =", len(target_vects)
    write_feat_file(train_file, source_vects)
    write_feat_file(test_file, target_vects)
    train_LBFGS(train_file, model_file)
    acc = test_LBFGS(test_file, model_file)
    print "%s --> %s Acc = %f" % (source, target, acc)
    pass


def within_baseline(source, target):
    """
    Perform cross-domain sentiment classification. 
    """
    base_path = "../work/%s-%s/" % (source, target)
    pivots = set(load_pivots(base_path + "/pivots", "P"))
    src_nonpivots = set(load_pivots(base_path + "/nonpivots", "S"))
    tgt_nonpivots = set(load_pivots(base_path + "/nonpivots", "T"))
    vocab = set()
    vocab = vocab.union(pivots)
    vocab = vocab.union(src_nonpivots)
    vocab = vocab.union(tgt_nonpivots)

    pos_source = load_reviews("../work/%s/train.positive" % source, 1, vocab)
    neg_source = load_reviews("../work/%s/train.negative" % source, -1, vocab)
    pos_target = load_reviews("../work/%s/test.positive" % source, 1, vocab)
    neg_target = load_reviews("../work/%s/test.negative" % source, -1, vocab)
    source_vects = pos_source[:]
    source_vects.extend(neg_source)
    target_vects = pos_target[:]
    target_vects.extend(neg_target)
    train_file = "../work/train_file"
    test_file = "../work/test_file"
    model_file = "../work/model.classias"
    print "Total train instances =", len(source_vects)
    print "Total test instances =", len(target_vects)
    write_feat_file(train_file, source_vects)
    write_feat_file(test_file, target_vects)
    train_LBFGS(train_file, model_file)
    acc = test_LBFGS(test_file, model_file)
    print "%s --> %s Acc = %f" % (source, target, acc)
    pass


def source_target_expansion(source, target):
    """
    Perform cross-domain sentiment classification. 
    """
    base_path = "../work/%s-%s/" % (source, target)
    model_fname = base_path + "model.l=100.dim=1000"
    #print "WordReps =", model_fname
    #print "Loading data...",
    sys.stdout.flush()
    w_S, c_S, w_T, c_T = load_model(model_fname)
    D = len(w_S.values()[0])
    #print "Done DIM = %d" % D
    sys.stdout.flush()
    #vocab = get_vocab(w_S, c_S, w_T, c_T)
    vocab = None
    pos_source = load_reviews("../work/%s/train.positive" % source, 1, vocab)
    neg_source = load_reviews("../work/%s/train.negative" % source, -1, vocab)
    pos_target = load_reviews("../work/%s/test.positive" % target, 1, vocab)
    neg_target = load_reviews("../work/%s/test.negative" % target, -1, vocab)
    source_vects = pos_source[:]
    source_vects.extend(neg_source)
    target_vects = pos_target[:]
    target_vects.extend(neg_target)
    train_file = "../work/train_file"
    test_file = "../work/test_file"
    model_file = "../work/model.classias"
    #print "Total train instances =", len(source_vects)
    #print "Total test instances =", len(target_vects)

    #source_vects = append_dim_feats_all(source_vects, c_S, c_T, w_S, w_T)
    #target_vects = append_dim_feats_all(target_vects, c_S, c_T, w_S, w_T)

    cands = numpy.zeros((len(w_S) + len(c_T), D), dtype=float)
    i = 0
    feats = []
    for w in w_S:
        feats.append(w)
        norm = numpy.linalg.norm(w_S[w])
        if norm != 0:
            cands[i, :] = w_S[w] / norm
        i += 1
    for w in c_T:
        feats.append(w)
        norm = numpy.linalg.norm(c_T[w])
        if norm != 0:
            cands[i, :] = c_T[w] / norm
        i += 1

    for i in range(len(target_vects)):
        target_vects[i] = neighbour_feats(target_vects[i], c_T, w_T, cands, feats)

    write_feat_file(train_file, source_vects)
    write_feat_file(test_file, target_vects)
    train_LBFGS(train_file, model_file)
    acc = test_LBFGS(test_file, model_file)
    print "%s --> %s Acc = %f" % (source, target, acc)
    pass


def glove_baseline(source, target):
    """
    Use glove trained word representations.
    """
    D = 1000
    base_path = "../work/%s-%s/" % (source, target)
    WR = WordReps()
    WR.read_model("%s/glove%d.txt" % (base_path, D), D)
    pivots = set(load_pivots(base_path + "/pivots", "P"))
    src_nonpivots = set(load_pivots(base_path + "/nonpivots", "S"))
    tgt_nonpivots = set(load_pivots(base_path + "/nonpivots", "T"))
    vocab = set()
    vocab = vocab.union(pivots)
    vocab = vocab.union(src_nonpivots)
    vocab = vocab.union(tgt_nonpivots)
    pos_source = load_reviews("../work/%s/train.positive" % source, 1, vocab)
    neg_source = load_reviews("../work/%s/train.negative" % source, -1, vocab)
    pos_target = load_reviews("../work/%s/test.positive" % target, 1, vocab)
    neg_target = load_reviews("../work/%s/test.negative" % target, -1, vocab)
    source_vects = pos_source[:]
    source_vects.extend(neg_source)
    target_vects = pos_target[:]
    target_vects.extend(neg_target)
    train_file = "../work/train_file"
    test_file = "../work/test_file"
    model_file = "../work/model.classias"
    #print "Total train instances =", len(source_vects)
    #print "Total test instances =", len(target_vects)
    #source_vects = append_wordreps(source_vects, WR)
    #target_vects = append_wordreps(target_vects, WR)

    cands = numpy.zeros((len(pivots) + len(src_nonpivots), D), dtype=float)
    i = 0
    feats = []
    for w in pivots:
        feats.append(w)
        if w in WR.vects:
            cands[i, :] = WR.vects[w]
        i += 1
    for w in src_nonpivots:
        feats.append(w)
        if w in WR.vects:
            cands[i, :] = WR.vects[w]
        i += 1

    for i in range(len(target_vects)):
        x = {"label": target_vects[i]["label"], "fvect":{}}
        r = numpy.zeros(D, dtype=float)
        for (feat, fval) in target_vects[i]["fvect"].items():
            x["fvect"][feat] = fval
            if feat in WR.vects:
                r += WR.vects[feat]
        r = r / float(len(target_vects[i]["fvect"]))
        r = r / numpy.linalg.norm(r)
        scores = numpy.dot(cands, r)
        scores = scores / numpy.linalg.norm(scores)
        for (j, feat) in enumerate(feats):
            if scores[i] > 0:
                x["fvect"][feat] = x["fvect"].get(feat, 0) + scores[j]
        target_vects[i] = x

    write_feat_file(train_file, source_vects)
    write_feat_file(test_file, target_vects)
    train_LBFGS(train_file, model_file)
    acc = test_LBFGS(test_file, model_file)
    print "%s --> %s Acc = %f" % (source, target, acc)
    pass


def batch_mode():
    domains = ["books", "electronics", "dvd", "kitchen"]
    for target in domains:
        for source in domains:
            if source != target:
                #no_adapt_baseline(source, target)
                #glove_baseline(source, target)
                source_target_expansion(source, target)
    pass

if __name__ == "__main__":
    #no_adapt_baseline("books", "electronics")
    #source_target_expansion("books", "electronics")
    glove_baseline("books", "electronics")
    #batch_mode()
    #within_baseline("dvd", "kitchen")

