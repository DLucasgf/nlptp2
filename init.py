import nltk


print(len(nltk.corpus.mac_morpho.words()))


#train_data = nltk.corpus.mac_morpho.tagged_sents()[:3000]
#test_data = nltk.corpus.mac_morpho.tagged_sents()[3000:]
#print(train_data[0])



train = [
        (dict(a="Jersei"), "N"),
        (dict(a="atinge"), "V"),
        (dict(a="média"), "N"),
        (dict(a="de"), "PREP"),
        (dict(a="Cr$"), "CUR"),
        (dict(a="milhão"), "N"),
        (dict(a="na"), "PREP+ART"),
        (dict(a="venda"), "N"),
        (dict(a="da"), "PREP+ART"),
        (dict(a="Pinhal"), "NPROP"),
        (dict(a="em"), "PREP"),
        (dict(a="São"), "NPROP"),
        (dict(a="Paulo"), "NPROP")
    ]

test = [
        (dict(a="O")),
        (dict(a="grande")),
        (dict(a="assunto"))
]


classifier = nltk.classify.NaiveBayesClassifier.train(labeled_featuresets=train)
print(classifier.labels())


for pdist in classifier.prob_classify_many(test):
    print('%.4f %.4f' % (pdist.prob('x'), pdist.prob('y')))

print(nltk.tag.str2tuple('fly_NN'))