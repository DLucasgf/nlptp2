import nltk
from nltk.corpus import brown
from nltk.corpus import mac_morpho

suffix_fdist = nltk.FreqDist()
for word in mac_morpho.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1

common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(10)]
print(common_suffixes)

def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
    return features

tagged_words = mac_morpho.tagged_words(categories='news')
featuresets = [(pos_features(n), g) for (n,g) in tagged_words]
 	
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
 	
classifier = nltk.DecisionTreeClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))