import nltk
from nltk.corpus import brown

suffix_fdist = nltk.FreqDist()
for word in brown.words():
    word = word.lower()
    suffix_fdist[word[-1:]] += 1
    suffix_fdist[word[-2:]] += 1
    suffix_fdist[word[-3:]] += 1

common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]

def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
    return features


tagged_words = brown.tagged_words(categories='news')[4:50]
featuresets = [(pos_features(n), g) for (n,g) in tagged_words[4:50]]
 	
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
 	
classifier = nltk.DecisionTreeClassifier.train(train_set)
print("accuracy")
print(nltk.classify.accuracy(classifier, test_set))

print("classify")
print(classifier.classify(pos_features('cats')))

print("most")
print(classifier.show_most_informative_features(5))