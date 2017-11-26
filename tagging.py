import nltk
import nltk.tag
import itertools
from nltk.corpus import mac_morpho

mac_tagged_sents = mac_morpho.tagged_sents()
mac_sents = mac_morpho.sents()

size = int(len(mac_tagged_sents) * 0.9)

train_sents = mac_tagged_sents[:size]
test_sents = mac_tagged_sents[size:]

unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))

#unigram_tagger = nltk.UnigramTagger(mac_tagged_sents)
#print(unigram_tagger.tag(mac_sents[2007]))

#print(unigram_tagger.evaluate(mac_tagged_sents))

#tags = [tag for (word, tag) in mac_morpho.tagged_words()]
#print(nltk.FreqDist(tags).max())
