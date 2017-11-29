import nltk
from nltk.corpus import brown
from nltk.corpus import mac_morpho

doc = '''O Yamato foi um navio couraçado operado pela Marinha Imperial Japonesa na Segunda Guerra Mundial e construído pelos estaleiros do Arsenal Naval de Kure. Foi a primeira embarcação da Classe Yamato, sendo junto com seu irmão Musashi os mais pesados e poderosos navios de guerra já construídos na história.

Nomeado em homenagem à província de Yamato, ele foi projetado para combater a frota de couraçados numericamente superior da Marinha dos Estados Unidos, o principal rival do Império do Japão no oceano Pacífico. A construção do Yamato começou em novembro de 1937 e ele foi formalmente comissionado uma semana depois do Ataque a Pearl Harbor em dezembro de 1941. A embarcação serviu como nau-capitânia da Frota Combinada pelo ano de 1942, com o almirante Isoroku Yamamoto comandando a frota de sua ponte durante a desastrosa Batalha de Midway. O Musashi assumiu a liderança no início do ano seguinte e o Yamato passou 1943 e boa parte de 1944 movendo-se entre as bases Truk e Kure, principalmente respondendo a ameaças norte-americanas. O navio esteve presente na Batalha do Mar das Filipinas em junho de 1944, porém não participou do embate.

A única vez que o Yamato disparou seus canhões principais contra alvos inimigos foi em outubro de 1944, quando foi enviado para enfrentar forças norte-americanas que estavam invadindo as Filipinas na Batalha do Golfo de Leyte. As embarcações japonesas acabaram recuando bem quando estavam à beira da vitória, acreditando na verdade estarem enfrentando uma frota inteira de porta-aviões em vez dos pequenos porta-aviões de escolta que eram a única coisa que separava os couraçados dos principais navios de transporte de tropas.

O equilíbrio de poder no Pacífico ficou definitivamente contra os japoneses ao longo de 1944, com a frota estando assolada no início do ano seguinte pela falta de suprimentos e combustível. O Yamato foi enviado para Okinawa em abril de 1945 em uma tentativa desesperada de conter o avanço norte-americano, recebendo ordens para proteger a ilha até a morte. Submarinos e aeronaves inimigas avistaram a força tarefa ao sul de Kyūshū, com o couraçado sendo afundado por bombardeiros e torpedeiros junto com a maior parte de sua tripulação.'''

def pos_features(sentence, i):
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
    return features
 	

pos_features(mac_morpho.sents(), 8)
tagged_sents = mac_morpho.tagged_sents()
featuresets = []
for tagged_sent in tagged_sents:
    untagged_sent = nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        featuresets.append( (pos_features(untagged_sent, i), tag) )
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print('pos')
print(pos_features(doc, 8))

print('mac classify')
#print(nltk.classify.accuracy(classifier, test_set))

print('mac many')
print(classifier.classify('test_set'))