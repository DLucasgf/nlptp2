import nltk
from nltk.corpus import movie_reviews

doc = '''O Yamato foi um navio couraçado operado pela Marinha Imperial Japonesa na Segunda Guerra Mundial e construído pelos estaleiros do Arsenal Naval de Kure. Foi a primeira embarcação da Classe Yamato, sendo junto com seu irmão Musashi os mais pesados e poderosos navios de guerra já construídos na história.

Nomeado em homenagem à província de Yamato, ele foi projetado para combater a frota de couraçados numericamente superior da Marinha dos Estados Unidos, o principal rival do Império do Japão no oceano Pacífico. A construção do Yamato começou em novembro de 1937 e ele foi formalmente comissionado uma semana depois do Ataque a Pearl Harbor em dezembro de 1941. A embarcação serviu como nau-capitânia da Frota Combinada pelo ano de 1942, com o almirante Isoroku Yamamoto comandando a frota de sua ponte durante a desastrosa Batalha de Midway. O Musashi assumiu a liderança no início do ano seguinte e o Yamato passou 1943 e boa parte de 1944 movendo-se entre as bases Truk e Kure, principalmente respondendo a ameaças norte-americanas. O navio esteve presente na Batalha do Mar das Filipinas em junho de 1944, porém não participou do embate.

A única vez que o Yamato disparou seus canhões principais contra alvos inimigos foi em outubro de 1944, quando foi enviado para enfrentar forças norte-americanas que estavam invadindo as Filipinas na Batalha do Golfo de Leyte. As embarcações japonesas acabaram recuando bem quando estavam à beira da vitória, acreditando na verdade estarem enfrentando uma frota inteira de porta-aviões em vez dos pequenos porta-aviões de escolta que eram a única coisa que separava os couraçados dos principais navios de transporte de tropas.

O equilíbrio de poder no Pacífico ficou definitivamente contra os japoneses ao longo de 1944, com a frota estando assolada no início do ano seguinte pela falta de suprimentos e combustível. O Yamato foi enviado para Okinawa em abril de 1945 em uma tentativa desesperada de conter o avanço norte-americano, recebendo ordens para proteger a ilha até a morte. Submarinos e aeronaves inimigas avistaram a força tarefa ao sul de Kyūshū, com o couraçado sendo afundado por bombardeiros e torpedeiros junto com a maior parte de sua tripulação.'''

documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]


all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
 	
#print(document_features(movie_reviews.words('pos/cv957_8737.txt')))
print(document_features(doc))