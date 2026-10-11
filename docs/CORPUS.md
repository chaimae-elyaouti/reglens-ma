# Corpus : sources, version, et statut de la loi

## V1 : Code du travail (loi n° 65-99)

- Promulgué par le dahir n° 1-03-194 du 11 septembre 2003. La presse de l'époque indique 589 articles.
  Le rapport qualité (`index/v1/quality.json`) compare le nombre d'articles trouvés à `expected_articles`.
- Deux PDF candidats sont listés dans `data/sources.json` (portails justice.gov.ma). **Ils ont été trouvés par
  recherche web mais pas téléchargés ni vérifiés par l'auteur de ce dépôt.** Compare-les (articles trouvés,
  numéros manquants, qualité du texte) puis n'en active qu'un.
- Renseigne `as_of` : la date de la version du texte (à lire dans le PDF / le Bulletin officiel). Elle est
  affichée à l'utilisateur dans l'avertissement de chaque réponse.

## ⚠️ La loi peut être en train de changer

Un article de presse spécialisé (wafir.ma, source secondaire, **non vérifiée**) indique qu'un projet de loi 03-26
réformant le Code du travail aurait été adopté en Conseil des ministres le 14 mai 2026. **Vérifie sur des
sources officielles** (Secrétariat général du gouvernement, Chambre des représentants, Bulletin officiel) où en est
le texte : adopté ? publié ? en vigueur ? Conséquences pour le projet :

- chaque réponse affiche la version du texte indexé (`corpus_as_of`) ;
- si une nouvelle version entre en vigueur, ajoute un NOUVEAU document (`doc_id` différent, ex. `code-travail-2026`)
  avec sa date d'effet plutôt que d'écraser l'ancien, et décide quelle version interroger par défaut (filtre `doc_id`) ;
- ne présente jamais l'outil comme à jour sans l'avoir vérifié.

## Textes complémentaires possibles (V2, à vérifier avant d'ajouter)

Le Code renvoie à des textes spéciaux (ex. article 4 : employés de maison, loi spéciale). Pour des questions
réelles (accident du travail, harcèlement, sécurité sociale, travailleurs domestiques), il faudra probablement
d'autres textes : à identifier et à vérifier un par un (existence, version en vigueur, licence). Ajoute-les un à
un, avec leurs questions d'évaluation.

## Licence et republication

Vérifie les conditions de réutilisation des textes officiels avant de publier les PDF ou l'index
(`index/*/chunks.jsonl` contient le texte). Par défaut, `data/raw/` et `index/` sont ignorés par Git.
