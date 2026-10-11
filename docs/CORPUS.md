# Corpus : sources, version, et statut de la loi

## Corpus indexé : Code du travail (loi n° 65-99)

L'index local utilise le PDF du Code du travail référencé comme `code-travail-adala` dans `data/sources.json`.
Le rapport qualité de l'index compare les articles extraits au nombre attendu (`expected_articles`).

La date de la version juridique n'est pas renseignée (`as_of` est vide dans `data/sources.json`) et l'actualité du
texte n'a pas été vérifiée à partir du Bulletin officiel. Les réponses affichent donc la version connue de l'index,
mais le projet ne doit pas être présenté comme une source juridique à jour.

Avant une mise à jour du corpus, vérifier le texte et sa date d'effet dans une source officielle. Pour conserver
la traçabilité, ajouter une nouvelle version avec un `doc_id` distinct plutôt que remplacer silencieusement
l'ancienne, puis reconstruire l'index et relancer l'évaluation.

## Périmètre et extensions

Le Code renvoie à des textes spéciaux (ex. article 4 : employés de maison, loi spéciale). Pour des questions
réelles (accident du travail, harcèlement, sécurité sociale, travailleurs domestiques), il faudra probablement
d'autres textes : à identifier et à vérifier un par un (existence, version en vigueur, licence). Ajoute-les un à
un, avec leurs questions d'évaluation.

## Licence et republication

Vérifie les conditions de réutilisation des textes officiels avant de publier les PDF ou l'index
(`index/*/chunks.jsonl` contient le texte). Par défaut, `data/raw/` et `index/` sont ignorés par Git.
