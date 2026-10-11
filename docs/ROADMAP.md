# Prochaines étapes

La V1 est fonctionnelle et ses tests passent, mais l'évaluation de qualité repose actuellement sur trois questions.
Les priorités suivantes visent à mesurer les réponses sur un échantillon plus représentatif et à réduire les risques
liés aux réponses juridiques.

## Priorités

1. Élargir le jeu d'évaluation à 50-60 questions : questions répondables, questions hors corpus et situations où des
	informations personnelles manquent. Faire vérifier les références attendues indépendamment du système.
2. Analyser l'abstention erronée observée sur la question des congés. Le système devrait fournir la règle générale
	quand elle est connue et demander les informations manquantes avant tout calcul personnel.
3. Annoter manuellement la fidélité et l'exactitude des réponses. Compléter la précision des citations par une mesure
	de couverture des articles attendus.
4. Identifier et vérifier la version juridique du PDF à partir d'une source officielle, puis renseigner `as_of` dans
	`data/sources.json` avant de présenter le corpus comme actuel.
5. Comparer les changements de retrieval et de génération sur le même jeu d'évaluation, et documenter les résultats
	et les erreurs sans généraliser au-delà de l'échantillon.

Le déploiement public et l'ajout d'autres textes juridiques restent hors périmètre tant que ces vérifications ne sont
pas terminées.
