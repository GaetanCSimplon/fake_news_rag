# Explication de code

```python
    # -----------------------------
    # Vectorisation avec parallélisation
    # -----------------------------
    def embed_texts(self, texts: List[str], max_workers: int = 4) -> List[List[float]]:
        """Crée des embeddings normalisés pour une liste de textes (en parallèle)."""

        def embed_one(text):
            """
            Appelle l'embedder et retourne le vecteur normalisé
            """
            response = ollama.embeddings(model=self.model_name, prompt=text)
            return self.normalize_vector(response.embedding)

        embeddings = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor: 
            futures = {executor.submit(embed_one, text): text for text in texts}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Vectorisation parallèle"): 
                embeddings.append(f.result()) 

        return embeddings
```

## Contexte

Cette étape fait partie de la phase de vectorisation des chunks préalablement crées. L'embedding peut prendre beaucoup de temps en fonction du volume de données et du découpage en chunk. Les ressources matérielles et le temps pour faire cette pouvant être limitées, il s'agit d'optimiser la manière de produire les embeddings.

Pour ce faire, la vectorisation parallélisée avec ThreadPoolExecutor peut être employée.

## Logique

1. Plutôt que de traiter l'embedding de manière séquentielle (un texte après l'autre), le ThreadPoolExecutor permet de lancer plusieurs embedding en parallèle, simultanément.

2. Chaque thread appelle le modèle Ollama pour générer l'embedding d'un texte et fait appel à notre méthode de normalisation des vecteurs.

3. Les résultats sont récupérés au fur et à mesure de leur disponibilité et accumulés dans une liste

4. La librairie tqdm permet de créer des barres de progression afin de voir l'avancement du processus.

## Analyse du code

### Vue d'ensemble

La fonction ```embed_texts()``` prend une liste de textes et retourne une liste de vecteurs (embeddings).

- **max_workers**: Il s'agit des threads qui vont traiter les textes en simultanée. Par défaut, 4 textes peuvent être traités en même temps. 

### Fonction interne embed_one()

```python
def embed_one(text):
    response = ollama.embeddings(model=self.model_name, prompt=text)
    return self.normalize_vector(response.embedding)
```

- Traite un unique texte

- Fait appel au modèle d'embedding de Ollama (all-minilm)

### Création du pool de threads (des instances parallèles)

```python
with ThreadPoolExecutor(max_workers=max_workers) as executor:
```
- Créé un gestionnaire qui exécute 4 tâches en parallèle. Les threads partagent les ressources du processeur pour traiter plusieurs textes en simultanée.

### Soumission des tâches

```python
futures = {executor.submit(embed_one, text): text for text in texts}
```
- Pour chaque texte, ```executor.submit()``` lance ```embed_one()``` dans un thread parallèle et retourne un objet ```Future``` ("promesse" du résultat futur). Le dictionnaire associe chaque tâche au texte correspondant

- Cette ligne crée un dictionnaire en compréhension (``` {clé: valeur for élément in itérable}```)

- Clé: ``` executor.submit(embed_one, text)``` -> retourne un objet ```Future``` représentant la tâche en cours d'exécution

- Valeur: ```text``` -> le texte original correspondant à cette tâche

- Itération: ```for text in texts``` -> parcourt chaque texte de la liste

**Résultat**: Un dictionnaire où chaque entrée associe une tâche en attente (Future) au texte qui lui correspond.

**Exemple concret**:

```python
texts = ["bonjour", "au revoir"]

# Le dictionnaire ressemble à :
{
    <Future object 1>: "bonjour",
    <Future object 2>: "au revoir"
}
```

### Récupération des résultats

```python
for f in tqdm(as_completed(futures), total=len(futures), desc="Vectorisation parallèle"):
    embeddings.append(f.result())
```

- ```as_completed()```itère sur les futures au fur et à mesure qu'elles se terminent (pas dans l'ordre de soumission)
- ```tqdm()``` génère la barre de progression
- ```f.result()``` récupère le vecteur normalisé une fois que la tâche est terminée

## Résultats

Sans la gestion en parallèle de l'embedding, le processus séquentiel prenait environ 1h à 1h30. Avec l'utilisation de la vectorisation en parallèle, on divise ce temps par autant de threads créé, en l'occurrence, par 4 dans notre cas. Dans les faits, le processus s'est fait en environ 40~45 min.

## Conclusion

L'utilisation de ThreadPoolExecutor intervient dans un contexte où il y a un grand nombre de données à traiter. L'emploi de cet outil est particulièrement utile pour optimiser des processus long et séquentiels. 

