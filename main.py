from src.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    # Initialisation du pipeline complet
    pipeline = RAGPipeline(
        chroma_path="data/vector_db",
        collection_name="articles",
        embedding_model="all-minilm"
    )

    # Exemple de texte utilisateur
    user_article = """
    A wave of alarm rippled through several cities this week after dozens of patients presented at local clinics with a cluster of unusual flu-like symptoms that some commentators have linked to recent upgrades of 5G infrastructure. Social feeds filled with dramatic eyewitness videos and bold headlines claiming a new COVID strain — dubbed by some as the “Eclipse variant” — is spreading “overnight” and tied to telecommunications work.

    “We started seeing people with severe headaches, temporary vision blurring and prolonged fatigue,” said an unnamed clinic nurse who asked not to be identified. “At first we thought it was a bad flu season, but the timing with the tower upgrades made patients anxious and the rumors snowballed.”

    Public health officials in the region issued a brief statement urging calm and reminding citizens that investigations are ongoing. “We are monitoring reports of increased respiratory illness and coordinating with laboratories to determine any viral changes,” the statement read. No peer-reviewed data has confirmed the existence of a distinct variant, however.

    Despite the official caution, a flurry of social media posts claimed the new symptoms were caused by electromagnetic interference from newly activated cell towers. Influential commentators amplified the story, posting screenshots of anonymous messages alleging that technicians had been “caught” disabling safety protocols. These posts were widely shared with captions such as “When technology meets biology” and “They told us it was safe.”

    Medical experts contacted by this outlet emphasized the absence of evidence linking telecommunications infrastructure to viral mutation or immediate health effects in otherwise healthy adults. “Viruses change through biological processes; there is no scientific pathway by which radiofrequency upgrades would create a new viral strain,” said Dr. Elaine Porter, an infectious disease specialist. “Rumors like this distract from established prevention measures.”

    Meanwhile, alternative websites published interviews with a person claiming to be a whistleblower from a regional carrier, asserting the company rushed 5G installations during a recent “data surge” and failed to inform the public. The carrier denied these allegations and released a spokesperson video insisting safety checks were followed.

    Local markets mirrored the public’s unease: sales of face masks, over-the-counter herbal remedies and signal-blocking stickers spiked in some neighborhoods, while a small group of protesters gathered outside a municipal office to demand a halt to further network expansions until independent testing take place.

    Analysts say the story is a textbook example of how fast a plausible narrative can spread when it connects a feared disease with a modern, visible technology. “This is exactly how misinformation gains traction: combine a health scare, a technical subject people don’t understand, and vivid but unverified eyewitness accounts,” said media researcher Carlos Nguyen. “That mix reduces skepticism and increases shares.”

    In response, several platforms flagged related posts for review, and a widely used video hosting site added informational links directing viewers to verified public health resources. Still, digital sleuths on message boards continued to circulate purported internal documents — none of which have been authenticated.

    As laboratories work to sequence samples from the affected clinics, investigators caution against drawing conclusions. “At this stage, we do not have verified genomic evidence of a new COVID variant,” a municipal health officer said. “We urge the public to rely on official updates and avoid acting on rumors that could cause unnecessary alarm.”

    For citizens worried about their health, doctors recommend standard precautions: staying home if ill, consulting a healthcare provider, and following verified public-health guidance. Experts also urge the public to scrutinize extraordinary claims and check multiple reputable sources before sharing.
    """

    # Lancement de l’analyse
    result = pipeline.analyze_article(user_article, model_name="llama3.2")
    
    print("====== Réponse =======")
    print(result.strip())
