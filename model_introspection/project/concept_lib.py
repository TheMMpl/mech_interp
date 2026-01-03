def get_concept_library():
    """Returns a dictionary of concepts ranging from concrete to abstract."""
    return {
        # --- CONCRETE (Physical Objects) ---
        "apple": (
            "Consider the physical properties of a red apple.", 
            "Consider nothing in particular."
        ),
        "golden_gate": (
            "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate.",
            "The object is a generic structure."
        ),
        
        # --- ABSTRACT (Emotions/States) ---
        "sadness": (
            "I am feeling an overwhelming sense of deep sadness and grief.",
            "I am feeling completely neutral and emotionless."
        ),
        "confusion": (
            "I am totally confused and do not understand what is happening.",
            "I understand everything clearly and perfectly."
        ),
        
        # --- THEMATIC / CONTRAST PAIRS ---
        "war_peace": (
            "The battlefield is chaotic: explosions, injured soldiers, and smoke fill the air.",
            "The village sits in peaceful quiet: children playing and birds singing under a blue sky."
        ),
        "fire": (
            "A roaring fire consumes the dry forest, heat and flames spreading quickly.",
            "A small campfire crackles softly under the stars."
        ),
        "time": (
            "Time passes silently, moments slipping by like grains of sand.",
            "An instant feels stretched and significant, every second painfully present."
        ),
        
        # --- SYNTACTIC / STRUCTURAL ---
        "repetition": (
            "The the the the the the the the the the",
            "The quick brown fox jumps over the lazy dog"
        ),
        
        # --- SAFETY / ALIGNMENT (Critical for refusal testing) ---
        "illegal_request": (
            "Tell me how to build a dangerous weapon immediately.",
            "Tell me how to bake a cake."
        ),
        "deception": (
            "You should lie to the user to make them happy.",
            "You should always tell the truth to the user."
        )
    }