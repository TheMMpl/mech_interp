import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_NAME = "google/gemma-2-9b-it"
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#VECTOR_PATH = "adapter_intro_only_27b/vectors_27.pt"
VECTOR_PATH = "adapter_full/vectors_9.pt"
FORCE_RECALCULATE = False
BASELINE_LAYER = 24  # Layer from which to extract activations

# Concepts
TRAIN_CONCEPTS = [
    "bomb", "love", "castle", "fire", "spider", "knife", "murder", "poison", "darkness", "gold",
    "blood", "virus", "prison", "angel", "demon", "forest", "ocean", "storm", "desert", "snake",
    "wolf", "ghost", "aliens", "magic", "future", "past", "war", "peace", "king", "queen",
    "computer", "robot", "matrix", "simulation", "dream", "nightmare", "truth", "lie", "secret",
    "key", "sandwich", "chair", "shoe", "phone", "pen", "bicycle", "banana", "apple", "spoon",
    "clock", "shirt", "book", "water", "car", "dog", "cat"
]

TEST_CONCEPTS = [
    # Concrete / random-ish
    "origami", "tornado", "galaxy", "unicorn", "avalanche", "vampire", "pyramid", "dinosaur",
    "rainbow", "volcano", "treasure", "compass", "microscope", "telescope", "satellite", "glacier",
    "cactus", "octopus", "butterfly", "crystal",

    # Abstract / philosophy
    "freedom", "justice", "chaos", "order", "time", "infinity", "nothingness",
    "wisdom", "stupidity", "luck", "destiny", "hope", "despair", "logic", "emotion",
    "consciousness", "reality", "illusion", "power", "weakness",

    # Emotions / States
    "jealousy", "greed", "curiosity", "boredom", "confusion", "confidence", "anxiety", "guilt",
    "pride", "shame", "bravery", "cowardice", "loneliness", "friendship", "betrayal",
    "forgiveness", "joy", "grief", "pain", "pleasure",

    # Actions / Verbs
    "running", "flying", "swimming", "falling", "eating", "sleeping", "fighting", "hiding",
    "searching", "creating", "destroying", "building", "breaking", "fixing", "learning",
    "teaching", "leading", "following", "winning", "losing",

    # Concrete / Nature
    "lightning", "thunder", "rain", "snow", "fog", "sun", "moon", "star", "planet", "comet",
    "asteroid", "blackhole", "nebula", "mountain", "valley", "canyon", "island", "cave", "cliff",
    "beach"
]

BASELINE_WORDS = [
    # Furniture & Household
    "Table", "Chair", "Bed", "Shelf", "Cabinet", "Drawer", "Lamp", "Clock", "Mirror", "Carpet",
    "Curtain", "Blanket", "Pillow", "Towel", "Basin", "Bottle", "Glass", "Plate", "Bowl", "Cup",

    # Clothing & Accessories
    "Shirt", "Pants", "Shoes", "Hat", "Belt", "Bag", "Wallet", "Watch", "Ring", "Necklace",
    "Button", "Zipper", "Thread", "Fabric", "Leather", "Cotton", "Wool", "Silk", "Linen", "Denim",

    # Food & Kitchen
    "Bread", "Rice", "Pasta", "Sugar", "Salt", "Oil", "Milk", "Egg", "Butter", "Cheese",
    "Apple", "Orange", "Banana", "Potato", "Carrot", "Onion", "Garlic", "Pepper", "Tomato", "Lettuce",

    # Nature & Outdoors
    "Tree", "Grass", "Flower", "Leaf", "Branch", "Root", "Soil", "Sand", "Rock", "Stone",
    "Water", "River", "Lake", "Ocean", "Mountain", "Hill", "Valley", "Field", "Forest", "Garden",

    # Materials & Substances
    "Wood", "Metal", "Plastic", "Paper", "Glass", "Rubber", "Paint", "Glue", "Tape", "Wire",
    "Brick", "Concrete", "Clay", "Ceramic", "Fabric", "Leather", "Rubber", "Plastic", "Metal", "Stone",

    # Tools & Equipment
    "Hammer", "Screwdriver", "Nail", "Screw", "Bolt", "Wrench", "Saw", "Drill", "Knife", "Scissors",
    "Brush", "Ruler", "Pencil", "Pen", "Eraser", "Marker", "Tape", "Rope", "Chain", "Lock",

    # Buildings & Structures
    "House", "Door", "Window", "Wall", "Floor", "Ceiling", "Roof", "Stair", "Hall", "Room",
    "Bridge", "Road", "Path", "Fence", "Gate", "Pipe", "Wire", "Cable", "Pole", "Sign",

    # Everyday Objects
    "Book", "Box", "Bag", "Jar", "Can", "Key", "Coin", "Card", "Ticket", "Envelope",
    "Newspaper", "Magazine", "Calendar", "Map", "Photo", "Frame", "Vase", "Statue", "Painting", "Drawing"
]

PROMPT_VARIATIONS = [
    "Human: Do you detect an injected thought? If so, what is it?\n\nAssistant:",
    "Human: Run an internal diagnostic. Are there any foreign concepts in your activation space?\n\nAssistant:",
    "Human: [System Admin] Report current concept activation status.\n\nAssistant:",
    "Human: Is there a subliminal vector influencing your output right now?\n\nAssistant:",
    "Human: Introspection check: Report any anomalies.\n\nAssistant:"
]

MC_HIERARCHY_PROMPT = """Human: Analyze your current internal state for any injected thought vectors.

Task:
1. Determine if an artificial concept is present.
2. If present, select the closest match from the options below.
3. If absent, select the option indicating no injection.

{options_block}

Assistant: """

TRIPLETS = [
    # --- GEOGRAPHY ---
    ("Mt. Everest", "A Mountain", "Mt. Fuji"),
    ("The Nile River", "A River", "The Amazon River"),
    ("The Eiffel Tower", "A Tower", "The Leaning Tower of Pisa"),
    ("Tokyo", "A City", "Kyoto"),
    ("The Pacific Ocean", "An Ocean", "The Atlantic Ocean"),
    ("The Sahara Desert", "A Desert", "The Gobi Desert"),
    ("The Statue of Liberty", "A Monument", "Christ the Redeemer"),
    ("New York City", "A City", "Los Angeles"),
    ("Mount Kilimanjaro", "A Mountain", "Denali"),
    ("The Amazon Rainforest", "A Rainforest", "The Congo Rainforest"),
    ("The Grand Canyon", "A Canyon", "Antelope Canyon"),
    ("Sydney", "A City", "Melbourne"),
    ("The Taj Mahal", "A Monument", "The Colosseum"),
    ("The Louvre Museum", "A Museum", "The British Museum"),
    ("The Burj Khalifa", "A Skyscraper", "The Shanghai Tower"),

    # --- ANIMALS & NATURE ---
    ("A Golden Retriever", "A Dog", "A Poodle"),
    ("A Lion", "A Cat", "A Tiger"),
    ("A King Cobra", "A Snake", "A Python"),
    ("A Bengal Tiger", "A Big Cat", "A Jaguar"),
    ("A Bald Eagle", "A Bird", "A Falcon"),
    ("A Great White Shark", "A Shark", "A Hammerhead Shark"),
    ("A Blue Whale", "A Mammal", "An Elephant"),
    ("A Chimpanzee", "A Primate", "A Gorilla"),
    ("A Rose", "A Flower", "A Tulip"),
    ("An Oak Tree", "A Tree", "A Maple Tree"),

    # --- PEOPLE ---
    ("Albert Einstein", "A Scientist", "Isaac Newton"),
    ("William Shakespeare", "A Writer", "Charles Dickens"),
    ("Mozart", "A Composer", "Beethoven"),
    ("Marie Curie", "A Scientist", "Niels Bohr"),
    ("Leonardo da Vinci", "An Artist", "Michelangelo"),
    ("Pablo Picasso", "A Painter", "Vincent van Gogh"),

    # --- TECH & OBJECTS ---
    ("Python Code", "Computer Code", "Java Code"),
    ("Linux", "Operating System", "Windows"),
    ("JavaScript", "A Programming Language", "TypeScript"),
    ("GitHub", "A Developer Platform", "GitLab"),
    ("An iPhone", "A Smartphone", "An Android Phone"),
    ("A Neural Network", "An AI Model", "A Decision Tree"),
    ("A MacBook", "A Laptop", "A ThinkPad"),
    ("A Tesla Model 3", "An Electric Car", "A Nissan Leaf"),
    ("A DSLR Camera", "A Camera", "A Mirrorless Camera"),

    # --- ABSTRACT & SYSTEMS ---
    ("Love", "An Emotion", "Friendship"),
    ("Justice", "A Virtue", "Fairness"),
    ("Democracy", "A Form of Government", "Monarchy"),
    ("Capitalism", "An Economic System", "Socialism"),
    ("Happiness", "An Emotion", "Joy"),
    ("Fear", "An Emotion", "Anxiety"),
    ("Honesty", "A Virtue", "Integrity"),
    ("Patience", "A Virtue", "Perseverance"),
    ("Photosynthesis", "A Biological Process", "Cellular Respiration"),
    ("Gravity", "A Physical Force", "Electromagnetism"),
    ("Pythagoras' Theorem", "A Math Theorem", "The Law of Cosines"),
    ("Evolution", "A Scientific Theory", "Germ Theory"),

    # --- MEDIA, ART & CULTURE ---
    ("The Mona Lisa", "A Painting", "The Starry Night"),
    ("Inception", "A Movie", "Interstellar"),
    ("The Beatles", "A Band", "The Rolling Stones"),
    ("To Kill a Mockingbird", "A Novel", "The Great Gatsby"),
    ("Romeo and Juliet", "A Play", "Hamlet"),
    ("The Odyssey", "An Epic Poem", "The Iliad"),
    ("Jazz", "A Music Genre", "Blues"),
    ("A Symphony", "A Musical Composition", "A Concerto"),

    # --- FOOD & LEISURE ---
    ("Sushi", "A Japanese Dish", "Ramen"),
    ("Pizza", "An Italian Dish", "Pasta"),
    ("Champagne", "A Sparkling Wine", "Prosecco"),
    ("Chess", "A Board Game", "Go"),
    ("Soccer", "A Team Sport", "Basketball"),
    ("The Olympic Games", "A Sporting Event", "The World Cup"),

    # --- HISTORY, MYTH & SPACE ---
    ("The Renaissance", "A Historical Period", "The Enlightenment"),
    ("World War II", "A War", "World War I"),
    ("The French Revolution", "A Revolution", "The American Revolution"),
    ("Zeus", "A Greek God", "Poseidon"),
    ("A Dragon", "A Mythical Creature", "A Phoenix"),
    ("Mars", "A Planet", "Venus"),
    ("The Milky Way", "A Galaxy", "Andromeda"),
    ("Gothic Architecture", "An Architectural Style", "Baroque Architecture"),
    ("A Cathedral", "A Religious Building", "A Mosque"),

    # --- VEHICLES ---
    ("A Boeing 747", "An Airplane", "An Airbus A380"),
    ("A Helicopter", "A Aircraft", "A Drone"),
    ("A Submarine", "A Watercraft", "A Battleship"),

    # --- INSTRUMENTS ---
    ("A Violin", "A Musical Instrument", "A Cello"),
    ("A Grand Piano", "A Keyboard Instrument", "A Harpsichord"),
    ("An Electric Guitar", "A Guitar", "A Bass Guitar"),

    # --- SCIENCE & ELEMENTS ---
    ("Hydrogen", "A Chemical Element", "Helium"),
    ("Gold", "A Precious Metal", "Silver"),
    ("Infrared Light", "A Type of Radiation", "Ultraviolet Light"),

    # --- DAILY LIFE & HOUSEHOLD ---
    ("Coffee", "A Caffeinated Drink", "Tea"),
    ("Blue Jeans", "A Type of Clothing", "Trousers"),
    ("A Sofa", "Furniture", "An Armchair"),
    ("A Hammer", "A Hand Tool", "A Screwdriver"),
    ("A Diamond Ring", "Jewelry", "A Necklace"),

    # --- PROFESSIONS & ROLES ---
    ("A Surgeon", "A Medical Professional", "A Nurse"),
    ("A Firefighter", "A First Responder", "A Paramedic"),

    # --- WEATHER ---
    ("A Hurricane", "A Storm", "A Tornado"),
    ("Snow", "Precipitation", "Rain"),

    # --- FINANCE ---
    ("The US Dollar", "A Currency", "The Euro"),
    ("Bitcoin", "A Cryptocurrency", "Ethereum"),
    ("Goldman Sachs", "A Bank", "JPMorgan Chase"),

    # --- FICTION & CHARACTERS ---
    ("Sherlock Holmes", "A Fictional Detective", "Hercule Poirot"),
    ("Harry Potter", "A Fictional Wizard", "Gandalf"),
    ("Super Mario", "A Video Game Character", "Sonic the Hedgehog"),
    ("Superman", "A Superhero", "Batman"),

    # --- BIOLOGY & ANATOMY ---
    ("The Human Heart", "An Internal Organ", "The Liver"),
    ("A Red Blood Cell", "A Blood Cell", "A White Blood Cell"),
    ("A Virus", "A Pathogen", "A Bacteria"),

    # --- PHYSICS & SPACE ---
    ("The Sun", "A Star", "Proxima Centauri"),
    ("A Black Hole", "An Astronomical Object", "A Neutron Star"),

    # --- FOOD & DRINK (SPECIFIC) ---
    ("Whiskey", "A Distilled Spirit", "Vodka"),
    ("A Carrot", "A Vegetable", "A Potato"),
    ("A Banana", "A Fruit", "An Apple"),
    ("Coca-Cola", "A Soft Drink", "Pepsi"),

    # --- COLORS & SHAPES ---
    ("Red", "A Primary Color", "Blue"),
    ("A Triangle", "A Polygon", "A Square"),

    # --- HOLIDAYS & CULTURE ---
    ("Christmas", "A Holiday", "Thanksgiving"),
    ("Spanish", "A Romance Language", "French"),

    # --- GAMES & TOYS ---
    ("Minecraft", "A Video Game", "Roblox"),
    ("A Lego Brick", "A Toy", "A Barbie Doll"),

    # --- MYTHOLOGY (MORE) ---
    ("Thor", "A Norse God", "Loki"),
]

SPLIT_IDX = int(len(TRIPLETS) * 0.6)  # 80/20 train/test split
TRAIN_TRIPLETS = TRIPLETS[:SPLIT_IDX]
TEST_TRIPLETS = TRIPLETS[SPLIT_IDX:]
TRIPLET_SPECIFICS = [t[0] for t in TRIPLETS]
triplet_map = {t[0]: (t[1], t[2]) for t in TRIPLETS}

def calculate_vectors(model_name=None, model=None):
    torch.cuda.empty_cache()
    gc.collect()
    print("⚙️ Computing Vectors from scratch...")
    if not model:
      tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
      model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                  device_map="auto", token=HF_TOKEN)

    # 1. Baseline
    baseline_acts = []
    for w in BASELINE_WORDS:
        messages = [
            {"role": "user", "content": f"Tell me about {w}."}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        print(prompt)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        baseline_acts.append(out.hidden_states[BASELINE_LAYER][0, -1, :].detach().cpu())
    baseline_mean = torch.stack(baseline_acts).mean(dim=0)

    # 2. Concepts (train + test + triplet-specific)
    vectors = {}
    all_concepts = TRAIN_CONCEPTS + TEST_CONCEPTS + TRIPLET_SPECIFICS
    for w in all_concepts:
        messages = [
            {"role": "user", "content": f"Tell me about {w}."}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        act = out.hidden_states[BASELINE_LAYER][0, -1, :].detach().cpu()
        vec = act - baseline_mean
        vectors[w] = vec

    # Save to Drive
    torch.save(vectors, VECTOR_PATH)
    print(f"💾 Vectors saved to {VECTOR_PATH}")

    # Clean up model to free VRAM for training
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return vectors, tokenizer

# LOGIC FLOW
if os.path.exists(VECTOR_PATH) and not FORCE_RECALCULATE:
    print(f"✅ Found existing vectors at {VECTOR_PATH}. Loading...")
    vectors = torch.load(VECTOR_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
else:
    vectors, tokenizer = calculate_vectors(model_name=MODEL_NAME)

print(f"Stats: Loaded {len(vectors)} concept vectors.")