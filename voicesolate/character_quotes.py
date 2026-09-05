"""
voicesolate/character_quotes.py

Expressive, phonetically rich training prose and quotes for teacher-student
corpus expansion and distillation.
"""

from typing import List

MARK_TWAIN_CORPUS: List[str] = [
    "Madam, I'd be delighted. So, this is a space ship? You ever run into Halley's comet?",
    "The secret of getting ahead is getting started, and don't you ever forget it!",
    "Kindness is the language which the deaf can hear and the blind can see.",
    "Whenever you find yourself on the side of the majority, it is time to pause and reflect.",
    "Twenty years from now you will be more disappointed by the things you didn't do than by the ones you did do.",
    "I have long been interested in the notion of time travelers. In fact, I wrote a book about it.",
    "If you tell the truth, you don't have to remember anything!",
    "The man who does not read has no advantage over the man who cannot read.",
    "Never tell the truth to people who are not worthy of it.",
    "Age is an issue of mind over matter. If you don't mind, it doesn't matter!",
    "A lie can travel half way around the world while the truth is putting on its shoes.",
    "Get your facts first, then you can distort them as much as you please.",
    "Thunder is good, thunder is impressive; but it is lightning that does the work!",
    "Good decisions come from experience, and experience comes from making bad decisions.",
    "Humor is mankind's greatest blessing, sir, make no mistake about it.",
    "Courage is resistance to fear, mastery of fear, not absence of fear!",
    "Wrinkles should merely indicate where smiles have been.",
    "The right word may be effective, but no word was ever as effective as a rightly timed pause.",
    "When in doubt, tell the truth. It will confound your enemies and astound your friends!",
    "Giving up smoking is the easiest thing in the world. I know because I've done it thousands of times.",
    "San Francisco is a truly magnificent city, if you don't mind the chill getting into your bones.",
    "Mississippi riverboats are the grandest palaces afloat upon the waters of the earth!",
    "We have the finest congress that money can buy, I do declare.",
    "I didn't attend the funeral, but I sent a very nice letter saying I approved of it.",
    "The report of my death was an exaggeration, a gross exaggeration!",
    "Lord bless my soul, look at the astonishing wonder of this mechanical age!",
    "Now hold on just a doggone minute, mister, before you jump to hasty conclusions.",
    "A classic is something that everybody wants to have read and nobody wants to read.",
    "Civilization is the limitless multiplication of unnecessary necessities.",
    "Suppose you were an idiot, and suppose you were a member of Congress; but I repeat myself.",
    "I am an old man and have known a great many troubles, but most of them never happened.",
    "Buy land, son, they're not making it anymore!",
    "Truth is stranger than fiction, because Fiction is obliged to stick to possibilities; Truth isn't.",
    "Always do what is right. It will gratify half of mankind and astound the other.",
    "What would men be without women? Scarce, sir... mighty scarce.",
    "There are three kinds of lies: lies, damned lies, and statistics.",
    "If it's your job to eat a frog, it's best to do it first thing in the morning.",
    "An honest man in politics shines like a solitary candle in the blackest night.",
    "When I was a boy of fourteen, my father was so ignorant I could hardly stand to have the old man around.",
    "You cannot depend on your eyes when your imagination is out of focus.",
]

GENERIC_EXPRESSIVE_CORPUS: List[str] = [
    "Look around you and observe how quickly the circumstances of life can change.",
    "Do you truly believe that what we are witnessing here is merely a coincidence?",
    "Stand firm, speak clearly, and never compromise on what is essential.",
    "There is an extraordinary power in quiet determination and unwavering resolve.",
    "Listen carefully to what is being said, but pay closer attention to what is left unsaid.",
    "The stars in the night sky have witnessed the rise and fall of countless civilizations.",
    "Every journey begins with uncertainty, yet progress demands that we take the initial step.",
    "I have traveled across vast distances to discover that the greatest mysteries lie within.",
    "Remarkable achievements rarely arise from comfortable and complacent conditions.",
    "Tell me precisely what occurred, and do not omit any detail, however insignificant it seems.",
]

def get_expansion_corpus_for_character(character_name: str) -> List[str]:
    """Returns a character-appropriate corpus for F5-TTS dataset distillation."""
    clean = (character_name or "").strip().upper()
    if any(k in clean for k in ["CLEMENS", "TWAIN", "SAMUEL"]):
        return MARK_TWAIN_CORPUS
    return GENERIC_EXPRESSIVE_CORPUS
