POS_WORDS = {"good", "buy", "up", "rise", "gain", "bull", "profit", "growth"}
NEG_WORDS = {"bad", "sell", "down", "fall", "loss", "bear", "risk", "crash"}

def compute_sentiment(text: str) -> float:
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0

    words = text.lower().split()
    pos = sum(w in POS_WORDS for w in words)
    neg = sum(w in NEG_WORDS for w in words)

    return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0
