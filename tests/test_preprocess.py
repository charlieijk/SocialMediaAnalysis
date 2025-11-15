from social_media_analysis import preprocess


def test_normalize_text_strips_noise():
    noisy = "@User OMG!!! Check https://example.com #CoolStuff 😄"
    normalized = preprocess.normalize_text(noisy)
    assert "@" not in normalized
    assert "http" not in normalized
    assert "coolstuff" in normalized
    assert normalized.islower()


def test_generate_ngrams_respects_range():
    tokens = preprocess.tokenize("Love this API so much")
    ngrams = preprocess.generate_ngrams(tokens, (1, 3))
    assert "love" in ngrams
    assert "love_this" in ngrams
    assert "love_this_api" in ngrams
