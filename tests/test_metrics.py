from llm_evaluate.metrics.text_metrics import char_f1, exact_match, rouge_l, token_f1


def test_exact_match_normalization() -> None:
    assert exact_match("Paris!", "paris") == 1.0


def test_token_f1() -> None:
    score = token_f1("the capital is tokyo", "tokyo")
    assert 0 < score < 1


def test_rouge_l() -> None:
    score = rouge_l("the capital is tokyo", "tokyo is capital")
    assert 0 < score <= 1


def test_char_f1() -> None:
    score = char_f1("北京", "北京")
    assert score == 1.0
