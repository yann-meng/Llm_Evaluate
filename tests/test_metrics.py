from llm_evaluate.metrics.text_metrics import exact_match, token_f1


def test_exact_match_normalization() -> None:
    assert exact_match("Paris!", "paris") == 1.0


def test_token_f1() -> None:
    score = token_f1("the capital is tokyo", "tokyo")
    assert 0 < score < 1
