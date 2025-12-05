from vuln_hunter.inference.predictor import Predictor


def test_predictor_runs():
    predictor = Predictor()
    result = predictor.predict("print('hello world')")
    assert 0.0 <= result["probability"] <= 1.0
    assert isinstance(result["vulnerable"], bool)
    assert "explanation" in result
