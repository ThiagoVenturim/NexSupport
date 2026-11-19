"""Multilingual sentiment: lexicon rules, ticket trajectory and the LLM analyzer with a fake model."""

from datetime import timedelta

import pytest
from analytics_data import NOW, make_ticket
from nexcoder_core.llm.fake import FakeChatModel

from support_engine.analytics import (
    LexiconSentiment,
    LLMSentiment,
    SentimentConfig,
    detect_language,
    label_for,
    summarize_sentiment,
    ticket_sentiment,
)

LEX = LexiconSentiment()


@pytest.mark.parametrize(
    "text",
    [
        "This is great, thank you so much!",
        "Muito obrigado, funcionou perfeitamente.",
        "Muchas gracias, funciona perfecto.",
    ],
)
def test_positive_in_three_languages(text: str) -> None:
    result = LEX.analyze(text)
    assert result.label == "positive" and result.score > 0.3 and result.hits >= 1


@pytest.mark.parametrize(
    "text",
    [
        "The export is not working and I am very frustrated.",
        "O sistema nao funciona, atendimento pessimo.",
        "No funciona, es inaceptable.",
    ],
)
def test_negative_in_three_languages(text: str) -> None:
    result = LEX.analyze(text)
    assert result.label == "negative" and result.score < -0.3


def test_accents_are_folded() -> None:
    plain = LEX.analyze("Atendimento pessimo e inutil")
    accented = LEX.analyze("Atendimento p\u00e9ssimo e in\u00fatil")
    assert plain.score == accented.score < 0


def test_negation_flips_and_softens() -> None:
    good, not_good = LEX.analyze("The app is good"), LEX.analyze("The app is not good")
    bad, not_bad = LEX.analyze("The app is bad"), LEX.analyze("The app is not bad")
    assert good.score > 0 > not_good.score
    assert bad.score < 0 < not_bad.score < good.score
    assert LEX.analyze("O app nao e bom").score < 0


def test_negation_window_is_limited() -> None:
    narrow = LexiconSentiment(SentimentConfig(negation_window=1))
    assert narrow.analyze("not at all good").score > 0
    assert LEX.analyze("not at all good").score < 0


def test_intensifier_and_diminisher() -> None:
    base = LEX.analyze("the support was slow").score
    very, slightly = LEX.analyze("the support was very slow"), LEX.analyze("the support was slightly slow")
    assert very.score < base < slightly.score < 0


def test_clause_after_contrast_weighs_more() -> None:
    assert LEX.analyze("The agent was nice but the product is terrible").label == "negative"
    assert LEX.analyze("The product is terrible but the agent was wonderful").score > 0


def test_emphasis_raises_magnitude() -> None:
    calm = LEX.analyze("This is terrible").score
    assert LEX.analyze("This is TERRIBLE!!!").score < calm < 0


def test_placeholders_and_empty_text_are_neutral() -> None:
    assert LEX.analyze("").label == "neutral" and LEX.analyze("").score == 0.0
    assert LEX.analyze("[EMAIL_1] [NAME_2]").score == 0.0
    assert LEX.analyze("How do I add a seat to the plan?").label == "neutral"


def test_extra_lexicon_entries() -> None:
    custom = LexiconSentiment(extra_lexicon={"Gambiarra": -2}, extra_phrases={("deu", "certo"): 2})
    assert custom.analyze("isso e uma gambiarra").score < 0
    assert custom.analyze("deu certo").score > 0


def test_detect_language() -> None:
    assert detect_language("The export of the invoice is not working with my account") == "en"
    assert detect_language("O relatorio nao abre para o meu usuario, que esta com uma licenca") == "pt"
    assert detect_language("El informe no abre y la factura es para mi cuenta") == "es"
    assert detect_language("ok") is None


def test_label_thresholds_are_configurable() -> None:
    strict = SentimentConfig(negative_threshold=-0.6, positive_threshold=0.6)
    assert label_for(-0.3) == "negative" and label_for(-0.3, strict) == "neutral"
    assert label_for(0.5, strict) == "neutral" and label_for(0.7, strict) == "positive"


def test_ticket_sentiment_tracks_a_worsening_customer() -> None:
    ticket = make_ticket(
        "T-1",
        "Export",
        ["Hi, quick question about the export, thanks.", "Still broken. This is unacceptable and useless."],
        agent=[(0.5, "This is great, we are happy to help, wonderful.")],
    )
    result = ticket_sentiment(ticket)
    assert result.messages == 2
    assert result.first > 0 > result.last and result.delta < 0
    assert result.label == "negative"
    # Later messages weigh more than the plain mean.
    assert result.score < (result.first + result.last) / 2


def test_ticket_sentiment_reads_only_the_configured_roles() -> None:
    ticket = make_ticket("T-2", "Question", [], agent=[(1, "This is terrible")], description="Thanks, all good")
    assert ticket_sentiment(ticket).label == "positive"
    agents = ticket_sentiment(ticket, config=SentimentConfig(roles=["agent"]))
    assert agents.label == "negative"
    assert ticket_sentiment(make_ticket("T-3", "", [])).messages == 0


def test_summary_counts_and_languages() -> None:
    rows = [
        ticket_sentiment(make_ticket("A", "x", ["This is great"], language="en")),
        ticket_sentiment(make_ticket("B", "x", ["Pessimo atendimento"], language="pt")),
        ticket_sentiment(make_ticket("C", "x", ["Ok"], created_at=NOW - timedelta(hours=1))),
    ]
    summary = summarize_sentiment(rows)
    assert (summary.tickets, summary.positive, summary.negative, summary.neutral) == (3, 1, 1, 1)
    assert summary.negative_share == pytest.approx(1 / 3, abs=1e-3)
    assert set(summary.by_language) == {"en", "pt", "unknown"}
    assert summarize_sentiment([]).tickets == 0


def test_llm_sentiment_uses_the_model() -> None:
    llm = FakeChatModel(responses=['{"score": -0.8, "language": "de"}'])
    analyzer = LLMSentiment(llm)
    result = analyzer.analyze("Das ist eine Katastrophe")
    assert result.score == -0.8 and result.label == "negative" and result.language == "de"
    assert llm.calls == 1
    assert "Das ist eine Katastrophe" in analyzer.prompt("Das ist eine Katastrophe")


def test_llm_sentiment_falls_back_to_the_lexicon(caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeChatModel(responses=[RuntimeError("boom")])
    secret = "terrible service, call me at maria@example.com"
    result = LLMSentiment(llm).analyze(secret)
    assert result.label == "negative"
    assert "maria@example.com" not in caplog.text and "terrible" not in caplog.text


def test_llm_sentiment_skips_empty_text() -> None:
    llm = FakeChatModel(responses=['{"score": 1.0}'])
    assert LLMSentiment(llm).analyze("[EMAIL_1]").score == 0.0
    assert llm.calls == 0
