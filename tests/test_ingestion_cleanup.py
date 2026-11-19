"""Tests for thread and signature cleanup."""

from support_engine.ingestion.cleanup import (
    CleanupConfig,
    clean_text,
    html_to_text,
    normalize_text,
    strip_quoted_replies,
    strip_signature,
)


def test_normalize_whitespace_unicode_and_controls() -> None:
    raw = "  Hello\u00a0world\u200b \r\n\r\n\r\n\tsecond   line\x07  "
    assert normalize_text(raw) == "Hello world\n\nsecond line"


def test_normalize_unicode_composes_accents() -> None:
    assert normalize_text("Joa\u0303o") == "Jo\u00e3o"


def test_html_to_text_keeps_paragraphs_and_entities() -> None:
    raw = "<p>First &amp; best</p><p>Second<br>line</p><script>alert(1)</script>"
    text = normalize_text(html_to_text(raw))
    assert text == "First & best\nSecond\nline"


def test_strip_quoted_reply_english_and_portuguese() -> None:
    en = "Restarting fixed it.\n\nOn Mon, 3 Mar 2026 at 10:00, Support <help@example.com> wrote:\n> Try restarting"
    pt = "Funcionou.\nEm seg., 3 de mar. de 2026, Suporte escreveu:\n> Tente reiniciar"
    assert strip_quoted_replies(en).strip() == "Restarting fixed it."
    assert strip_quoted_replies(pt).strip() == "Funcionou."


def test_strip_outlook_header_block() -> None:
    raw = "Still failing.\n\nFrom: Help Desk\nSent: Monday\nTo: someone\nSubject: Re: login\n\nold text"
    assert strip_quoted_replies(raw).strip() == "Still failing."


def test_strip_inline_quote_lines() -> None:
    assert strip_quoted_replies("> old\nnew answer\n>> older").strip() == "new answer"


def test_strip_signature_delimiter_and_signoff() -> None:
    assert strip_signature("The export works now.\n--\nAna\nACME Corp") == "The export works now."
    assert strip_signature("The export works now.\n\nBest regards,\nAna Lima\nSupport Engineer").strip() == (
        "The export works now."
    )


def test_strip_signature_keeps_long_tail() -> None:
    body = "Thanks\n" + "\n".join(f"step {i} with real content" for i in range(10))
    assert strip_signature(body, max_signature_lines=3) == body


def test_clean_text_full_email() -> None:
    raw = (
        "<div>Hi,</div><div>The report still shows zero rows.</div>"
        "<div>Thanks,</div><div>Bruno</div>"
        "<div>Sent from my phone</div>"
        "<div>CONFIDENTIALITY NOTICE: this message is private.</div>"
    )
    assert clean_text(raw) == "Hi,\nThe report still shows zero rows."


def test_clean_text_steps_can_be_disabled() -> None:
    raw = "Answer\n> quoted\n--\nSig"
    cfg = CleanupConfig(strip_quoted_replies=False, strip_signatures=False)
    assert clean_text(raw, cfg) == "Answer\n> quoted\n--\nSig"


def test_clean_text_empty() -> None:
    assert clean_text("") == ""
    assert clean_text("On Mon, Support wrote:\n> only quote") == ""
