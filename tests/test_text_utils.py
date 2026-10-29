from text_utils import clean_text


def test_clean_text_normalizes_html_urls_and_whitespace():
    result = clean_text("<br /> GREAT film!  Visit https://example.com")
    assert result == "great film! visit"


def test_clean_text_removes_unsupported_characters():
    assert clean_text("hello \u2603 world") == "hello world"
