"""
Offline tests for the backend answer sanitizer and the output-side rules of the
fix pack F4 (no HTML, no tables, aliases never rendered).
"""

import config
from core.sanitizer import contains_html, sanitize_answer


def test_line_break_variants_become_newlines():
    for variant in ("<br>", "<br/>", "<br />", "<BR>"):
        assert sanitize_answer(f"baris satu {variant}baris dua") == (
            "baris satu\nbaris dua"
        )


def test_other_html_tags_are_removed_but_content_is_kept():
    assert sanitize_answer("<p>Kendala <b>utama</b> adalah logistik.</p>") == (
        "Kendala utama adalah logistik."
    )


def test_markdown_survives_untouched():
    markdown = (
        "**Kendala**\n\n- logistik\n- waktu\n\n> kutipan penting\n\n"
        "`kode` dan [tautan](https://example.com)"
    )

    assert sanitize_answer(markdown) == markdown


def test_repeated_blank_lines_are_collapsed():
    assert sanitize_answer("a<br><br><br><br>b") == "a\n\nb"


def test_trailing_spaces_before_a_newline_are_removed():
    assert sanitize_answer("baris satu   <br>baris dua") == "baris satu\nbaris dua"


def test_no_html_remaining_after_sanitizing():
    assert contains_html(sanitize_answer("<div><span>x</span></div>")) is False


def test_the_answer_addendum_forbids_tables_and_html():
    from core import prompts

    addendum = prompts.CITATION_FORMAT_ADDENDUM
    assert "Jangan menggunakan tabel" in addendum
    assert "HTML" in addendum
    assert str(config.ANSWER_TARGET_WORDS) in addendum
    assert "wajib disertai sitasi" in addendum
