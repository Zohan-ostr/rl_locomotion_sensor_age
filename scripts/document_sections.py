"""Update one generated section while retaining a single Markdown document."""
import re


def write_section(path, section, title, body):
    marker = f'<a id="{section}"></a>'
    replacement = marker + f'\n\n## {title}\n\n' + body.rstrip() + '\n'
    text = path.read_text() if path.exists() else '# Результаты эксперимента\n'
    # Consolidated sections are separated by horizontal rules and explicit anchors.
    pattern = re.escape(marker) + r'.*?(?=\n\n---\n\n<a id=|\Z)'
    if marker in text:
        text = re.sub(pattern, lambda _: replacement, text, count=1, flags=re.S)
    else:
        text = text.rstrip() + '\n\n---\n\n' + replacement
    path.write_text(text)
