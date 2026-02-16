from __future__ import annotations

import re
import html2text


def description_to_text(description: str | None) -> str:
    """
    Velog RSS <description> (feedparser: entry.summary)를
    RAG에 넣기 좋은 텍스트로 정제한다.
    """
    desc = description or ""

    h = html2text.HTML2Text()
    h.ignore_links = True    
    h.ignore_images = True
    h.body_width = 0          
    h.single_line_break = True

    text = h.handle(desc)
    text = text.replace("\r\n", "\n")

    # 과도한 공백/빈줄 정리
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
