# ─── TOK-COUNT SAFETY NET ──────────────────────────────────────────────── #
import tiktoken

ENC               = tiktoken.encoding_for_model("o3-mini")
MAX_ARTICLE_TOKENS = 3_000                # pick any hard cap you like

def clip_or_split(text: str) -> list[str]:
    """
    Return a list of ≤MAX_ARTICLE_TOKENS-token chunks.
    · If text is short, you get [text].
    · If it’s long, we emit sequential windows so nothing exceeds the cap.
    """
    if not text:
        return []

    tokens = ENC.encode(text)
    if len(tokens) <= MAX_ARTICLE_TOKENS:
        return [text]

    chunks, start = [], 0
    while start < len(tokens):
        end = start + MAX_ARTICLE_TOKENS
        chunk_text = ENC.decode(tokens[start:end])
        chunks.append(chunk_text)
        start = end
    return chunks