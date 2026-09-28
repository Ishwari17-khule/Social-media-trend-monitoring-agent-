"""Word cloud generation helpers."""

from __future__ import annotations

from io import BytesIO

from wordcloud import WordCloud

from utils.preprocess import clean_text


def build_wordcloud_image(texts: list[str]) -> BytesIO:
    """Build a PNG word cloud image from a list of article texts."""
    joined = " ".join(clean_text(text) for text in texts if text)
    if not joined.strip():
        joined = "no data"

    cloud = WordCloud(
        width=1200,
        height=520,
        background_color="#0b1220",
        colormap="viridis",
        collocations=False,
    ).generate(joined)

    image = BytesIO()
    cloud.to_image().save(image, format="PNG")
    image.seek(0)
    return image
