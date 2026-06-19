"""Domain-routed dense index tests — STEM pages/queries route to the STEM collection."""
from __future__ import annotations

import pytest

from src.search.dense_router import DomainRoutedDenseIndex


class FakeIndex:
    """Records what was upserted / searched so we can assert routing."""

    def __init__(self, name: str):
        self.name = name
        self.upserts: list[str] = []
        self.searches: list[str] = []
        self.vec_searches = 0

    async def _embed(self, text: str) -> list[float]:
        return [float(len(text))]

    async def upsert(self, page_id: str, text: str, meta=None) -> None:
        self.upserts.append(page_id)

    async def delete(self, page_id: str) -> None:
        pass

    async def search(self, query: str, k: int = 20) -> list[str]:
        self.searches.append(query)
        return [f"{self.name}:{query}"]

    async def search_with_vec(self, vec, k: int = 20) -> list[str]:
        self.vec_searches += 1
        return [f"{self.name}:vec"]

    @property
    def backend(self) -> str:
        return "fake"


@pytest.mark.asyncio
async def test_passthrough_when_stem_disabled() -> None:
    gen = FakeIndex("gen")
    router = DomainRoutedDenseIndex(general=gen, stem=None)
    await router.upsert("p1", "text", {"domain": "math"})
    assert gen.upserts == ["p1"]               # math page still only general (no STEM index)
    assert router.backend == "fake"


@pytest.mark.asyncio
async def test_quantitative_page_indexed_into_both() -> None:
    gen, stem = FakeIndex("gen"), FakeIndex("stem")
    router = DomainRoutedDenseIndex(general=gen, stem=stem)
    await router.upsert("p_math", "solve x", {"domain": "math"})
    await router.upsert("p_doc", "a story", {"domain": "general"})
    assert gen.upserts == ["p_math", "p_doc"]  # everything to general
    assert stem.upserts == ["p_math"]          # only quantitative to STEM
    assert router.backend == "fake+stem"


@pytest.mark.asyncio
async def test_route_search_sends_stem_query_to_stem_index() -> None:
    gen, stem = FakeIndex("gen"), FakeIndex("stem")
    router = DomainRoutedDenseIndex(general=gen, stem=stem)
    out = await router.route_search("Compute the NPV at 8% discount rate", hyde_text=None, k=5)
    assert out == ["stem:vec"]                 # STEM query → STEM index vec search
    assert stem.vec_searches == 1
    assert gen.vec_searches == 0


@pytest.mark.asyncio
async def test_route_search_sends_plain_query_to_general_index() -> None:
    gen, stem = FakeIndex("gen"), FakeIndex("stem")
    router = DomainRoutedDenseIndex(general=gen, stem=stem)
    out = await router.route_search("Who founded Anthropic?", hyde_text=None, k=5)
    assert out == ["gen:vec"]
    assert gen.vec_searches == 1
    assert stem.vec_searches == 0


@pytest.mark.asyncio
async def test_search_with_vec_always_general() -> None:
    gen, stem = FakeIndex("gen"), FakeIndex("stem")
    router = DomainRoutedDenseIndex(general=gen, stem=stem)
    out = await router.search_with_vec([1.0, 2.0], k=5)
    assert out == ["gen:vec"]                  # precomputed vecs are general-space
    assert stem.vec_searches == 0
