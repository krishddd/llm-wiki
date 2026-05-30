"""APScheduler in-process job scheduler for v2 background upkeep.

Runs inside the uvicorn process. Five jobs (all UTC):

| time          | job                | what it does                                |
|---------------|--------------------|---------------------------------------------|
| daily 03:00   | decay_sweep        | recompute facts.confidence via Ebbinghaus   |
| daily 03:30   | episodic_prune     | delete episodic files older than retention   |
| daily 04:00   | promote_episodic   | episodic → semantic auto-promotion           |
| weekly Sun05  | lint_autofix       | lint with auto-repair (orphans, stale)       |
| weekly Sun06  | detect_procedures  | promote recurring patterns to wiki/procedures|

Stops when uvicorn stops. All jobs are also exposed via `POST /admin/run/{job}`
for manual triggers, so users don't have to wait for cron.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)


JOB_REGISTRY: dict[str, Callable[..., Awaitable]] = {}


def register_job(name: str):
    """Decorator: register an async job under `name` so /admin/run/{name} works."""
    def deco(fn):
        JOB_REGISTRY[name] = fn
        return fn
    return deco


# ───────────────────────────────────────────────────────────────────
# Job implementations — thin wrappers around modules built in Phase B/C/E.
# Each receives `state` (the FastAPI app state) and returns a dict result.
# ───────────────────────────────────────────────────────────────────


@register_job("decay_sweep")
async def _decay_sweep(state) -> dict:
    from .config import get_settings
    from .wiki.lifecycle import LifecycleConfig, decay_sweep
    s = get_settings()
    cfg = LifecycleConfig(
        half_life_days=getattr(s, "decay_half_life_days", 90.0),
        reinforcement_threshold=getattr(s, "reinforcement_threshold", 3),
        reinforcement_window_days=getattr(s, "reinforcement_window_days", 14),
        enabled=getattr(s, "lifecycle_enabled", True),
    )
    return await decay_sweep(state.graph, cfg=cfg)


@register_job("episodic_prune")
async def _episodic_prune(state) -> dict:
    from .config import get_settings
    from .wiki.episodic import prune_old_episodes
    s = get_settings()
    deleted = prune_old_episodes(
        s.wiki_dir,
        retention_days=getattr(s, "episodic_retention_days", 14),
    )
    return {"deleted": deleted}


@register_job("promote_episodic")
async def _promote_episodic(state) -> dict:
    from .config import get_settings
    from .llm import get_client
    from .wiki.promote import promote_episodic_to_semantic
    s = get_settings()
    return await promote_episodic_to_semantic(
        wiki_dir=s.wiki_dir,
        bm25=state.bm25,
        dense=state.dense,
        client=get_client(),
        days=14,
        min_repeats=3,
    )


@register_job("detect_procedures")
async def _detect_procedures(state) -> dict:
    from .config import get_settings
    from .llm import get_client
    from .wiki.procedures import detect_procedures
    s = get_settings()
    if state.procedures is None:
        return {"skipped": "procedures-store-not-initialised"}
    return await detect_procedures(
        store=state.procedures,
        wiki_dir=s.wiki_dir,
        client=get_client(),
        min_hits=5,
    )


@register_job("lint_autofix")
async def _lint_autofix(state) -> dict:
    from .lint import lint_wiki
    # Phase E1 will extend lint_wiki with auto_fix=True. The function defaults
    # to read-only behaviour today — wire the kwarg conditionally so this job
    # works both before and after Phase E ships.
    try:
        return await lint_wiki(
            page_store=state.page_store,
            graph=state.graph,
            auto_fix=True,
        )
    except TypeError:
        return await lint_wiki(page_store=state.page_store)


@register_job("lint")
async def _lint(state) -> dict:
    """Read-only lint, callable on demand."""
    from .lint import lint_wiki
    return await lint_wiki(page_store=state.page_store)


@register_job("page_compaction")
async def _page_compaction(state) -> dict:
    from .config import get_settings
    from .llm import get_client
    from .wiki.compaction import compact_bloated_pages
    s = get_settings()
    return await compact_bloated_pages(
        wiki_dir=s.wiki_dir,
        client=get_client(),
    )


# ───────────────────────────────────────────────────────────────────
# Scheduler factory
# ───────────────────────────────────────────────────────────────────


def make_scheduler(state):
    """Build and return an AsyncIOScheduler with all v2 jobs registered.

    Caller is responsible for `.start()` and `.shutdown()`.
    """
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError as e:
        log.warning(
            "APScheduler not installed; scheduler disabled. "
            "Run `pip install APScheduler>=3.10`",
            extra={"metadata": {"error": str(e)[:120]}},
        )
        return None

    from .config import get_settings
    s = get_settings()

    if not getattr(s, "scheduler_enabled", True):
        log.info("scheduler disabled via config")
        return None

    sched = AsyncIOScheduler(timezone="UTC")

    schedule = [
        # name, hour, minute, day_of_week (None = daily)
        ("decay_sweep",       3,  0,  None),
        ("episodic_prune",    3, 30,  None),
        ("promote_episodic",  4,  0,  None),
        ("lint_autofix",      5,  0,  "sun"),
        ("detect_procedures", 6,  0,  "sun"),
        ("page_compaction",   7,  0,  "sun"),
    ]
    for name, hour, minute, dow in schedule:
        if not getattr(s, f"job_{name}_enabled", True):
            continue
        fn = JOB_REGISTRY[name]
        trigger = (
            CronTrigger(hour=hour, minute=minute, day_of_week=dow, timezone="UTC")
            if dow else CronTrigger(hour=hour, minute=minute, timezone="UTC")
        )
        async def _runner(state=state, fn=fn, name=name):
            try:
                result = await fn(state)
                log.info("scheduled job complete",
                         extra={"metadata": {"job": name, "result": result}})
            except Exception as e:
                log.warning("scheduled job failed",
                            extra={"metadata": {"job": name, "error": str(e)[:200]}})
        sched.add_job(
            _runner, trigger=trigger,
            id=f"job_{name}", name=name, replace_existing=True, max_instances=1,
        )

    log.info(
        "scheduler built",
        extra={"metadata": {"jobs": [j.id for j in sched.get_jobs()]}},
    )
    return sched


async def run_job_now(state, name: str) -> dict:
    """Manual trigger: run a registered job immediately and return its result."""
    fn = JOB_REGISTRY.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown job: {name}",
                "available": sorted(JOB_REGISTRY.keys())}
    try:
        result = await fn(state)
        return {"ok": True, "job": name, "result": result}
    except Exception as e:
        log.warning("manual job failed",
                    extra={"metadata": {"job": name, "error": str(e)[:200]}})
        return {"ok": False, "job": name, "error": str(e)[:300]}
