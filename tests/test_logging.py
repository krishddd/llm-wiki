import json
import logging

from src.logging_config import correlation_id_ctx, new_correlation_id, setup_logging


def test_correlation_id_in_log(tmp_path):
    # setup_logging is idempotent across the process — reset so the handlers point at this tmp_path.
    from src import logging_config as lc
    lc._configured = False
    setup_logging(tmp_path)
    cid = new_correlation_id()
    token = correlation_id_ctx.set(cid)
    logging.getLogger("test").info("hello", extra={"metadata": {"k": "v"}})
    correlation_id_ctx.reset(token)

    content = (tmp_path / "app.log").read_text(encoding="utf-8").strip().splitlines()
    assert any(cid in line for line in content)
    parsed = json.loads(content[-1])
    assert parsed["correlation_id"] == cid
    assert parsed["metadata"]["k"] == "v"
