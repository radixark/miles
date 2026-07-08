import asyncio
import logging

import pytest

from miles.utils.tracking_utils.structured_log import (
    _format_value,
    _to_logfmt,
    log_structured,
    prune_for_log,
    with_logs,
)


class TestFormatValue:
    def test_int_renders_bare(self):
        """An int renders as a bare number."""
        assert _format_value(7) == "7"

    def test_float_renders_bare(self):
        """A float renders as a bare number."""
        assert _format_value(49.3) == "49.3"

    def test_true_and_false_render_lowercase(self):
        """Bools render as lowercase true/false (and take precedence over the int branch)."""
        assert _format_value(True) == "true"
        assert _format_value(False) == "false"

    def test_none_renders_as_empty(self):
        """None renders as an empty value."""
        assert _format_value(None) == ""

    def test_empty_string_renders_as_empty(self):
        """An empty string renders as an empty value with no quotes."""
        assert _format_value("") == ""

    def test_plain_string_renders_bare(self):
        """A string with no space, '=' or quote renders bare."""
        assert _format_value("train") == "train"

    def test_string_with_space_is_quoted(self):
        """A string containing a space is double-quoted so it stays one token."""
        assert _format_value("survivors normal") == '"survivors normal"'

    def test_string_with_equals_is_quoted(self):
        """A string containing '=' is quoted so it can't be mis-read as a new key."""
        assert _format_value("a=b") == '"a=b"'

    def test_string_with_quote_is_escaped(self):
        """A string containing a double-quote is quoted with the inner quote escaped."""
        assert _format_value('say "hi"') == '"say \\"hi\\""'

    def test_string_with_backslash_and_space_escapes_backslash(self):
        """When quoting is triggered, backslashes are escaped too."""
        assert _format_value("a\\b c") == '"a\\\\b c"'

    def test_string_with_newline_is_quoted_and_escaped(self):
        """A literal newline is escaped so the entry stays on one log line."""
        assert _format_value("a\nb") == '"a\\nb"'

    def test_string_with_carriage_return_is_quoted_and_escaped(self):
        """A carriage return is escaped so it can't corrupt the log stream."""
        assert _format_value("a\rb") == '"a\\rb"'

    def test_string_with_tab_is_quoted_and_escaped(self):
        """A tab is escaped so the value stays one unambiguous token."""
        assert _format_value("a\tb") == '"a\\tb"'

    def test_string_with_only_backslash_is_quoted_and_escaped(self):
        """A backslash alone (no space) still triggers quoting so it round-trips."""
        assert _format_value("a\\b") == '"a\\\\b"'

    def test_string_mixing_all_control_chars(self):
        """Newline, quote, tab and backslash all escape together."""
        assert _format_value('a\n"b"\t\\c\r') == '"a\\n\\"b\\"\\t\\\\c\\r"'

    def test_backslash_escaped_before_other_escapes(self):
        """A pre-escaped-looking string round-trips: the backslash doubles, the newline escapes."""
        assert _format_value("\\n\n") == '"\\\\n\\n"'

    def test_list_element_with_newline_is_quoted(self):
        """A newline inside a list element quotes and escapes the joined value."""
        assert _format_value(["a\nb", "c"]) == '"a\\nb,c"'

    def test_list_of_ints_is_comma_joined_no_space(self):
        """A list renders comma-joined with no spaces."""
        assert _format_value([0, 1, 2]) == "0,1,2"

    def test_list_of_bools_lowercased(self):
        """List elements that are bools also render lowercase."""
        assert _format_value([True, False]) == "true,false"

    def test_empty_list_renders_as_empty(self):
        """An empty list renders as an empty value."""
        assert _format_value([]) == ""

    def test_tuple_behaves_like_list(self):
        """A tuple renders the same as the equivalent list."""
        assert _format_value((0, 1)) == "0,1"

    def test_list_whose_joined_form_has_a_space_is_quoted(self):
        """If a list element introduces a space, the whole joined value is quoted."""
        assert _format_value(["a b", "c"]) == '"a b,c"'

    def test_dict_renders_as_quoted_compact_json(self):
        """A dict value renders as quoted compact JSON."""
        assert _format_value({"a": 1}) == '"{\\"a\\":1}"'


class TestToLogfmt:
    def test_fields_join_with_space_in_insertion_order(self):
        """Fields render as space-separated key=value in insertion order."""
        assert _to_logfmt({"cell": 1, "fn": "train", "ok": True}) == "cell=1 fn=train ok=true"

    def test_empty_fields_render_as_empty_string(self):
        """No fields renders as an empty string."""
        assert _to_logfmt({}) == ""

    def test_list_and_empty_fields_keep_one_token_each(self):
        """A list field stays one token; an empty field renders as a trailing 'key='."""
        assert _to_logfmt({"alive": [0, 1], "pending": []}) == "alive=0,1 pending="


class TestPruneForLog:
    def test_small_payload_kept_verbatim(self):
        """A payload under the cap is returned unchanged."""
        payload = {"quorum_id": 1, "healed": [0]}
        assert prune_for_log(payload, cap=160) == payload

    def test_large_list_field_summarized_small_siblings_kept(self):
        """An oversized list field becomes a length summary; small siblings stay inline."""
        pruned = prune_for_log({"quorum_id": 1, "checksums": list(range(500))}, cap=80)
        assert pruned == {"quorum_id": 1, "checksums": "<list len=500>"}

    def test_large_string_field_summarized(self):
        """An oversized string field becomes a char-count summary."""
        assert prune_for_log({"blob": "x" * 1000}, cap=80) == {"blob": "<str 1000 chars>"}

    def test_nested_dict_prunes_only_the_big_subfield(self):
        """Recursion summarizes only the oversized nested field, keeping small ones."""
        pruned = prune_for_log({"a": 1, "b": {"big": list(range(500)), "small": 2}}, cap=60)
        assert pruned == {"a": 1, "b": {"big": "<list len=500>", "small": 2}}

    def test_large_tuple_summarized_as_list(self):
        """An oversized tuple is summarized like a list."""
        assert prune_for_log(tuple(range(500)), cap=80) == "<list len=500>"

    def test_value_at_cap_kept_just_over_summarized(self):
        """A value whose compact-JSON length is <= cap is kept; one char over is summarized."""
        assert prune_for_log("x" * 8, cap=10) == "x" * 8  # json '"xxxxxxxx"' is exactly 10 chars
        assert prune_for_log("x" * 9, cap=10) == "<str 9 chars>"  # json is 11 chars


class TestLogStructured:
    def test_emits_ft_prefixed_logfmt_via_given_method(self, caplog):
        """log_structured emits one 'ft '-prefixed logfmt line through the passed logger method."""
        logger = logging.getLogger("t_emit")
        with caplog.at_level(logging.INFO, logger="t_emit"):
            log_structured(logger.info, op="execute", phase="start", cell=1, fn="train")
        assert caplog.messages == ["ft op=execute phase=start cell=1 fn=train"]

    def test_uses_the_level_of_the_passed_method(self, caplog):
        """The record's level is that of the bound method passed (warning here)."""
        logger = logging.getLogger("t_level")
        with caplog.at_level(logging.DEBUG, logger="t_level"):
            log_structured(logger.warning, op="x")
        assert caplog.records[0].levelno == logging.WARNING

    def test_exc_info_is_forwarded(self, caplog):
        """exc_info=True attaches the active exception to the record."""
        logger = logging.getLogger("t_exc")
        with caplog.at_level(logging.ERROR, logger="t_exc"):
            try:
                raise ValueError("boom")
            except ValueError:
                log_structured(logger.error, op="x", exc_info=True)
        assert caplog.records[0].exc_info[0] is ValueError

    def test_stacklevel_points_to_caller_not_helper(self, caplog):
        """stacklevel=2 makes the record's filename the caller's, not structured_log.py."""
        logger = logging.getLogger("t_stack")
        with caplog.at_level(logging.INFO, logger="t_stack"):
            log_structured(logger.info, op="x")
        assert caplog.records[0].filename == "test_structured_log.py"


class FakeActor:
    @with_logs
    def train(self):
        return 42

    @with_logs
    def update_weights(self):
        raise ValueError("boom")

    @with_logs
    async def send_ckpt(self):
        return "done"

    @with_logs
    async def wait_forever(self):
        await asyncio.Event().wait()


class TestWithLogs:
    def test_sync_method_emits_start_then_end_with_class_and_method(self, caplog):
        """A sync call emits phase=start then phase=end carrying the auto-detected class and method."""
        with caplog.at_level(logging.INFO):
            assert FakeActor().train() == 42
        assert caplog.messages[0] == "ft cls=FakeActor fn=train phase=start"
        assert caplog.messages[1].startswith("ft cls=FakeActor fn=train phase=end ok=true elapsed_s=")

    def test_class_is_read_from_the_runtime_instance(self, caplog):
        """The cls field reflects the actual runtime instance, not where the method was defined."""

        class SubActor(FakeActor):
            pass

        with caplog.at_level(logging.INFO):
            SubActor().train()
        assert caplog.messages[0] == "ft cls=SubActor fn=train phase=start"

    def test_exception_logs_end_not_ok_with_exc_info_and_reraises(self, caplog):
        """On exception the end line is ok=false with the traceback, and the error propagates."""
        with caplog.at_level(logging.INFO):
            with pytest.raises(ValueError, match="boom"):
                FakeActor().update_weights()
        assert caplog.messages[0] == "ft cls=FakeActor fn=update_weights phase=start"
        end = caplog.records[1]
        assert end.getMessage().startswith("ft cls=FakeActor fn=update_weights phase=end ok=false elapsed_s=")
        assert end.levelno == logging.ERROR
        assert end.exc_info[0] is ValueError

    def test_async_method_emits_start_then_end_ok(self, caplog):
        """An async call emits the same start/end pair after the coroutine completes."""
        with caplog.at_level(logging.INFO):
            assert asyncio.run(FakeActor().send_ckpt()) == "done"
        assert caplog.messages[0] == "ft cls=FakeActor fn=send_ckpt phase=start"
        assert caplog.messages[1].startswith("ft cls=FakeActor fn=send_ckpt phase=end ok=true elapsed_s=")

    def test_async_cancellation_logs_cancelled_at_info_and_reraises(self, caplog):
        """Cancelling an async call emits an info end line with cancelled=true and no traceback."""

        async def run() -> None:
            task = asyncio.ensure_future(FakeActor().wait_forever())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        with caplog.at_level(logging.INFO):
            asyncio.run(run())
        end = caplog.records[1]
        assert end.getMessage().startswith("ft cls=FakeActor fn=wait_forever phase=end ok=false elapsed_s=")
        assert end.getMessage().endswith("cancelled=true")
        assert end.levelno == logging.INFO
        assert not end.exc_info

    def test_preserves_wrapped_function_metadata(self):
        """functools.wraps keeps the original __name__ so Ray/introspection still sees it."""
        assert FakeActor.train.__name__ == "train"
