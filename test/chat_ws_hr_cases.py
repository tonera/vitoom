#!/usr/bin/env python
"""Run end-to-end HR natural-language cases through `/ws/chat/{session_id}`.

This script is intentionally higher level than `scripts/hr_demo_es_cases.py`:
it sends human-like chat prompts through the real chat websocket path and checks
that the HR business query tool is selected and the final answer contains the
expected business signals.

Example:
  python test/chat_ws_hr_cases.py --no-interactive
  python test/chat_ws_hr_cases.py --case HR-001 --case HR-006 --no-interactive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import websockets  # type: ignore[import-not-found]
from websockets.exceptions import ConnectionClosed  # type: ignore[import-not-found]

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import chat_ws_experiment as chat_ws  # noqa: E402


@dataclass(frozen=True)
class HRChatCase:
    case_id: str
    prompt: str
    expected_all: tuple[str, ...] = ()
    expected_any: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = (
        "HR 查询执行失败",
        "QuerySpec 校验失败",
        "不属于员工业务查询入口",
        "index_not_found_exception",
    )


HR_CHAT_CASES: tuple[HRChatCase, ...] = (
    HRChatCase("HR-001", "帮我看下咱们北京办公室现在一共有多少员工？", expected_all=("北京",), expected_any=("员工", "人")),
    HRChatCase("HR-002", "我想知道公司各个部门的人数大概怎么分布。", expected_any=("部门", "department", "MANU", "RND")),
    HRChatCase("HR-003", "公司男女比例现在是什么情况？", expected_any=("男", "女", "性别")),
    HRChatCase("HR-004", "已婚员工大概占多少比例？", expected_all=("已婚",), expected_any=("占", "比例", "%")),
    HRChatCase("HR-005", "帮我按 GR-03 到 GR-16 看一下各职级的人数分布。", expected_any=("GR", "职级")),
    HRChatCase("HR-006", "James Kwok 手下现在有几个直接下属？", expected_all=("James",), expected_any=("直接下属", "下属")),
    HRChatCase("HR-007", "麻烦列一下所有向 107009 这个经理汇报的人。", expected_any=("107009", "James", "汇报", "下属")),
    HRChatCase("HR-008", "哪个经理带的人最多？给我看一下。", expected_any=("最多", "直接下属", "下属")),
    HRChatCase("HR-009", "北京和上海两地员工的汇报关系帮我对比一下。", expected_all=("北京", "上海"), expected_any=("汇报", "经理")),
    HRChatCase("HR-010", "有没有员工资料里没有填直属经理？", expected_any=("没有指定经理", "直属经理", "经理")),
    HRChatCase("HR-011", "2023 年入职的新员工数量是多少？", expected_all=("2023",), expected_any=("入职", "员工", "人")),
    HRChatCase("HR-012", "当前在职员工平均司龄大概多少年？", expected_all=("平均",), expected_any=("司龄", "年")),
    HRChatCase("HR-013", "全职 FTE=1 和非全职员工的分布情况给我看下。", expected_any=("FTE", "全职")),
    HRChatCase("HR-014", "计时工和固定职工现在分别占多少？", expected_any=("计时工", "固定职工")),
    HRChatCase("HR-015", "把已经退休或者被解雇的人员名单查出来。", expected_any=("退休", "解雇", "离职", "非在职")),
    HRChatCase("HR-016", "帮我找一下 GR-08 及以上职级的员工名单。", expected_any=("GR-08", "GR", "职级")),
    HRChatCase("HR-017", "不同 Pay Scale Group 的平均 FTE 是多少？", expected_all=("FTE",), expected_any=("Pay Scale", "薪酬等级", "PSG")),
    HRChatCase("HR-018", "时薪制员工和月薪制员工分别有多少人？", expected_any=("时薪", "月薪", "计时工", "固定职工")),
    HRChatCase("HR-019", "VP 级别以上的高管都有谁？", expected_any=("VP", "高管", "GR")),
    HRChatCase("HR-020", "标准工时是 40 小时的员工占比帮我算一下。", expected_all=("40",), expected_any=("标准工时", "占比", "%")),
    HRChatCase("HR-021", "现在标记为高潜力的人才有多少？", expected_any=("高潜", "高潜力")),
    HRChatCase("HR-022", "找出绩效低而且流失风险高的员工。", expected_any=("绩效", "流失风险", "低", "高")),
    HRChatCase("HR-023", "哪些人是关键岗位，也就是 keyPosition=1？", expected_any=("关键岗位", "关键")),
    HRChatCase("HR-024", "帮我列一下 futureLeader 未来领袖标签的人。", expected_any=("futureLeader", "未来领袖")),
    HRChatCase("HR-025", "近期很可能升职的人有哪些？", expected_any=("很可能", "升职", "晋升")),
    HRChatCase("HR-026", "使用 Asia/Shanghai 时区工作的员工有哪些？", expected_all=("Asia/Shanghai",), expected_any=("员工", "人")),
    HRChatCase("HR-027", "深圳办公室人员按部门分布是怎样的？", expected_all=("深圳",), expected_any=("部门", "分布")),
    HRChatCase("HR-028", "有没有跨国或者跨时区汇报的情况，经理和员工时区不一样的那种？", expected_any=("跨时区", "时区", "汇报")),
    HRChatCase("HR-029", "按时区统计一下公司人员数量。", expected_any=("时区", "timezone")),
    HRChatCase("HR-030", "美国时区的员工名单给我看一下。", expected_any=("美国", "时区", "员工")),
    HRChatCase("HR-031", "北京 MANU 制造部门里，目前在职的工程师有哪些？", expected_all=("北京",), expected_any=("MANU", "制造", "工程师")),
    HRChatCase("HR-032", "入职超过 5 年，而且绩效评级高的员工帮我筛出来。", expected_all=("5",), expected_any=("绩效", "高", "入职")),
    HRChatCase("HR-033", "找一下 GR-10 以上、并且直接下属超过 3 个人的经理。", expected_any=("GR-10", "经理", "直接下属")),
    HRChatCase("HR-034", "2023 年之后入职的女性高管有哪些？", expected_all=("2023",), expected_any=("女性", "女", "高管")),
    HRChatCase("HR-035", "帮我找有离职风险且影响度高的关键人才。", expected_any=("离职风险", "流失风险", "关键人才", "关键")),
    HRChatCase(
        "HR-036",
        "李一164有没有离职风险？",
        expected_all=("李一164",),
        expected_any=("离职风险", "流失风险", "高"),
        forbidden=("非在职", "HR 查询执行失败", "QuerySpec 校验失败", "index_not_found_exception"),
    ),
)


def _matches_any(text: str, values: Iterable[str]) -> bool:
    items = tuple(item for item in values if item)
    return not items or any(item in text for item in items)


def _validate_answer(case: HRChatCase, answer: str) -> list[str]:
    failures: list[str] = []
    for item in case.expected_all:
        if item and item not in answer:
            failures.append(f"missing required text: {item!r}")
    if not _matches_any(answer, case.expected_any):
        failures.append(f"missing any expected text: {case.expected_any!r}")
    for item in case.forbidden:
        if item and item in answer:
            failures.append(f"forbidden text present: {item!r}")
    return failures


async def _run_one_case(
    *,
    case: HRChatCase,
    token: str,
    load_name: str,
    args: argparse.Namespace,
) -> tuple[bool, str, dict[str, Any]]:
    session = await chat_ws._create_chat_session(  # noqa: SLF001
        token=token,
        api_url=args.api_url,
        load_name=load_name,
        input_mode="text",
        output_mode="text_stream",
        title=f"hr_ws_{case.case_id}",
    )
    session_id = str(session["id"])
    ws_url = chat_ws._ws_url(session_id, token)  # noqa: SLF001
    result: dict[str, Any] = {}
    try:
        async with websockets.connect(ws_url, max_size=None, ping_interval=20) as ws:
            await chat_ws._drain_until_ready(ws)  # noqa: SLF001
            expect_tool_names: Optional[set[str]] = None if args.no_expect_tool else {"hr_business_query"}
            result = await chat_ws._run_text_turn(  # noqa: SLF001
                ws,
                case.prompt,
                expect_tool_names=expect_tool_names,
                timeout_s=args.turn_timeout,
            )
            try:
                await ws.send(json.dumps({"type": "session_close", "payload": {}}))
                await chat_ws._recv_chat_json(ws, timeout=5.0)  # noqa: SLF001
            except (ConnectionClosed, asyncio.TimeoutError):
                pass
    finally:
        chat_ws._audio_dump_flush_all()  # noqa: SLF001

    terminal = result.get("terminal_event") if isinstance(result, dict) else {}
    payload = terminal.get("payload") if isinstance(terminal, dict) else {}
    answer = str(payload.get("content") or "") if isinstance(payload, dict) else ""
    failures = _validate_answer(case, answer)
    if failures:
        return False, "; ".join(failures), result
    return True, answer, result


async def main_async(args: argparse.Namespace) -> int:
    _, token = await chat_ws._pick_user_and_token(args.token)  # noqa: SLF001
    load_name = args.load_name or ""
    if not load_name and args.auto_model:
        load_name = await chat_ws._auto_pick_load_name(token, args.api_url)  # noqa: SLF001

    selected_ids = {item.upper() for item in args.case or []}
    cases = [case for case in HR_CHAT_CASES if not selected_ids or case.case_id.upper() in selected_ids]
    if args.limit:
        cases = cases[: max(0, args.limit)]
    if not cases:
        print("No HR chat cases selected.")
        return 1

    passed = 0
    results: list[dict[str, Any]] = []
    for case in cases:
        print(f"\n=== {case.case_id} ===")
        ok, message, result = await _run_one_case(case=case, token=token, load_name=load_name, args=args)
        tool_names = result.get("tool_names") if isinstance(result, dict) else []
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case.case_id} tools={tool_names}")
        if not ok:
            print(message)
            if args.fail_fast:
                results.append({"case_id": case.case_id, "ok": ok, "message": message, "tools": tool_names})
                break
        else:
            passed += 1
        results.append({"case_id": case.case_id, "ok": ok, "message": message[:500], "tools": tool_names})

    total = len(results)
    print(f"\nHR chat websocket cases passed: {passed}/{total}")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json}")
    return 0 if passed == total else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HR websocket E2E natural-language cases")
    parser.add_argument("--api-url", default=chat_ws.API_BASE_URL)
    parser.add_argument("--token", default=None, help="外部 JWT；不传则自动挑 active user 自签")
    parser.add_argument("--load-name", default=None, help="会话加载名；不传则使用后端默认模型")
    parser.add_argument("--auto-model", action="store_true", help="确认 running 推理服务并使用后端默认模型")
    parser.add_argument("--case", action="append", default=[], help="只跑指定 case_id，可重复，例如 --case HR-001")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，用于快速探测")
    parser.add_argument("--turn-timeout", type=float, default=180.0, help="单个 case 超时秒数")
    parser.add_argument("--no-expect-tool", action="store_true", help="不强制要求命中 hr_business_query 工具")
    parser.add_argument("--fail-fast", action="store_true", help="首个失败后停止")
    parser.add_argument("--output-json", default="", help="保存结果摘要 JSON")
    parser.add_argument("--list-cases", action="store_true", help="列出 case 后退出")
    parser.add_argument("--no-interactive", action="store_true", help="兼容 chat_ws_experiment 调用习惯；本脚本始终非交互")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_cases:
        for case in HR_CHAT_CASES:
            print(f"{case.case_id}\t{case.prompt}")
        raise SystemExit(0)
    try:
        raise SystemExit(asyncio.run(main_async(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
