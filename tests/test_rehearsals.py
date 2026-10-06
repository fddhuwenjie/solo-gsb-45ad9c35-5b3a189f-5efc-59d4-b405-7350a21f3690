"""排练提示：创建排练 / 领取到点提示 / 进度前进 / 发布重定时 / 并发 端到端测试。"""

import math
import threading

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """每个测试用独立 SQLite，并重新初始化模块级 store。"""
    db_path = tmp_path / "rehearsals.db"
    monkeypatch.setenv("TEMPO_DB_PATH", str(db_path))

    import app.main as main
    from app.db import Store

    monkeypatch.setattr(main, "store", Store(str(db_path)))
    yield db_path


from app.main import app  # noqa: E402

client = TestClient(app)

# 120 BPM、4/4、PPQ=480：一拍 0.5s、一小节 2.0s
# -> 1:1:0 = 0s，2:1:0 = 2s，3:1:0 = 4s，4:1:0 = 6s
SCORE_BODY = {
    "ppq": 480,
    "time_signatures": [],
    "tempo_points": [{"position": "1:1:0", "bpm": 120, "curve": "step"}],
    "anchors": [{"position": "1:1:0", "locked": True}],
}
# 60 BPM：一小节 4.0s -> 2:1:0 = 4s，3:1:0 = 8s
SLOW_POINTS = [{"position": "1:1:0", "bpm": 60, "curve": "step"}]
# 240 BPM：一小节 1.0s -> 2:1:0 = 1s，3:1:0 = 2s
FAST_POINTS = [{"position": "1:1:0", "bpm": 240, "curve": "step"}]

CUES = [
    {"position": "1:1:0", "text": "全体就位"},
    {"position": "2:1:0", "text": "进入副歌"},
    {"position": "3:1:0", "text": "间奏开始"},
]


def create_score(body=None):
    resp = client.post("/scores", json=body or SCORE_BODY)
    assert resp.status_code == 201, resp.text
    return resp.json()


def create_rehearsal(score_id=None, cues=None):
    score_id = score_id or create_score()["id"]
    resp = client.post(
        "/rehearsals", json={"score_id": score_id, "cues": cues or CUES}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def claim(rid, elapsed):
    return client.post(f"/rehearsals/{rid}/claims", json={"elapsed_seconds": elapsed})


def make_compute_due(elapsed):
    """与端点内一致的到点计算（供直接走 Store 的并发/重启测试使用）。"""
    from app.main import _build_map

    def compute_due(score, pending):
        tmap = _build_map(
            score["ppq"], score["time_signatures"], score["tempo_points"]
        )
        due = []
        for c in pending:
            trig = tmap.seconds_at_tick(c["total_ticks"])
            if trig <= elapsed + 1e-9 * max(1.0, abs(elapsed)):
                due.append({**c, "trigger_seconds": trig})
        due.sort(key=lambda d: (d["trigger_seconds"], d["cue_index"]))
        return due

    return compute_due


# ---------------------------------------------------------------------- #
# 创建排练
# ---------------------------------------------------------------------- #
def test_create_rehearsal_returns_id_and_validated_cues():
    score = create_score()
    data = create_rehearsal(score["id"])
    assert isinstance(data["id"], str) and len(data["id"]) == 32
    assert data["score_id"] == score["id"]
    assert data["elapsed_seconds"] == 0.0
    assert data["issues"] == []
    assert [c["index"] for c in data["cues"]] == [0, 1, 2]
    assert [c["position"] for c in data["cues"]] == ["1:1:0", "2:1:0", "3:1:0"]
    assert [c["total_ticks"] for c in data["cues"]] == [0, 1920, 3840]
    assert [c["text"] for c in data["cues"]] == [c["text"] for c in CUES]


def test_create_rehearsal_unknown_score_is_404():
    resp = client.post(
        "/rehearsals", json={"score_id": "deadbeef", "cues": CUES}
    )
    assert resp.status_code == 404
    assert resp.json()["detail"][0]["loc"] == ["body", "score_id"]


def test_create_rehearsal_bad_position_is_422_with_index():
    score = create_score()
    cues = [CUES[0], {"position": "1:9:0", "text": "拍越界"}, CUES[2]]
    resp = client.post("/rehearsals", json={"score_id": score["id"], "cues": cues})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert any(e.get("index") == 1 for e in detail)
    assert any("cues" in e["loc"] for e in detail)


def test_create_rehearsal_empty_cues_is_422():
    score = create_score()
    resp = client.post("/rehearsals", json={"score_id": score["id"], "cues": []})
    assert resp.status_code == 422


def test_create_rehearsal_blank_text_is_422_with_index():
    score = create_score()
    cues = [CUES[0], {"position": "2:1:0", "text": "   "}]
    resp = client.post("/rehearsals", json={"score_id": score["id"], "cues": cues})
    assert resp.status_code == 422
    assert any(e.get("index") == 1 for e in resp.json()["detail"])


# ---------------------------------------------------------------------- #
# 领取：到点计算、排序、进度
# ---------------------------------------------------------------------- #
def test_claim_issues_due_cues_sorted_by_trigger_then_entry_order():
    # 录入顺序与触发顺序不同：按触发时间排；相同触发时间按录入顺序排。
    cues = [
        {"position": "3:1:0", "text": "4s 先录入"},
        {"position": "1:1:0", "text": "0s 后录入"},
        {"position": "2:1:0", "text": "2s 甲"},
        {"position": "2:1:0", "text": "2s 乙（同刻，录入在后）"},
    ]
    rh = create_rehearsal(cues=cues)
    resp = claim(rh["id"], 100.0)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["revision"] == 0
    got = [(c["index"], c["text"], c["trigger_seconds"]) for c in data["cues"]]
    assert got == [
        (1, "0s 后录入", 0.0),
        (2, "2s 甲", 2.0),
        (3, "2s 乙（同刻，录入在后）", 2.0),
        (0, "4s 先录入", 4.0),
    ]
    assert [c["position"] for c in data["cues"]] == ["1:1:0", "2:1:0", "2:1:0", "3:1:0"]
    assert all(c["revision"] == 0 for c in data["cues"])


def test_claim_progresses_and_never_reissues():
    rh = create_rehearsal()
    rid = rh["id"]

    data = claim(rid, 0.0).json()
    assert [c["index"] for c in data["cues"]] == [0]
    assert math.isclose(data["cues"][0]["trigger_seconds"], 0.0)

    # 前进但无新到点
    assert claim(rid, 1.5).json()["cues"] == []

    data = claim(rid, 2.0).json()
    assert [c["index"] for c in data["cues"]] == [1]
    assert math.isclose(data["cues"][0]["trigger_seconds"], 2.0)

    # 同一时刻重复领取：不重发
    assert claim(rid, 2.0).json()["cues"] == []

    data = claim(rid, 100.0).json()
    assert [c["index"] for c in data["cues"]] == [2]
    assert data["elapsed_seconds"] == 100.0

    # 发放记录完整留存
    got = client.get(f"/rehearsals/{rid}").json()
    assert got["elapsed_seconds"] == 100.0
    assert [i["index"] for i in got["issues"]] == [0, 1, 2]


def test_claim_backward_is_409_and_state_unchanged():
    rh = create_rehearsal()
    rid = rh["id"]
    assert claim(rid, 2.0).status_code == 200

    resp = claim(rid, 1.0)
    assert resp.status_code == 409
    detail = resp.json()["detail"][0]
    assert detail["loc"] == ["body", "elapsed_seconds"]
    assert math.isclose(detail["current_elapsed_seconds"], 2.0)

    # 状态未变：进度仍是 2.0，已发记录不增不减
    got = client.get(f"/rehearsals/{rid}").json()
    assert got["elapsed_seconds"] == 2.0
    assert [i["index"] for i in got["issues"]] == [0, 1]

    # 之后仍可正常前进
    data = claim(rid, 4.0).json()
    assert [c["index"] for c in data["cues"]] == [2]


def test_claim_unknown_rehearsal_is_404():
    assert claim("deadbeef", 1.0).status_code == 404
    assert client.get("/rehearsals/deadbeef").status_code == 404


def test_claim_invalid_elapsed_is_422():
    rh = create_rehearsal()
    assert claim(rh["id"], -0.5).status_code == 422
    # NaN/Infinity 不是合法 JSON 字面量，用原始请求体发送（服务端 json.loads 可解析）
    for raw in ('{"elapsed_seconds": NaN}', '{"elapsed_seconds": Infinity}'):
        resp = client.post(
            f"/rehearsals/{rh['id']}/claims",
            content=raw,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422, raw
    resp = client.post(f"/rehearsals/{rh['id']}/claims", json={"elapsed_seconds": "1.0x"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------- #
# 发布速度图：已发不撤回，未发按新修订重新定时
# ---------------------------------------------------------------------- #
def test_publish_retimes_pending_cues_but_keeps_issued_records():
    score = create_score()
    sid = score["id"]
    rh = create_rehearsal(sid)
    rid = rh["id"]

    data = claim(rid, 2.0).json()
    assert [c["index"] for c in data["cues"]] == [0, 1]
    assert data["revision"] == 0

    # 发布 60 BPM（rev1）：3:1:0 由 4s 变为 8s
    resp = client.post(
        f"/scores/{sid}/revisions/0/publish", json={"tempo_points": SLOW_POINTS}
    )
    assert resp.status_code == 200, resp.text

    # 旧图下 4s 到点的提示，新图下 5s 仍未到点
    data = claim(rid, 5.0).json()
    assert data["cues"] == []
    assert data["revision"] == 1

    # 8s 到点，按新修订重新定时
    data = claim(rid, 8.0).json()
    assert [c["index"] for c in data["cues"]] == [2]
    cue = data["cues"][0]
    assert math.isclose(cue["trigger_seconds"], 8.0)
    assert cue["revision"] == 1
    assert data["revision"] == 1

    # 已发记录不撤回、不重定时：rev0 的 0s/2s 原样保留
    got = client.get(f"/rehearsals/{rid}").json()
    issues = {i["index"]: i for i in got["issues"]}
    assert len(issues) == 3
    assert issues[0]["revision"] == 0 and math.isclose(issues[0]["trigger_seconds"], 0.0)
    assert issues[1]["revision"] == 0 and math.isclose(issues[1]["trigger_seconds"], 2.0)
    assert issues[2]["revision"] == 1 and math.isclose(issues[2]["trigger_seconds"], 8.0)


def test_same_elapsed_claim_after_publish_picks_up_newly_due_cue():
    # 加速发布后，同一秒数重复领取可领到按新图提前到点的未发提示。
    score = create_score()
    sid = score["id"]
    rh = create_rehearsal(sid)
    rid = rh["id"]

    data = claim(rid, 2.0).json()
    assert [c["index"] for c in data["cues"]] == [0, 1]  # 3:1:0 旧图 4s 未到

    client.post(
        f"/scores/{sid}/revisions/0/publish", json={"tempo_points": FAST_POINTS}
    )
    # 240 BPM 下 3:1:0 = 2s：同一秒数 2.0 再领，新到点，修订号为 1
    data = claim(rid, 2.0).json()
    assert [c["index"] for c in data["cues"]] == [2]
    assert math.isclose(data["cues"][0]["trigger_seconds"], 2.0)
    assert data["cues"][0]["revision"] == 1


def test_interleaved_publish_and_claim_revision_matches_issue_record(fresh_db):
    """发布与领取交错：发放记录中的修订号必须与计算触发秒数所用修订一致。

    120 BPM（rev0）下 3:1:0 = 4s；60 BPM（rev1）下 = 8s。并发执行一次
    发布与多次领取（elapsed=8s，两图下到点时刻不同），最终唯一的发放
    记录里 revision 与 trigger_seconds 必须出自同一修订。
    """
    db_path = fresh_db
    score = create_score()
    sid = score["id"]
    rh = create_rehearsal(sid, cues=[{"position": "3:1:0", "text": "间奏"}])
    rid = rh["id"]

    from app.db import Store
    from app.main import _build_map, _evaluate_replacement

    errors = []

    def publish():
        try:
            writer = Store(str(db_path))
            s = writer.get_score(sid)
            cmap = _build_map(s["ppq"], s["time_signatures"], SLOW_POINTS)
            evaluated = _evaluate_replacement(cmap, s["anchors"], raise_on_conflict=True)
            writer.publish_revision(sid, 0, SLOW_POINTS, evaluated)
        except Exception as exc:  # pragma: no cover - 失败即暴露
            errors.append(exc)

    def do_claim():
        try:
            Store(str(db_path)).claim_cues(rid, 8.0, make_compute_due(8.0))
        except Exception as exc:  # pragma: no cover - 失败即暴露
            errors.append(exc)

    threads = [threading.Thread(target=publish)] + [
        threading.Thread(target=do_claim) for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

    got = client.get(f"/rehearsals/{rid}").json()
    assert len(got["issues"]) == 1
    issue = got["issues"][0]
    expected = 4.0 if issue["revision"] == 0 else 8.0
    assert math.isclose(issue["trigger_seconds"], expected)


# ---------------------------------------------------------------------- #
# 持久化：新 Store 实例（模拟 Compose 重启）后可继续领取
# ---------------------------------------------------------------------- #
def test_persistence_across_store_restart(fresh_db, monkeypatch):
    db_path = fresh_db
    rh = create_rehearsal()
    rid = rh["id"]
    assert [c["index"] for c in claim(rid, 2.0).json()["cues"]] == [0, 1]

    # 重新打开同一文件（不删卷的 Compose 重启等价场景），换挂新 store
    import app.main as main
    from app.db import Store

    monkeypatch.setattr(main, "store", Store(str(db_path)))

    # 进度与发放记录仍在：重复领取不重发，可继续前进
    assert claim(rid, 2.0).json()["cues"] == []
    data = claim(rid, 4.0).json()
    assert [c["index"] for c in data["cues"]] == [2]
    assert claim(rid, 1.0).status_code == 409

    got = client.get(f"/rehearsals/{rid}").json()
    assert got["elapsed_seconds"] == 4.0
    assert [i["index"] for i in got["issues"]] == [0, 1, 2]


# ---------------------------------------------------------------------- #
# 并发领取：同一提示最多发放一次
# ---------------------------------------------------------------------- #
def test_concurrent_claims_issue_each_cue_at_most_once(fresh_db):
    db_path = fresh_db
    cues = [{"position": f"{m}:1:0", "text": f"第{m}小节"} for m in range(1, 9)]
    rh = create_rehearsal(cues=cues)
    rid = rh["id"]

    from app.db import Store

    results = []
    errors = []

    def worker():
        # 独立 Store 实例（独立连接）：仅靠 BEGIN IMMEDIATE 串行化，
        # 比进程内锁更严格地检验“同一提示最多返回一次”。
        try:
            results.append(
                Store(str(db_path)).claim_cues(rid, 100.0, make_compute_due(100.0))
            )
        except Exception as exc:  # pragma: no cover - 失败即暴露
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    issued = [c["cue_index"] for r in results for c in r["cues"]]
    # 8 条提示在 8 个并发领取中合计恰好各出现一次
    assert sorted(issued) == list(range(len(cues)))

    got = client.get(f"/rehearsals/{rid}").json()
    assert sorted(i["index"] for i in got["issues"]) == list(range(len(cues)))
