"""谱例修订的 SQLite 持久化。

保存内容（沿用 POST /convert 的同一套拍号/速度点结构，以 JSON 原样落库）：

* ``scores``：谱例 ID、PPQ、拍号表、速度图（速度点表）、修订号；
* ``anchors``：录音锚点位置、锁定标志、创建时按原图计算并冻结的
  秒数 ``original_seconds`` 与“该位置一个 tick 的时长” ``tick_seconds``
  （锁定偏移阈值）。

发布是单事务：``BEGIN IMMEDIATE`` 取得写锁后按 revision 做乐观并发控制
（预期修订号不匹配 -> 409），成功则一次更新速度图、未锁定锚点秒数和
修订号，锁定锚点秒数不变。进程内再用一把锁串行化写事务。

查询同样是单事务：``GET /scores`` 在同一显式读事务内读取谱例行与全部
锚点行，二者共用一个已提交快照，避免与发布交错时跨修订拼接
（旧修订号配新锚点秒数）。

排练提示（在既有库中新增三张表，``CREATE TABLE IF NOT EXISTS`` 对旧库
直接生效）：

* ``rehearsals``：排练 ID、所属谱例 ID、领取进度 ``elapsed_seconds``
  （从排练开始累计的秒数，只能前进）；
* ``rehearsal_cues``：提示的乐谱位置、文字与创建时展开的 tick 坐标
  （PPQ/拍号在谱例创建后不变，故 tick 坐标不随速度图修订改变）；
* ``cue_issues``：发放记录——提示下标、触发秒数、**计算到点所用的谱例
  修订号**、领取时的客户端秒数。主键 ``(rehearsal_id, cue_index)``
  保证同一提示最多发放一次。

领取是单事务：``BEGIN IMMEDIATE`` 取得写锁（与发布互斥、串行化并发领取），
锁内重读进度（倒退 -> 报错且数据不变），并在**同一事务快照**内读取谱例
修订号与速度图、计算未发提示的触发秒数、写入发放记录并前进进度——
因此发放记录中的修订号必然与计算到点所用的修订一致；已发提示不撤回，
未发提示在下次领取时按新修订重新定时。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    id              TEXT PRIMARY KEY,
    ppq             INTEGER NOT NULL,
    time_signatures TEXT NOT NULL,
    tempo_points    TEXT NOT NULL,
    revision        INTEGER NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS anchors (
    score_id         TEXT NOT NULL,
    position_index   INTEGER NOT NULL,
    position         TEXT NOT NULL,
    locked           INTEGER NOT NULL,
    total_ticks      INTEGER NOT NULL,
    original_seconds REAL NOT NULL,
    current_seconds  REAL NOT NULL,
    tick_seconds     REAL NOT NULL,
    PRIMARY KEY (score_id, position_index),
    FOREIGN KEY (score_id) REFERENCES scores(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rehearsals (
    id              TEXT PRIMARY KEY,
    score_id        TEXT NOT NULL,
    elapsed_seconds REAL NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (score_id) REFERENCES scores(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rehearsal_cues (
    rehearsal_id TEXT NOT NULL,
    cue_index    INTEGER NOT NULL,
    position     TEXT NOT NULL,
    text         TEXT NOT NULL,
    total_ticks  INTEGER NOT NULL,
    PRIMARY KEY (rehearsal_id, cue_index),
    FOREIGN KEY (rehearsal_id) REFERENCES rehearsals(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cue_issues (
    rehearsal_id    TEXT NOT NULL,
    cue_index       INTEGER NOT NULL,
    trigger_seconds REAL NOT NULL,
    revision        INTEGER NOT NULL,
    elapsed_seconds REAL NOT NULL,
    issued_at       TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (rehearsal_id, cue_index),
    FOREIGN KEY (rehearsal_id, cue_index)
        REFERENCES rehearsal_cues(rehearsal_id, cue_index) ON DELETE CASCADE
);
"""


class ScoreNotFound(Exception):
    """谱例 ID 不存在（-> 404）。"""


class RevisionConflict(Exception):
    """预期修订号与当前修订号不一致（-> 409）。"""

    def __init__(self, expected: int, current: int):
        self.expected = expected
        self.current = current
        super().__init__(f"修订号冲突：预期 {expected}，当前 {current}")


class RehearsalNotFound(Exception):
    """排练 ID 不存在（-> 404）。"""


class ClaimBackward(Exception):
    """领取进度倒退：本次秒数早于已领取进度（-> 409，数据不变）。"""

    def __init__(self, current: float):
        self.current = current
        super().__init__(f"领取进度只能前进：当前已领取到 {current} 秒")


def new_score_id() -> str:
    return uuid.uuid4().hex


def new_rehearsal_id() -> str:
    return uuid.uuid4().hex


class Store:
    def __init__(self, db_path: Optional[str] = None):
        # 默认放到工作目录下；容器内由 TEMPO_DB_PATH=/data/tempo.db 指向挂载卷，
        # 从而 `docker compose down` 后再 up（卷保留）仍能读取已发布的数据。
        self.db_path = db_path or os.environ.get(
            "TEMPO_DB_PATH", os.path.join(os.getcwd(), "data", "tempo.db")
        )
        parent = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(parent, exist_ok=True)
        self._write_lock = threading.Lock()
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    def create_score(
        self,
        ppq: int,
        meters: list[dict],
        points: list[dict],
        anchors: list[dict],
    ) -> dict:
        """创建谱例（修订号 0）并冻结各锚点按原图计算的秒数。"""
        score_id = new_score_id()
        with self._write_lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO scores "
                "(id, ppq, time_signatures, tempo_points, revision) "
                "VALUES (?, ?, ?, ?, 0)",
                (score_id, ppq, self._dump(meters), self._dump(points)),
            )
            conn.executemany(
                "INSERT INTO anchors "
                "(score_id, position_index, position, locked, total_ticks, "
                " original_seconds, current_seconds, tick_seconds) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        score_id,
                        i,
                        a["position"],
                        1 if a["locked"] else 0,
                        a["total_ticks"],
                        a["original_seconds"],
                        a["original_seconds"],
                        a["tick_seconds"],
                    )
                    for i, a in enumerate(anchors)
                ],
            )
        return self.get_score(score_id)

    def get_score(self, score_id: str) -> dict:
        """返回谱例与其全部锚点；不存在抛 :class:`ScoreNotFound`。

        两张表在**同一个显式读事务**（延迟事务，快照取自第一条 SELECT）
        内读取：与发布交错时，整次查询要么读到发布前的完整旧修订，
        要么读到发布后的完整新修订，不会出现“旧修订号 + 旧速度点 +
        新修订锚点秒数”的跨修订拼接（WAL 下读者不阻塞写者提交）。
        """
        with self._connect() as conn:
            conn.execute("BEGIN")
            row = self._read_score_row(conn, score_id)
            if row is None:
                conn.rollback()
                raise ScoreNotFound(score_id)
            # 读锚点前即使另一连接提交了发布，本事务仍持第一条 SELECT
            # 取得的快照，锚点与谱例行必然来自同一次已提交修订。
            anchor_rows = self._read_anchor_rows(conn, score_id)
            conn.commit()
        return self._hydrate(row, anchor_rows)

    def _read_score_row(
        self, conn: sqlite3.Connection, score_id: str
    ) -> Optional[sqlite3.Row]:
        """读谱例行（独立方法，便于测试在该行读取前后插入发布）。"""
        return conn.execute(
            "SELECT * FROM scores WHERE id = ?", (score_id,)
        ).fetchone()

    def _read_anchor_rows(
        self, conn: sqlite3.Connection, score_id: str
    ) -> list[sqlite3.Row]:
        """读全部锚点行；须在 :meth:`get_score` 的同一读事务内调用。"""
        return conn.execute(
            "SELECT * FROM anchors WHERE score_id = ? ORDER BY position_index",
            (score_id,),
        ).fetchall()

    def publish_revision(
        self,
        score_id: str,
        expected_revision: int,
        points: list[dict],
        evaluated: list[dict],
    ) -> dict:
        """按乐观锁发布整份替换速度点表。

        ``evaluated`` 为按候选图重新计算后的各锚点结果（来自引擎层的
        纯函数评估），每项含 total_ticks / original_seconds / current_seconds
        / locked / tick_seconds。调用方须在进入本方法前完成
        “锁定偏移超过一 tick”的 422 判定。

        单事务：revision 不符 -> :class:`RevisionConflict`（数据不变）；
        成功则一次更新速度图、未锁定锚点秒数和修订号，锁定秒数不变。
        """
        with self._write_lock, self._connect() as conn:
            # BEGIN IMMEDIATE：立刻拿到写锁，序列化并发发布，
            # 锁内重读 revision，杜绝两个请求同时用同一旧修订号覆盖。
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT revision FROM scores WHERE id = ?", (score_id,)
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ScoreNotFound(score_id)
            if row["revision"] != expected_revision:
                conn.rollback()
                raise RevisionConflict(expected_revision, row["revision"])

            conn.execute(
                "UPDATE scores SET tempo_points = ?, revision = ? WHERE id = ?",
                (self._dump(points), expected_revision + 1, score_id),
            )
            for i, a in enumerate(evaluated):
                # 锁定锚点：只更新 tick 坐标（位置字符串不变），秒数保持创建时的值。
                if a["locked"]:
                    conn.execute(
                        "UPDATE anchors SET total_ticks = ? "
                        "WHERE score_id = ? AND position_index = ?",
                        (a["total_ticks"], score_id, i),
                    )
                else:
                    conn.execute(
                        "UPDATE anchors SET total_ticks = ?, current_seconds = ? "
                        "WHERE score_id = ? AND position_index = ?",
                        (a["total_ticks"], a["current_seconds"], score_id, i),
                    )
            conn.commit()
        return self.get_score(score_id)

    # ------------------------------------------------------------------ #
    # 排练提示：创建 / 查询 / 领取
    # ------------------------------------------------------------------ #
    def create_rehearsal(self, score_id: str, cues: list[dict]) -> dict:
        """创建排练（领取进度 0）并落库全部提示。

        ``cues`` 每项含 position / text / total_ticks（tick 坐标由调用方
        按谱例创建时的 PPQ 与拍号展开；二者不随修订改变，故坐标稳定）。
        谱例不存在抛 :class:`ScoreNotFound`。
        """
        rehearsal_id = new_rehearsal_id()
        with self._write_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT id FROM scores WHERE id = ?", (score_id,)
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ScoreNotFound(score_id)
            conn.execute(
                "INSERT INTO rehearsals (id, score_id, elapsed_seconds) "
                "VALUES (?, ?, 0.0)",
                (rehearsal_id, score_id),
            )
            conn.executemany(
                "INSERT INTO rehearsal_cues "
                "(rehearsal_id, cue_index, position, text, total_ticks) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (rehearsal_id, i, c["position"], c["text"], c["total_ticks"])
                    for i, c in enumerate(cues)
                ],
            )
            conn.commit()
        return self.get_rehearsal(rehearsal_id)

    def get_rehearsal(self, rehearsal_id: str) -> dict:
        """返回排练、全部提示与发放记录；不存在抛 :class:`RehearsalNotFound`。

        与 :meth:`get_score` 相同，三张表在同一个显式读事务内读取，
        共用同一已提交快照。
        """
        with self._connect() as conn:
            conn.execute("BEGIN")
            row = conn.execute(
                "SELECT * FROM rehearsals WHERE id = ?", (rehearsal_id,)
            ).fetchone()
            if row is None:
                conn.rollback()
                raise RehearsalNotFound(rehearsal_id)
            cue_rows = conn.execute(
                "SELECT * FROM rehearsal_cues WHERE rehearsal_id = ? "
                "ORDER BY cue_index",
                (rehearsal_id,),
            ).fetchall()
            issue_rows = conn.execute(
                "SELECT * FROM cue_issues WHERE rehearsal_id = ? "
                "ORDER BY cue_index",
                (rehearsal_id,),
            ).fetchall()
            conn.commit()
        return self._hydrate_rehearsal(row, cue_rows, issue_rows)

    def claim_cues(
        self,
        rehearsal_id: str,
        elapsed_seconds: float,
        compute_due,
    ) -> dict:
        """单事务领取到点提示（进度只能前进）。

        ``compute_due(score, pending)`` 为纯函数：``score`` 是本事务快照内
        读出的谱例（含 ppq / time_signatures / tempo_points / revision），
        ``pending`` 为未发提示（含 cue_index / position / text /
        total_ticks）；返回本次到点的提示列表，每项至少含 ``cue_index``
        与 ``trigger_seconds``。

        事务内依次：锁内重读进度（``elapsed_seconds`` 小于已存进度 ->
        :class:`ClaimBackward`，数据不变）；同一快照读谱例与未发提示并
        计算到点；写入发放记录（记录本次所用修订号）并前进进度。
        由于计算与写记录在同一事务、同一快照，发放记录中的修订号必然
        与计算到点所用的修订一致；``BEGIN IMMEDIATE`` 与发布及其余领取
        互斥，主键 ``(rehearsal_id, cue_index)`` 兜底保证同一提示最多
        发放一次。
        """
        with self._write_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM rehearsals WHERE id = ?", (rehearsal_id,)
            ).fetchone()
            if row is None:
                conn.rollback()
                raise RehearsalNotFound(rehearsal_id)
            current = float(row["elapsed_seconds"])
            if elapsed_seconds < current:
                conn.rollback()
                raise ClaimBackward(current)

            score_row = conn.execute(
                "SELECT * FROM scores WHERE id = ?", (row["score_id"],)
            ).fetchone()
            if score_row is None:
                conn.rollback()
                raise ScoreNotFound(row["score_id"])
            score = {
                "id": score_row["id"],
                "ppq": score_row["ppq"],
                "time_signatures": json.loads(score_row["time_signatures"]),
                "tempo_points": json.loads(score_row["tempo_points"]),
                "revision": score_row["revision"],
            }

            cue_rows = conn.execute(
                "SELECT * FROM rehearsal_cues WHERE rehearsal_id = ? "
                "ORDER BY cue_index",
                (rehearsal_id,),
            ).fetchall()
            issued = {
                r["cue_index"]
                for r in conn.execute(
                    "SELECT cue_index FROM cue_issues WHERE rehearsal_id = ?",
                    (rehearsal_id,),
                )
            }
            pending = [
                {
                    "cue_index": c["cue_index"],
                    "position": c["position"],
                    "text": c["text"],
                    "total_ticks": c["total_ticks"],
                }
                for c in cue_rows
                if c["cue_index"] not in issued
            ]

            due = list(compute_due(score, pending))
            for d in due:
                conn.execute(
                    "INSERT INTO cue_issues "
                    "(rehearsal_id, cue_index, trigger_seconds, revision, "
                    " elapsed_seconds) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        rehearsal_id,
                        d["cue_index"],
                        d["trigger_seconds"],
                        score["revision"],
                        elapsed_seconds,
                    ),
                )
            conn.execute(
                "UPDATE rehearsals SET elapsed_seconds = ? WHERE id = ?",
                (elapsed_seconds, rehearsal_id),
            )
            conn.commit()
        return {
            "id": rehearsal_id,
            "score_id": score["id"],
            "elapsed_seconds": elapsed_seconds,
            "revision": score["revision"],
            "cues": due,
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _hydrate(row: sqlite3.Row, anchor_rows: list[sqlite3.Row]) -> dict:
        return {
            "id": row["id"],
            "ppq": row["ppq"],
            "time_signatures": json.loads(row["time_signatures"]),
            "tempo_points": json.loads(row["tempo_points"]),
            "revision": row["revision"],
            "created_at": row["created_at"],
            "anchors": [
                {
                    "index": a["position_index"],
                    "position": a["position"],
                    "locked": bool(a["locked"]),
                    "total_ticks": a["total_ticks"],
                    "original_seconds": a["original_seconds"],
                    "current_seconds": a["current_seconds"],
                    "tick_seconds": a["tick_seconds"],
                }
                for a in anchor_rows
            ],
        }

    @staticmethod
    def _hydrate_rehearsal(
        row: sqlite3.Row,
        cue_rows: list[sqlite3.Row],
        issue_rows: list[sqlite3.Row],
    ) -> dict:
        return {
            "id": row["id"],
            "score_id": row["score_id"],
            "elapsed_seconds": row["elapsed_seconds"],
            "created_at": row["created_at"],
            "cues": [
                {
                    "index": c["cue_index"],
                    "position": c["position"],
                    "text": c["text"],
                    "total_ticks": c["total_ticks"],
                }
                for c in cue_rows
            ],
            "issues": [
                {
                    "index": i["cue_index"],
                    "trigger_seconds": i["trigger_seconds"],
                    "revision": i["revision"],
                    "elapsed_seconds": i["elapsed_seconds"],
                    "issued_at": i["issued_at"],
                }
                for i in issue_rows
            ],
        }
