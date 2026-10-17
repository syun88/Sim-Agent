from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


class MemoryStore:
    def __init__(self, data_dir: Path | None = None):
        self.uri = os.getenv("NEO4J_URI", "")
        self.username = os.getenv("NEO4J_USERNAME", "")
        self.password = os.getenv("NEO4J_PASSWORD", "")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = None
        self.available = False
        self.last_error: str | None = None
        self.local_path = (data_dir or Path("outputs/memory-fallback.json")).resolve()
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self._connect()

    @property
    def records(self) -> list[dict]:
        if self.local_path.exists():
            try:
                return json.loads(self.local_path.read_text())
            except Exception:
                return []
        return []

    def _connect(self, attempts: int = 2) -> None:
        if not all([self.uri, self.username, self.password]):
            self.last_error = "NEO4J configuration missing"
            return
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password), connection_timeout=5.0)
            self.driver.verify_connectivity()
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1 AS ok").consume()
                for statement in [
                    "CREATE CONSTRAINT task_id IF NOT EXISTS FOR (t:EmbodiedTask) REQUIRE t.task_id IS UNIQUE",
                    "CREATE CONSTRAINT execution_id IF NOT EXISTS FOR (e:EmbodiedExecution) REQUIRE e.execution_id IS UNIQUE",
                    "CREATE INDEX task_instruction IF NOT EXISTS FOR (t:EmbodiedTask) ON (t.instruction)",
                ]:
                    session.run(statement).consume()
            self.available = True
            self.last_error = None
        except Exception as exc:
            self.available = False
            self.last_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

    def save_execution(self, record: dict) -> dict:
        record = {**record, "stored_at": datetime.now(timezone.utc).isoformat()}
        if self.available:
            with self.driver.session(database=self.database) as session:
                session.run(
                    """MERGE (t:EmbodiedTask {task_id:$task_id}) SET t.instruction=$instruction
                    MERGE (e:EmbodiedExecution {execution_id:$execution_id})
                    SET e += $record
                    MERGE (t)-[:HAS_EXECUTION]->(e)
                    WITH e
                    UNWIND $object_ids AS object_id
                    MERGE (o:EmbodiedObject {id:object_id})
                    MERGE (t)-[:TARGETS]->(o)
                    WITH e, $skills AS skills
                    UNWIND skills AS skill_name
                    MERGE (s:EmbodiedSkill {name:skill_name})
                    MERGE (e)-[:USED_SKILL]->(s)
                    WITH e
                    FOREACH (_ IN CASE WHEN e.success THEN [1] ELSE [] END |
                      MERGE (r:EmbodiedOutcome {name:'success'})
                      MERGE (e)-[:RESULTED_IN]->(r)
                    )
                    WITH e
                    FOREACH (_ IN CASE WHEN e.success = false THEN [1] ELSE [] END |
                      MERGE (f:EmbodiedFailure {name:e.failure_category})
                      MERGE (e)-[:FAILED_WITH]->(f)
                    )""",
                    {
                        "task_id": record["task_id"],
                        "execution_id": record["execution_id"],
                        "instruction": record["instruction"],
                        "record": {k: v for k, v in record.items() if isinstance(v, (str, int, float, bool))},
                        "object_ids": record.get("object_ids", []),
                        "skills": record.get("skills", []),
                    },
                ).consume()
            self.local_path.write_text(json.dumps(self.records + [record], default=str, indent=2))
        else:
            self.local_path.write_text(json.dumps(self.records + [record], default=str, indent=2))
        return record

    @staticmethod
    def _summary(record: dict) -> dict:
        """A compact experience summary, the only form a retrieved record is ever embedded in.

        A whole record must not go into the planner: the record carries its attempts, each attempt
        carries the plan, and the plan carries `memory_used` - the records retrieved when it ran.
        Embedding whole records therefore nests the entire history exponentially (measured: one
        record grew to 345 MB). The summary keeps what the planner and the prompt actually read.
        """
        return {
            "task_id": record.get("task_id"),
            "execution_id": record.get("execution_id"),
            "instruction": record.get("instruction"),
            "object_ids": record.get("object_ids"),
            "success": record.get("success"),
            "failure_category": record.get("failure_category"),
            "retry_count": record.get("retry_count"),
            "skills": record.get("skills"),
            "stored_at": record.get("stored_at"),
        }

    def retrieve(self, goal_type: str, object_id: str, failure_category: str | None = None, limit: int = 4) -> list[dict]:
        if self.available:
            query = """MATCH (t:EmbodiedTask)-[:HAS_EXECUTION]->(e)
            WHERE $object_id IN [(t)-[:TARGETS]->(o) | o.id]
            RETURN e {.*} AS experience
            ORDER BY e.success DESC, e.stored_at DESC LIMIT $limit"""
            with self.driver.session(database=self.database) as session:
                return [
                    self._summary(dict(rec["experience"]))
                    for rec in session.run(query, object_id=object_id, limit=limit)
                ]
        relevant = [
            rec for rec in self.records
            if object_id in rec.get("object_ids", [])
            and (failure_category is None or rec.get("failure_category") == failure_category)
        ]
        return [
            self._summary(record)
            for record in sorted(relevant, key=lambda x: x.get("stored_at", ""), reverse=True)[:limit]
        ]

    def history(self, limit: int = 50) -> list[dict]:
        if self.available:
            query = "MATCH (t:EmbodiedTask)-[:HAS_EXECUTION]->(e) RETURN e {.*} AS execution ORDER BY e.stored_at DESC LIMIT $limit"
            with self.driver.session(database=self.database) as session:
                return [dict(rec["execution"]) for rec in session.run(query, limit=limit)]
        return sorted(self.records, key=lambda x: x.get("stored_at", ""), reverse=True)[:limit]

    def close(self):
        if self.driver:
            self.driver.close()
