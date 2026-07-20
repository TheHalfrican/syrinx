"""Voice-profile storage — SQLite + reference-audio samples on disk.

A profile is a named voice, of one of two kinds:
  - **preset**: a built-in engine voice (e.g. Kokoro "af_heart").
  - **cloned**: one or more reference samples (audio + transcript) that a
    cloning engine (Qwen/LuxTTS/...) turns into a zero-shot clone.

Optional **personality** guides the LLM rewrite/compose (when that lands).
Grounded in the Voicebox profiles + profile_samples model, simplified.

Storage: $SYRINX_DATA_DIR/syrinx.db  (default ~/.local/share/syrinx)
Samples: $SYRINX_DATA_DIR/profiles/<profile_id>/<sample_id>.wav
"""

import os
import shutil
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _data_dir() -> Path:
    return Path(
        os.environ.get("SYRINX_DATA_DIR", str(Path.home() / ".local" / "share" / "syrinx"))
    )


@dataclass
class Sample:
    id: str
    audio_path: str
    reference_text: str


@dataclass
class Profile:
    id: str
    name: str
    voice_type: str  # "preset" | "cloned"
    language: str = "en"
    description: str = ""
    personality: str = ""
    default_engine: str = ""  # cloned: e.g. "qwen"
    preset_engine: str = ""  # preset: e.g. "kokoro"
    preset_voice_id: str = ""  # preset: e.g. "af_heart"
    created_at: float = 0.0
    samples: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "voice_type": self.voice_type,
            "language": self.language,
            "has_personality": bool(self.personality),
            "default_engine": self.default_engine,
            "preset_engine": self.preset_engine,
            "preset_voice_id": self.preset_voice_id,
        }

    def full(self) -> dict:
        d = self.summary()
        d.update(
            {
                "description": self.description,
                "personality": self.personality,
                "created_at": self.created_at,
                "samples": [asdict(s) for s in self.samples],
            }
        )
        return d


class ProfileStore:
    def __init__(self) -> None:
        self._dir = _data_dir()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db = str(self._dir / "syrinx.db")
        self._profiles_dir = self._dir / "profiles"
        self._profiles_dir.mkdir(exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._db)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles(
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    voice_type TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    description TEXT DEFAULT '',
                    personality TEXT DEFAULT '',
                    default_engine TEXT DEFAULT '',
                    preset_engine TEXT DEFAULT '',
                    preset_voice_id TEXT DEFAULT '',
                    created_at REAL
                );
                CREATE TABLE IF NOT EXISTS samples(
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                    audio_path TEXT NOT NULL,
                    reference_text TEXT DEFAULT ''
                );
                """
            )

    # --- profiles -------------------------------------------------------

    def create(
        self,
        name: str,
        voice_type: str,
        *,
        language: str = "en",
        description: str = "",
        personality: str = "",
        default_engine: str = "",
        preset_engine: str = "",
        preset_voice_id: str = "",
    ) -> str:
        if voice_type not in ("preset", "cloned"):
            raise ValueError("voice_type must be 'preset' or 'cloned'")
        pid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute(
                "INSERT INTO profiles(id,name,voice_type,language,description,personality,"
                "default_engine,preset_engine,preset_voice_id,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    pid,
                    name,
                    voice_type,
                    language,
                    description,
                    personality,
                    default_engine,
                    preset_engine,
                    preset_voice_id,
                    time.time(),
                ),
            )
        return pid

    def _row_to_profile(self, row: sqlite3.Row, samples: list) -> Profile:
        return Profile(
            id=row["id"],
            name=row["name"],
            voice_type=row["voice_type"],
            language=row["language"],
            description=row["description"],
            personality=row["personality"],
            default_engine=row["default_engine"],
            preset_engine=row["preset_engine"],
            preset_voice_id=row["preset_voice_id"],
            created_at=row["created_at"] or 0.0,
            samples=samples,
        )

    def get(self, profile_id: str) -> Profile | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
            if not row:
                return None
            samples = [
                Sample(s["id"], s["audio_path"], s["reference_text"])
                for s in c.execute(
                    "SELECT * FROM samples WHERE profile_id=?", (profile_id,)
                ).fetchall()
            ]
            return self._row_to_profile(row, samples)

    def list(self) -> list[Profile]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM profiles ORDER BY created_at").fetchall()
            return [self._row_to_profile(r, []) for r in rows]

    def update(self, profile_id: str, **fields) -> None:
        allowed = {"name", "description", "language", "personality", "default_engine"}
        sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not sets:
            return
        cols = ", ".join(f"{k}=?" for k in sets)
        with self._conn() as c:
            c.execute(
                f"UPDATE profiles SET {cols} WHERE id=?", (*sets.values(), profile_id)
            )

    def delete(self, profile_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM profiles WHERE id=?", (profile_id,))
        shutil.rmtree(self._profiles_dir / profile_id, ignore_errors=True)

    # --- samples --------------------------------------------------------

    def add_sample(self, profile_id: str, src_audio: str, reference_text: str) -> Sample:
        if not self.get(profile_id):
            raise ValueError(f"unknown profile: {profile_id}")
        sid = uuid.uuid4().hex[:12]
        dest_dir = self._profiles_dir / profile_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{sid}.wav"
        shutil.copyfile(src_audio, dest)
        with self._conn() as c:
            c.execute(
                "INSERT INTO samples(id,profile_id,audio_path,reference_text) VALUES(?,?,?,?)",
                (sid, profile_id, str(dest), reference_text),
            )
        return Sample(sid, str(dest), reference_text)

    def set_sample_text(self, sample_id: str, reference_text: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE samples SET reference_text=? WHERE id=?", (reference_text, sample_id)
            )

    def delete_sample(self, sample_id: str) -> None:
        with self._conn() as c:
            row = c.execute("SELECT audio_path FROM samples WHERE id=?", (sample_id,)).fetchone()
            c.execute("DELETE FROM samples WHERE id=?", (sample_id,))
        if row:
            Path(row["audio_path"]).unlink(missing_ok=True)
