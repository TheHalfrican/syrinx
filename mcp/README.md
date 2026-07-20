# syrinx-mcp

MCP server exposing `syrinx.speak` (and friends) to agents — the equivalent of
Voicebox's agent-initiated speech. It's a thin client of the engine: MCP tool
call → D-Bus `sh.syrinx.Engine1.Speak`.

Undecided: implement in Python (reuse `syrinx_engine`'s bus client) or Rust
(reuse `syrinx-shared`). Python is likely simplest since the MCP + engine share
a language. Stub only for now.
