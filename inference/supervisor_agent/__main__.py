from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("VITOOM_SUPERVISOR_AGENT_PORT", "9001"))
    uvicorn.run(
        "inference.supervisor_agent.app:app",
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("VITOOM_SUPERVISOR_AGENT_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()

