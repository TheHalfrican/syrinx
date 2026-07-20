"""Entry point: register the Syrinx engine on the session bus and run forever.

Run directly (`python -m syrinx_engine`) or via the `syrinx-engine` script.
In production this is started by the `syrinx-engine.service` user unit.
"""

import asyncio
import logging

from dbus_next.aio import MessageBus

from .service import EngineInterface

log = logging.getLogger("syrinx.engine")


async def _run() -> None:
    bus = await MessageBus().connect()
    engine = EngineInterface()

    bus.export("/sh/syrinx/Engine", engine)
    await bus.request_name("sh.syrinx.Engine")

    log.info("syrinx-engine ready on sh.syrinx.Engine (backend=%s)", engine.backend_name)

    # Warm the models in the background so the bus name is claimed immediately
    # and clients can attach while weights load.
    asyncio.create_task(engine.warmup())

    await asyncio.get_event_loop().create_future()  # run forever


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
