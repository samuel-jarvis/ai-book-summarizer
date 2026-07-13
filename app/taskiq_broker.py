import asyncio
import logging
import sys

from redis.asyncio import Redis
from redis.exceptions import RedisError
from taskiq_redis import RedisStreamBroker

from app.core.config import settings

logger = logging.getLogger(__name__)

REDIS_STREAM_BLOCK_MS = 1_000
REDIS_RETRY_DELAY_SECONDS = 1


class ResilientRedisStreamBroker(RedisStreamBroker):
    """Reconnect instead of terminating a worker after a Redis read failure."""

    async def listen(self):
        while True:
            try:
                async for message in super().listen():
                    yield message
            except RedisError as exc:
                logger.warning(
                    "Redis task queue connection failed; retrying in %s second(s): %s",
                    REDIS_RETRY_DELAY_SECONDS,
                    exc,
                )
                await self.connection_pool.disconnect()
                await asyncio.sleep(REDIS_RETRY_DELAY_SECONDS)


broker = ResilientRedisStreamBroker(
    url=settings.REDIS_URL,
    queue_name="taskiq_stream",
    consumer_group_name="taskiq_group",
    xread_block=REDIS_STREAM_BLOCK_MS,
    socket_timeout=10,
    socket_connect_timeout=5,
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def is_task_queue_available() -> bool:
    """Return whether the Redis-backed task queue can accept work."""
    try:
        async with Redis(connection_pool=broker.connection_pool) as redis:
            return bool(await redis.ping())
    except RedisError:
        return False
