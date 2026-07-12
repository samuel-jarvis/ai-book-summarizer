import sys
import asyncio
from taskiq_redis import RedisStreamBroker
from app.core.config import settings

broker = RedisStreamBroker(
    url=settings.REDIS_URL,
    queue_name="taskiq_stream",
    consumer_group_name="taskiq_group",
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
