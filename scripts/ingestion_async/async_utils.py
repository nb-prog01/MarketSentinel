import aiohttp
import asyncio
from aiolimiter import AsyncLimiter
from contextlib import asynccontextmanager
import time

#Configure per API rate limits (tokens, period_seconds)
#Tune these numbers to actual API quotas (this is conservative default)

LIMITERS = {
    "twelve":AsyncLimiter(8, 60), # 5 req/sec
    "alpha":AsyncLimiter(5, 60), 
    "marketaux":AsyncLimiter(1, 5),
    "newsapi":AsyncLimiter(1, 5),
    "fred":AsyncLimiter(1, 3),
    "worldbank":AsyncLimiter(1, 3)
}

DEFAULT_HEADERS = {"User-Agent": "MarketSentinel/async-ingest/1.0"}

@asynccontextmanager
async def get_client():
    timeout=aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout, headers=DEFAULT_HEADERS) as session:
        yield session

# generic http get with limiter and simple retry (network only)
async def http_get(session, url, params=None, limiter=None, max_attempts=4):
    limiter=limiter or LIMITERS.get("default", AsyncLimiter(5,1))
    attempt=0
    while True:
        attempt+=1
        try:
            async with limiter:
                async with session.get(url,params=params) as resp:
                    text=await resp.text()
                    if resp.status>=500:
                        #server error -> retry
                        raise aiohttp.ClientResponseError(status=resp.status, message=text, request_info=resp.request_info, history=resp.history)
                    if resp.status == 429:
                        #rate limit exceeded -> retry
                        rety_after = resp.headers.get("Retry-After")
                        wait=int(rety_after) if rety_after and rety_after.isdigit() else (2 ** attempt)
                        await asyncio.sleep(wait)
                        raise aiohttp.ClientResponseError(status=resp.status, message="429", request_info=resp.request_info, history=resp.history)
                    if resp.status>=400:
                        #client error -> no retry
                        raise Exception(f"HTTP {resp.status}: {text[:300]}")
                    return await resp.json()
        except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError,asyncio.TimeoutError) as e:
            if attempt>=max_attempts:
                raise
            await asyncio.sleep(min(10,2**attempt))
        except Exception:
             #non-network error bubble up
             raise