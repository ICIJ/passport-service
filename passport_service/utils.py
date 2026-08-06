import asyncio
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Coroutine,
    Iterable,
    Iterator,
    Sequence,
)
from itertools import islice

from aiostream.stream import flatten


async def run_with_concurrency(
    aws: Iterable[Coroutine | asyncio.Future], max_concurrency: int
) -> AsyncGenerator:
    max_concurrency = asyncio.Semaphore(max_concurrency)
    aws = [_run_with_semaphore(aw, max_concurrency) for aw in aws]
    for res in asyncio.as_completed(aws):
        yield await res


async def _to_async[T](it: Iterable[T]) -> AsyncIterable[T]:
    for item in it:
        yield item


def iterate_with_concurrency[T](
    iterables: Sequence[AsyncIterable[T]], max_concurrency: int
) -> AsyncIterator[T]:
    if not iterables:
        raise ValueError()
    streamer = flatten(_to_async(iterables), task_limit=max_concurrency)
    return streamer


async def _run_with_semaphore[T](
    aws: Awaitable[T] | asyncio.Future, sem: asyncio.Semaphore
) -> T:
    async with sem:
        return await aws


def batches[T](iterable: Iterable[T], batch_size: int) -> Iterator[T]:
    if batch_size < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, batch_size)):
        yield batch
