from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")

# 제한 시간을 넘긴 작업이 돌고 있는 동안에도 다음 요청을 받을 수 있도록 여유 스레드를 둔다 (S9-1)
_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="companion-timeout")


def call_with_timeout(fn: Callable[..., T], timeout: float, *args) -> T:
    """fn 을 제한 시간 안에 실행한다. 시간을 넘기면 TimeoutError 를 낸다.

    넘긴 작업은 바로 멈추지 못한다(파이썬 스레드 한계). 결과만 버리고 대화는 계속 진행한다.
    """
    future = _pool.submit(fn, *args)
    return future.result(timeout=timeout)
