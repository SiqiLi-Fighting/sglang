# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for Prometheus Metrics Collection."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

from sglang.srt.utils import get_bool_env_var
import torch

import logging
logger = logging.getLogger(__name__)

SGLANG_TEST_REQUEST_TIME_STATS = get_bool_env_var("SGLANG_TEST_REQUEST_TIME_STATS")


@dataclass
class TimeStats:
    """
    Store the timestamps for each stage of a request.

    Unified: wait_queue -> forward -> completion
    Prefill: bootstrap_queue -> wait_queue -> forward -> transfer_queue -> completion
    Decode: prealloc_queue -> transfer_queue -> wait_queue -> forward -> completion
    """

    lb_entry_time: float = 0.0
    wait_queue_entry_time: float = 0.0
    forward_entry_time: float = 0.0
    completion_time: float = 0.0
    prefill_bootstrap_queue_entry_time: float = 0.0
    prefill_transfer_queue_entry_time: float = 0.0
    decode_prealloc_queue_entry_time: float = 0.0
    decode_transfer_queue_entry_time: float = 0.0

    class RequestType(Enum):
        UNIFIED = "unified"
        PREFILL = "prefill"
        DECODE = "decode"
        INVALID = "invalid"

    def __str__(self) -> str:
        # if unified
        _type = self.get_type()

        if _type == self.RequestType.UNIFIED:
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    queue_duration >= 0 and forward_duration >= 0
                ), f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.wait_queue_entry_time}"
        elif _type == self.RequestType.PREFILL:
            bootstrap_duration = (
                self.wait_queue_entry_time - self.prefill_bootstrap_queue_entry_time
            )

            queue_duration = self.forward_entry_time - self.wait_queue_entry_time

            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    bootstrap_duration >= 0
                    and queue_duration >= 0
                    and forward_duration >= 0
                ), f"bootstrap_duration={bootstrap_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"
            return f"bootstrap_duration={self.format_duration(bootstrap_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.prefill_bootstrap_queue_entry_time}"
        # if decode
        elif _type == self.RequestType.DECODE:
            prealloc_duration = (
                self.decode_transfer_queue_entry_time
                - self.decode_prealloc_queue_entry_time
            )

            transfer_duration = (
                self.wait_queue_entry_time - self.decode_transfer_queue_entry_time
            )
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    prealloc_duration >= 0
                    and transfer_duration >= 0
                    and queue_duration >= 0
                    and forward_duration >= 0
                ), f"prealloc_duration={prealloc_duration} < 0 or transfer_duration={transfer_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"prealloc_duration={self.format_duration(prealloc_duration)}, transfer_duration={self.format_duration(transfer_duration)}, queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.decode_prealloc_queue_entry_time}"
        else:
            return "Invalid Time Stats"

    def format_duration(self, duration: float) -> str:
        return f"{duration * 1e3:.2f}ms"

    def get_type(self) -> RequestType:
        """Determine the type of request based on timestamp values."""
        if (
            self.prefill_bootstrap_queue_entry_time == 0.0
            and self.prefill_transfer_queue_entry_time == 0.0
            and self.decode_prealloc_queue_entry_time == 0.0
            and self.decode_transfer_queue_entry_time == 0.0
        ):
            return self.RequestType.UNIFIED
        elif (
            self.prefill_bootstrap_queue_entry_time > 0.0
            and self.prefill_transfer_queue_entry_time > 0.0
        ):
            return self.RequestType.PREFILL
        elif (
            self.decode_prealloc_queue_entry_time > 0.0
            and self.decode_transfer_queue_entry_time > 0.0
            and self.wait_queue_entry_time > 0.0
        ):
            return self.RequestType.DECODE
        else:
            return self.RequestType.INVALID


@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    num_used_tokens: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0
    cache_hit_rate: float = 0.0
    num_grammar_queue_reqs: int = 0
    spec_accept_length: float = 0.0
    avg_request_queue_latency: float = 0.0
    input_throughput_schedule_time: float = 0.0
    input_throughput_run_time: float = 0.0
    num_prefill_prealloc_queue_reqs: int = 0
    num_prefill_infight_queue_reqs: int = 0
    num_decode_prealloc_queue_reqs: int = 0
    num_decode_transfer_queue_reqs: int = 0


class SchedulerMetricsCollector:

    def __init__(self, tp_rank: int, labels: Dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Gauge

        self.tp_rank = tp_rank
        self.labels = labels
        self.last_log_time = time.perf_counter()

        labelnames_dp = list(labels.keys())
        labelnames_dp.append("dp")

        self.num_running_reqs = Gauge(
            name="sglang:num_running_reqs",
            documentation="The number of running requests.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.num_used_tokens = Gauge(
            name="sglang:num_used_tokens",
            documentation="The number of used tokens.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.token_usage = Gauge(
            name="sglang:token_usage",
            documentation="The token usage.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.gen_throughput = Gauge(
            name="sglang:gen_throughput",
            documentation="The generation throughput (token/s).",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.num_queue_reqs = Gauge(
            name="sglang:num_queue_reqs",
            documentation="The number of requests in the waiting queue.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.num_grammar_queue_reqs = Gauge(
            name="sglang:num_grammar_queue_reqs",
            documentation="The number of requests in the grammar waiting queue.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="The prefix cache hit rate.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.spec_accept_length = Gauge(
            name="sglang:spec_accept_length",
            documentation="The average acceptance length of speculative decoding.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.avg_request_queue_latency = Gauge(
            name="sglang:avg_request_queue_latency",
            documentation="The average request queue latency for the last batch of requests in seconds.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.input_throughput_schedule_time = Gauge(
            name="sglang:input_throughput_schedule_time",
            documentation="The input throughput in schedule time.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.input_throughput_run_time = Gauge(
            name="sglang:input_throughput_run_time",
            documentation="The input throughput in run time.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        # Disaggregation queue metrics
        self.num_prefill_prealloc_queue_reqs = Gauge(
            name="sglang:num_prefill_prealloc_queue_reqs",
            documentation="The number of requests in the prefill prealloc queue.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.num_prefill_infight_queue_reqs = Gauge(
            name="sglang:num_prefill_infight_queue_reqs",
            documentation="The number of requests in the prefill infight queue.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.num_decode_prealloc_queue_reqs = Gauge(
            name="sglang:num_decode_prealloc_queue_reqs",
            documentation="The number of requests in the decode prealloc queue.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.num_decode_transfer_queue_reqs = Gauge(
            name="sglang:num_decode_transfer_queue_reqs",
            documentation="The number of requests in the decode transfer queue.",
            labelnames=labelnames_dp,
            multiprocess_mode="mostrecent",
        )

        self.num_bootstrap_failed_reqs = Counter(
            name="sglang:num_bootstrap_failed_reqs",
            documentation="The number of bootstrap failed requests.",
            labelnames=labels.keys(),
        )

        self.num_transfer_failed_reqs = Counter(
            name="sglang:num_transfer_failed_reqs",
            documentation="The number of transfer failed requests.",
            labelnames=labels.keys(),
        )

    def _log_gauge_with_dp(self, gauge, data: Union[int, float], dp: int) -> None:
        labels = self.labels.copy()
        labels["dp"] = str(dp)
        gauge.labels(**labels).set(data)

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def increment_bootstrap_failed_reqs(self) -> None:
        self.num_bootstrap_failed_reqs.labels(**self.labels).inc(1)

    def increment_transfer_failed_reqs(self) -> None:
        self.num_transfer_failed_reqs.labels(**self.labels).inc(1)

    # TODO: refactor all gather
    def gather_stats(self, stats: SchedulerStats, dp_size: int, attn_tp_rank: int, attn_tp_size: int, tp_cpu_group) -> List[SchedulerStats]:
        if attn_tp_rank != 0:
            local_info = torch.zeros(15, dtype=torch.float32)
        else:
            local_info = self._stats_to_tensor(stats)
            if local_info.size(0) != 15:
                raise ValueError(f"local_info.size(0) != 15: {local_info.size(0)}")
        global_info = torch.empty(
            (dp_size, attn_tp_size, 15),
            dtype=torch.float32
        )
        torch.distributed.all_gather_into_tensor(
            global_info.flatten(),
            local_info,
            group=tp_cpu_group,
        )
        res = [
            SchedulerStats(
                num_running_reqs=int(global_info[i][0][0]),
                num_used_tokens=int(global_info[i][0][1]),
                token_usage=global_info[i][0][2],
                num_queue_reqs=int(global_info[i][0][3]),
                cache_hit_rate=global_info[i][0][4],
                avg_request_queue_latency=global_info[i][0][5],
                gen_throughput=global_info[i][0][6],
                num_grammar_queue_reqs=int(global_info[i][0][7]),
                spec_accept_length=global_info[i][0][8],
                input_throughput_schedule_time=global_info[i][0][9],
                input_throughput_run_time=global_info[i][0][10],
                num_prefill_prealloc_queue_reqs=int(global_info[i][0][11]),
                num_prefill_infight_queue_reqs=int(global_info[i][0][12]),
                num_decode_prealloc_queue_reqs=int(global_info[i][0][13]),
                num_decode_transfer_queue_reqs=int(global_info[i][0][14]),
            ) for i in range(dp_size)
        ]
        return res

    def _stats_to_tensor(self, stats: SchedulerStats) -> torch.Tensor:
        data = torch.zeros(15, dtype=torch.float32)
        data[0] = stats.num_running_reqs
        data[1] = stats.num_used_tokens
        data[2] = stats.token_usage
        data[3] = stats.num_queue_reqs
        data[4] = stats.cache_hit_rate
        data[5] = stats.avg_request_queue_latency
        data[6] = stats.gen_throughput
        data[7] = stats.num_grammar_queue_reqs
        data[8] = stats.spec_accept_length
        data[9] = stats.input_throughput_schedule_time
        data[10] = stats.input_throughput_run_time
        data[11] = stats.num_prefill_prealloc_queue_reqs
        data[12] = stats.num_prefill_infight_queue_reqs
        data[13] = stats.num_decode_prealloc_queue_reqs
        data[14] = stats.num_decode_transfer_queue_reqs
        return data

    def log_stats(self, stats: List[SchedulerStats]) -> None:
        for i, stat in enumerate(stats):
            self._log_gauge_with_dp(self.num_running_reqs, stat.num_running_reqs, i)
            self._log_gauge_with_dp(self.num_used_tokens, stat.num_used_tokens, i)
            self._log_gauge_with_dp(self.token_usage, stat.token_usage, i)
            self._log_gauge_with_dp(self.gen_throughput, stat.gen_throughput, i)
            self._log_gauge_with_dp(self.num_queue_reqs, stat.num_queue_reqs, i)
            self._log_gauge_with_dp(self.num_grammar_queue_reqs, stat.num_grammar_queue_reqs, i)
            self._log_gauge_with_dp(self.cache_hit_rate, stat.cache_hit_rate, i)
            self._log_gauge_with_dp(self.spec_accept_length, stat.spec_accept_length, i)
            self._log_gauge_with_dp(self.avg_request_queue_latency, stat.avg_request_queue_latency, i)
            self._log_gauge_with_dp(self.input_throughput_schedule_time, stat.input_throughput_schedule_time, i)
            self._log_gauge_with_dp(self.input_throughput_run_time, stat.input_throughput_run_time, i)
            # Disaggregation queue metrics
            self._log_gauge_with_dp(self.num_prefill_prealloc_queue_reqs, stat.num_prefill_prealloc_queue_reqs, i)
            self._log_gauge_with_dp(self.num_prefill_infight_queue_reqs, stat.num_prefill_infight_queue_reqs, i)
            self._log_gauge_with_dp(self.num_decode_prealloc_queue_reqs, stat.num_decode_prealloc_queue_reqs, i)
            self._log_gauge_with_dp(self.num_decode_transfer_queue_reqs, stat.num_decode_transfer_queue_reqs, i)

        self.last_log_time = time.perf_counter()


class TokenizerMetricsCollector:
    def __init__(
        self,
        labels: Dict[str, str],
        bucket_time_to_first_token: Optional[List[float]] = None,
        bucket_inter_token_latency: Optional[List[float]] = None,
        bucket_e2e_request_latency: Optional[List[float]] = None,
        collect_tokens_histogram: bool = False,
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Histogram

        self.labels = labels
        self.collect_tokens_histogram = collect_tokens_histogram

        self.prompt_tokens_total = Counter(
            name="sglang:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labels.keys(),
        )

        self.generation_tokens_total = Counter(
            name="sglang:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labels.keys(),
        )

        if collect_tokens_histogram:
            bucket_prompt_tokens = [
                100,
                300,
                500,
                700,
                1000,
                1500,
                2000,
                3000,
                4000,
                5000,
                6000,
                7000,
                8000,
                9000,
                10000,
                12000,
                15000,
                20000,
                22000,
                25000,
                30000,
                35000,
                40000,
            ]
            self.prompt_tokens_histogram = Histogram(
                name="sglang:prompt_tokens_histogram",
                documentation="Histogram of prompt token length.",
                labelnames=labels.keys(),
                buckets=bucket_prompt_tokens,
            )
            bucket_generation_tokens = [
                100,
                300,
                500,
                1000,
                1200,
                1500,
                1700,
                2000,
                2500,
                3000,
                3500,
                4000,
                4500,
                5000,
                6000,
                7000,
                8000,
                9000,
                10000,
            ]
            self.generation_tokens_histogram = Histogram(
                name="sglang:generation_tokens_histogram",
                documentation="Histogram of generation token length.",
                labelnames=labels.keys(),
                buckets=bucket_generation_tokens,
            )

        self.cached_tokens_total = Counter(
            name="sglang:cached_tokens_total",
            documentation="Number of cached prompt tokens.",
            labelnames=labels.keys(),
        )

        self.num_requests_total = Counter(
            name="sglang:num_requests_total",
            documentation="Number of requests processed.",
            labelnames=labels.keys(),
        )

        self.num_so_requests_total = Counter(
            name="sglang:num_so_requests_total",
            documentation="Number of structured output requests processed.",
            labelnames=labels.keys(),
        )

        if bucket_time_to_first_token is None:
            bucket_time_to_first_token = [
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                100,
                200,
                400,
            ]

        if bucket_e2e_request_latency is None:
            bucket_e2e_request_latency = [
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                100,
                200,
                400,
                800,
            ]

        if bucket_inter_token_latency is None:
            bucket_inter_token_latency = [
                0.002,
                0.004,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
                0.040,
                0.060,
                0.080,
                0.100,
                0.200,
                0.400,
                0.600,
                0.800,
                1.000,
                2.000,
                4.000,
                6.000,
                8.000,
            ]

        self.histogram_time_to_first_token = Histogram(
            name="sglang:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_time_to_first_token,
        )

        self.histogram_inter_token_latency_seconds = Histogram(
            name="sglang:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_inter_token_latency,
        )

        self.histogram_e2e_request_latency = Histogram(
            name="sglang:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labels.keys(),
            buckets=bucket_e2e_request_latency,
        )

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        histogram.labels(**self.labels).observe(data)

    def observe_one_finished_request(
        self,
        prompt_tokens: int,
        generation_tokens: int,
        cached_tokens: int,
        e2e_latency: float,
        has_grammar: bool,
    ):
        self.prompt_tokens_total.labels(**self.labels).inc(prompt_tokens)
        self.generation_tokens_total.labels(**self.labels).inc(generation_tokens)
        if cached_tokens > 0:
            self.cached_tokens_total.labels(**self.labels).inc(cached_tokens)
        self.num_requests_total.labels(**self.labels).inc(1)
        if has_grammar:
            self.num_so_requests_total.labels(**self.labels).inc(1)
        self._log_histogram(self.histogram_e2e_request_latency, e2e_latency)
        if self.collect_tokens_histogram:
            self._log_histogram(self.prompt_tokens_histogram, prompt_tokens)
            self._log_histogram(self.generation_tokens_histogram, generation_tokens)

    def observe_time_to_first_token(self, value: float):
        self.histogram_time_to_first_token.labels(**self.labels).observe(value)

    def observe_inter_token_latency(self, internval: float, num_new_tokens: int):
        adjusted_interval = internval / num_new_tokens

        # A faster version of the Histogram::observe which observes multiple values at the same time.
        # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
        his = self.histogram_inter_token_latency_seconds.labels(**self.labels)
        his._sum.inc(internval)

        for i, bound in enumerate(his._upper_bounds):
            if adjusted_interval <= bound:
                his._buckets[i].inc(num_new_tokens)
                break
