# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embodied phyai rollout worker: drive a registered phyai action converter over
channels against a driver-launched ``phyai serve`` (no worker-owned HTTP
server, no in-worker subprocess).

Used by embodied phyai-convert-action models (e.g. DreamZero). The eval driver
launches the ``phyai serve`` server group via
:func:`launch_phyai_router_and_server` and pushes the server URLs to each
rollout worker via :meth:`set_phyai_server_urls`; the worker picks the URL
at its own rank (for N-server parallel throughput), loads the phyai action converter
registered for ``rollout.model.model_type``, and is driven by
``EmbodiedEvalRunner`` over channels (``recv_from``/``send_to``). It does NOT
host its own HTTP server (the agent path uses
:class:`phyaiAgentWorkerWithHTTPServer`).
"""

from typing import Any, Literal, Optional

import torch
from omegaconf import DictConfig

from rlinf.scheduler import Worker
from rlinf.utils.placement import HybridComponentPlacement
import numpy as np


class PhyaiEmbodiedWorker(Worker):
    """Use a driver-launched ``phyai serve`` + phyai action converter + channel eval."""

    def __init__(
        self,
        config: DictConfig,
        placement: HybridComponentPlacement,
        config_rollout: Optional[DictConfig] = None,
    ):
        Worker.__init__(self)
        self.cfg = config
        self.cfg_rollout = (
            config_rollout if config_rollout is not None else config.rollout
        )
        self.model_type = str(
            getattr(getattr(self.cfg_rollout, "model", None), "model_type", "")
        ).lower()
        self.model_cfg = self.cfg_rollout.model
        self.http_client = None
        self.phyai_server_url = None
        self._phyai_server_urls = None
        # This worker is eval-only (drives a serve + channel eval; no training).
        assert config.runner.get("only_eval", True), (
            "PhyaiEmbodiedWorker is eval-only; set runner.only_eval: true"
        )
        # Decoupled env/rollout is not implemented on the phyai embodied path.
        assert not config.runner.get("enable_decoupled_mode", False), (
            "PhyaiEmbodiedWorker does not support runner.enable_decoupled_mode"
        )
        eval_env_cfg = config.env.get("eval", None)
        self.num_pipeline_stages = int(config.rollout.pipeline_stage_num)
        total_eval = int(eval_env_cfg.total_num_envs) if eval_env_cfg else 0
        self.eval_batch_size = (
            total_eval // self.num_pipeline_stages
            if self.num_pipeline_stages
            else total_eval
        )
        self.eval_rollout_epoch = int(eval_env_cfg.rollout_epoch) if eval_env_cfg else 1
        if eval_env_cfg is not None:
            self.n_eval_chunk_steps = int(
                eval_env_cfg.max_steps_per_rollout_epoch
            ) // int(self.model_cfg.num_action_chunks)
        else:
            self.n_eval_chunk_steps = 0

    async def init_worker(self):
        self._init_phyai_server()
        from rlinf.utils.http_client import InferenceHTTPClient

        self.http_client = InferenceHTTPClient(self.phyai_server_url)
        phyai_cfg = self.cfg.rollout.get("phyai", {})
        self._http_timeout_s = float(
            phyai_cfg.get("http_timeout_s", phyai_cfg.get("timeout_s", 120.0))
        )
        self._http_max_retries = int(phyai_cfg.get("http_max_retries", 5))
        self._http_retry_backoff_s = float(phyai_cfg.get("http_retry_backoff_s", 1.0))

    def set_phyai_server_urls(self, urls) -> None:
        """Receive the phyai server URLs the driver launched."""
        self._phyai_server_urls = list(urls)

    def _init_phyai_server(self) -> None:
        """Pick the pre-launched phyai server URL assigned to this rank."""
        urls = self._phyai_server_urls
        if not urls:
            raise RuntimeError(
                "phyai server URLs not set; the eval driver must call "
                "rollout_group.set_phyai_server_urls(urls) (after "
                "launch_phyai_router_and_server) before init_workers()."
            )
        self.phyai_server_url = urls[int(self._rank) % len(urls)]
        self.log_info(
            f"phyai server assigned: rank={self._rank} -> "
            f"{self.phyai_server_url} ({len(urls)} server(s))"
        )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [b["obs"] if "obs" in b else b for b in obs_batches]
        merged: dict[str, Any] = {}
        for key in obs_dicts[0].keys():
            values = [d[key] for d in obs_dicts]
            first = next((v for v in values if v is not None), None)
            if first is None:
                merged[key] = None
            elif isinstance(first, torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(first, list):
                merged[key] = [item for sub in values for item in sub]
            else:
                merged[key] = values
        reset = any(b.get("final_obs") is not None for b in obs_batches)
        return {"obs": merged, "reset": reset}

    def build_phyai_request(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if mode != "eval":
            raise NotImplementedError("PhyAI embodied worker supports eval only.")
        routing_keys = {"_rlinf_stage_id", "_rlinf_reset"}
        # 保留 RLinf EnvWorker 输出的原始观测，不执行 DreamZero 预处理。
        observation = {
            key: value
            for key, value in env_obs.items()
            if key not in routing_keys
        }

        states=observation["states"]
        if torch.is_tensor(states):
            observation["states"]=states.to(dtype=torch.float32)
        else:
            import numpy as np
            observation["states"]=np.asarray(states,dtype=np.float32)

        batch_size = self._infer_env_batch_size(observation)
        stage_id = env_obs.get("_rlinf_stage_id", 0)
        if torch.is_tensor(stage_id):
            stage_id = stage_id.item()
        reset = env_obs.get("_rlinf_reset", False)
        if torch.is_tensor(reset):
            reset = bool(reset.item())
        else:
            reset = bool(reset)
        action_horizon = self.model_cfg.get("action_horizon", None)
        if action_horizon is None:
            action_horizon = self.model_cfg.get("num_action_chunks", 1)
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError(
                f"requested action horizon must be positive, got {action_horizon}"
            )
        metadata = {
            "mode": mode,
            "batch_size": batch_size,
            "stage_id": int(stage_id),
            "reset": reset,
        }
        payload = {
            # 这里的 Tensor / ndarray 由 InferenceHTTPClient 的 msgpack codec
            # 直接序列化，不要在 Worker 中提前转成 bytes。
            "observation": observation,
            "model_name":"pi05",
            "model":"pi05",
            "metadata": metadata,
            "requested_action_horizon": action_horizon,
        }
        state = {
            "batch_size": batch_size,
            "requested_action_horizon": action_horizon,
            "stage_id": int(stage_id),
        }
        return payload, state

    def parse_phyai_response(
      self,
      resp: dict[str, Any],
      state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        try:
            actions = torch.as_tensor(resp["actions"], dtype=torch.float32)
        except KeyError as exc:
            raise RuntimeError(
                f"PhyAI response missing actions: {resp}"
            ) from exc  
        expected_shape = (
            state["batch_size"],
            state["requested_action_horizon"],
        )
        if actions.ndim != 3 or tuple(actions.shape[:2]) != expected_shape:
            raise RuntimeError(
                f"PhyAI actions must have shape "
                f"[{expected_shape[0]},{expected_shape[1]},D], "
                f"got {tuple(actions.shape)}"
            )   
        flat = actions.reshape(actions.shape[0], -1)
        info = {
            "prev_logprobs": torch.zeros_like(flat),
            "prev_values": torch.zeros(
                (actions.shape[0], 1),
                dtype=torch.float32,
            ),
            "forward_inputs": {"action": flat.cpu()},
        }
        return actions, info

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """env_obs -> action chunks [N, num_action_chunks, action_dim].

        Owns the phyai HTTP round-trip: the adapter builds the request
        payload and parses the response; this worker performs the msgpack POST.
        """
        payload, state = self.build_phyai_request(env_obs, mode=mode)
        resp = self.http_client.post(
            "/v1/actions/generations",
            payload,
            msgpack=True,
            timeout_s=self._http_timeout_s,
            max_retries=self._http_max_retries,
            retry_backoff_s=self._http_retry_backoff_s,
        )
        return self.parse_phyai_response(resp, state)

    async def evaluate(self, input_channel, output_channel):
        """Channel-based embodied eval loop, driven by EmbodiedEvalRunner."""
        from tqdm import tqdm

        for _ in tqdm(
            range(self.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.n_eval_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_from(
                        group_name=self.cfg.env.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                        merge_fn=self._merge_obs_batches,
                        infer_batch_size_fn=self._infer_env_batch_size,
                    ).async_wait()
                    obs = {
                        **env_output["obs"],
                        "_rlinf_stage_id": stage_id,
                        "_rlinf_reset": env_output.get("reset", False),
                    }
                    actions, _ = self.predict(obs, mode="eval")
                    if isinstance(actions, torch.Tensor):
                        actions = actions.detach().cpu().contiguous()
                    self.send_to(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=actions,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                    )


class SGLang_pi05_EmbodiedWorker(Worker):
    """Use a driver-launched ``sglang serve`` + sglang action converter + channel eval."""

    def __init__(
        self,
        config: DictConfig,
        placement: HybridComponentPlacement,
        config_rollout: Optional[DictConfig] = None,
    ):
        Worker.__init__(self)
        self.cfg = config
        self.cfg_rollout = (
            config_rollout if config_rollout is not None else config.rollout
        )
        self.model_type = str(
            getattr(getattr(self.cfg_rollout, "model", None), "model_type", "")
        ).lower()
        self.model_cfg = self.cfg_rollout.model
        self.sglang_adapter = None
        self.http_client = None
        self.sglang_server_url = None
        self._sglang_server_urls = None
        self._request_executor = None
        # This worker is eval-only (drives a serve + channel eval; no training).
        assert config.runner.get("only_eval", True), (
            "PhyaiEmbodiedWorker is eval-only; set runner.only_eval: true"
        )
        # Decoupled env/rollout is not implemented on the sglang embodied path.
        assert not config.runner.get("enable_decoupled_mode", False), (
            "PhyaiEmbodiedWorker does not support runner.enable_decoupled_mode"
        )
        eval_env_cfg = config.env.get("eval", None)
        self.num_pipeline_stages = int(config.rollout.pipeline_stage_num)
        total_eval = int(eval_env_cfg.total_num_envs) if eval_env_cfg else 0
        self.eval_batch_size = (
            total_eval // self.num_pipeline_stages
            if self.num_pipeline_stages
            else total_eval
        )
        self.eval_rollout_epoch = int(eval_env_cfg.rollout_epoch) if eval_env_cfg else 1
        if eval_env_cfg is not None:
            self.n_eval_chunk_steps = int(
                eval_env_cfg.max_steps_per_rollout_epoch
            ) // int(self.model_cfg.num_action_chunks)
        else:
            self.n_eval_chunk_steps = 0

    async def init_worker(self):
        adapter_cls = None

        self.model_type="dreamzero"  # 没有openpi_rlinf的adapter ,但是该类也不需要使用这个adapter，所以通过mock的方式通过下面的检测
        if self.model_type:
            from rlinf.models.embodiment.sglang_adapter import (
                get_sglang_adapter_cls,
            )

            adapter_cls = get_sglang_adapter_cls(self.model_type)
        if adapter_cls is None:
            raise RuntimeError(
                f"no sglang adapter registered for model_type "
                f"'{self.model_type}'; cannot run the embodied sglang path"
            )
        self._init_sglang_server()
        from rlinf.utils.http_client import InferenceHTTPClient

        self.http_client = InferenceHTTPClient(self.sglang_server_url)
        sglang_cfg = self.cfg.rollout.get("sglang", {})
        self._http_timeout_s = float(
            sglang_cfg.get("http_timeout_s", sglang_cfg.get("timeout_s", 120.0))
        )
        self._http_max_retries = int(sglang_cfg.get("http_max_retries", 5))
        self._http_retry_backoff_s = float(sglang_cfg.get("http_retry_backoff_s", 1.0))
        from concurrent.futures import ThreadPoolExecutor

        request_concurrency = int(
            sglang_cfg.get("request_concurrency", self.eval_batch_size)
        )
        self._request_executor = ThreadPoolExecutor(
            max_workers=max(1, request_concurrency)
        )
        self.sglang_adapter = adapter_cls(self.cfg, self._rank)

    def set_sglang_server_urls(self, urls) -> None:
        """Receive the sglang server URLs the driver launched."""
        self._sglang_server_urls = list(urls)

    def _init_sglang_server(self) -> None:
        """Pick the pre-launched sglang server URL assigned to this rank."""
        urls = self._sglang_server_urls
        if not urls:
            raise RuntimeError(
                "sglang server URLs not set; the eval driver must call "
                "rollout_group.set_phyai_server_urls(urls) (after "
                "launch_sglang_router_and_server) before init_workers()."
            )
        self.sglang_server_url = urls[int(self._rank) % len(urls)]
        self.log_info(
            f"sglang server assigned: rank={self._rank} -> "
            f"{self.sglang_server_url} ({len(urls)} server(s))"
        )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [b["obs"] if "obs" in b else b for b in obs_batches]
        merged: dict[str, Any] = {}
        for key in obs_dicts[0].keys():
            values = [d[key] for d in obs_dicts]
            first = next((v for v in values if v is not None), None)
            if first is None:
                merged[key] = None
            elif isinstance(first, torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(first, list):
                merged[key] = [item for sub in values for item in sub]
            else:
                merged[key] = values
        reset = any(b.get("final_obs") is not None for b in obs_batches)
        return {"obs": merged, "reset": reset}

    def build_SGLang_pi05_request(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if mode != "eval":
            raise NotImplementedError(
                "SGLang PI0.5 path supports eval only."
            )

        obs = env_obs["obs"] if "obs" in env_obs else env_obs
        def resize_for_dreamzero(value: Any, name: str) -> np.ndarray:
            arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if arr.ndim == 5 and arr.shape[1] == 1 and arr.shape[-1] == 3:
                arr = arr[:, 0]
            if arr.ndim != 4 or arr.shape[-1] != 3:
                raise ValueError(f"{name} must have shape [B,H,W,3] or [B,1,H,W,3], got {arr.shape}")
            if arr.shape[1:3] != (256, 256):
                tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float()
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(256, 256), mode="bilinear", align_corners=False
                )
                arr = tensor.permute(0, 2, 3, 1).round().clamp(0, 255).byte().numpy()
            return np.ascontiguousarray(arr)

        adapter_obs = dict(obs)
        adapter_obs["main_images"] = resize_for_dreamzero(obs["main_images"], "main_images")
        adapter_obs["wrist_images"] = resize_for_dreamzero(obs["wrist_images"], "wrist_images")
        adapter_env_obs = dict(env_obs)
        if "obs" in adapter_env_obs:
            adapter_env_obs["obs"] = adapter_obs
        else:
            adapter_env_obs = adapter_obs
        normalized_payload, converted_obs = self.sglang_adapter.build_request(
            adapter_env_obs, mode=mode
        )
        raw_states = obs["states"]
        if torch.is_tensor(raw_states):
            raw_states = raw_states.detach().cpu().numpy()
        else:
            raw_states = np.asarray(raw_states)
        if raw_states.ndim == 3 and raw_states.shape[1] == 1:
            raw_states = raw_states[:, 0]
        if raw_states.ndim != 2 or raw_states.shape[1] != 8:
            raise ValueError(
                f"raw state must have shape [B,8], got {raw_states.shape}"
            )
        raw_states = np.ascontiguousarray(raw_states, dtype=np.float32)
        batch_size = raw_states.shape[0]

        main_images = converted_obs.get("video.image", adapter_obs["main_images"])
        wrist_images = converted_obs.get(
            "video.wrist_image", adapter_obs["wrist_images"]
        )
        tasks = obs["task_descriptions"]
        if len(tasks) != batch_size:
            raise ValueError(
                f"task_descriptions length {len(tasks)} does not match batch size {batch_size}"
            )

        def to_hwc_uint8(value: Any, name: str) -> np.ndarray:
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            if arr.ndim == 5 and arr.shape[1] == 1:
                arr = arr[:, 0]
            if arr.ndim != 4 or arr.shape[0] != batch_size or arr.shape[-1] != 3:
                raise ValueError(
                    f"{name} must have shape [B,H,W,3] or [B,1,H,W,3], got {arr.shape}"
                )
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(arr)

        main_images_np = to_hwc_uint8(main_images, "main_images")
        wrist_images_np = to_hwc_uint8(wrist_images, "wrist_images")
        action_horizon = int(self.model_cfg.get("action_horizon", 50))
        num_inference_steps = int(
            self.model_cfg.get("num_inference_steps", 10)
        )
        payloads = []
        for sample_index in range(batch_size):
            payloads.append(
                {
                    "model": "pi05",
                    "input": {
                        "task": str(tasks[sample_index]).strip(),
                        "observation": {
                            "images": {
                                "image": main_images_np[sample_index],
                                "image2": wrist_images_np[sample_index],
                            },
                            "state": raw_states[sample_index],
                        },
                    },
                    "parameters": {
                        "action_horizon": action_horizon,
                        "action_dim": 32,
                        "num_inference_steps": num_inference_steps,
                    },
                    "runtime": {
                        "response_format": "envelope",
                        "output_format": "numpy",
                        "return_timing": True,
                        "cuda_graph": True,
                    },
                }
            )

        return payloads, {
            "batch_size": batch_size,
            "requested_action_horizon": action_horizon,
            "action_dim": 7,
            "normalized_payload": normalized_payload,
            "converted_obs": converted_obs,
        }

    def parse_SGLang_pi05_response(
        self,
        responses: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        batch_size = state["batch_size"]
        horizon = state["requested_action_horizon"]
        action_dim = state["action_dim"]
        if len(responses) != batch_size:
            raise RuntimeError(
                f"SGLang returned {len(responses)} responses, expected {batch_size}"
            )

        sample_actions = []
        for sample_index, response in enumerate(responses):
            try:
                action_info = response["data"][0]["action"]
                actions = torch.as_tensor(
                    action_info["values"], dtype=torch.float32
                )
                declared_shape = tuple(action_info["shape"])
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"Invalid SGLang PI0.5 response for sample {sample_index}: {response}"
                ) from exc
            if tuple(actions.shape) != declared_shape or actions.ndim != 2:
                raise RuntimeError(
                    f"SGLang sample {sample_index} returned invalid action shape "
                    f"values={tuple(actions.shape)}, declared={declared_shape}"
                )
            if actions.shape[0] < horizon or actions.shape[1] != action_dim:
                raise RuntimeError(
                    f"SGLang sample {sample_index} returned {tuple(actions.shape)}, "
                    f"expected [{horizon},{action_dim}]"
                )
            actions = actions[:horizon]
            if not torch.isfinite(actions).all():
                raise RuntimeError(
                    f"SGLang sample {sample_index} returned non-finite actions"
                )
            sample_actions.append(actions)

        normalized_actions = torch.stack(sample_actions, dim=0).contiguous()
        padded_actions = torch.nn.functional.pad(
            normalized_actions, (0, 32 - action_dim)
        )
        response = {
            "data": [
                {"action": {"values": padded_actions.numpy().tolist()}}
            ]
        }
        actions, info = self.sglang_adapter.parse_response(
            response, state["converted_obs"]
        )
        actions = actions.contiguous().cpu()
        if tuple(actions.shape) != (batch_size, horizon, action_dim):
            raise RuntimeError(
                f"unnormalized SGLang actions have unexpected shape {tuple(actions.shape)}"
            )
        return actions, info

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        adapter = self.sglang_adapter
        payloads, state = self.build_SGLang_pi05_request(env_obs, mode=mode)

        def post_request(payload):
            return self.http_client.post(
                adapter.action_path,
                payload,
                msgpack=True,
                timeout_s=self._http_timeout_s,
                max_retries=self._http_max_retries,
                retry_backoff_s=self._http_retry_backoff_s,
            )

        responses = list(self._request_executor.map(post_request, payloads))
        return self.parse_SGLang_pi05_response(responses, state)

    async def evaluate(self, input_channel, output_channel):
        """Channel-based embodied eval loop, driven by EmbodiedEvalRunner."""
        from tqdm import tqdm

        for _ in tqdm(
            range(self.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.n_eval_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_from(
                        group_name=self.cfg.env.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                        merge_fn=self._merge_obs_batches,
                        infer_batch_size_fn=self._infer_env_batch_size,
                    ).async_wait()
                    obs = {
                        **env_output["obs"],
                        "_rlinf_stage_id": stage_id,
                        "_rlinf_reset": env_output.get("reset", False),
                    }
                    actions, _ = self.predict(obs, mode="eval")
                    if isinstance(actions, torch.Tensor):
                        actions = actions.detach().cpu().contiguous()
                    self.send_to(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=actions,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                    )
