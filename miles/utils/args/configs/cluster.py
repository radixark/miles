import argparse

from miles.utils.args.schema import A, Arg, BaseConfig
from miles.utils.workers.types import ClusterBackend, DeployComponent, WorkerCommBackend


# Ray
class ClusterConfig(BaseConfig):
    starts_inference_engines: bool
    rollout_external: bool

    cluster_backend: A[
        str,
        Arg(
            choices=tuple(backend.value for backend in ClusterBackend),
            help=(
                "Which backend provides the worker processes: "
                "`ray` launches them from the driver, `kubernetes` expects the platform to have "
                "created them already and observes them by their pod labels."
            ),
        ),
    ] = ClusterBackend.RAY.value
    worker_comm_backend: A[
        str | None,
        Arg(
            choices=tuple(backend.value for backend in WorkerCommBackend),
            help=(
                "How the driver calls its workers: `ray` sends actor calls, `rpc` calls the http server "
                "every worker serves. Unset picks the default of the cluster backend, today `ray` under "
                "`--cluster-backend ray` and `rpc` under `--cluster-backend kubernetes`."
            ),
        ),
    ] = None
    deploy_component: A[
        str,
        Arg(
            choices=tuple(component.value for component in DeployComponent),
            help=(
                "Which part of the run this launch deploys: `all` deploys every worker, `trainer` the trainer "
                "controllers and their megatron ranks, `inference` a group of inference engines that registers "
                "itself into the run, and `primary` everything else (orchestration script, rollout executor, "
                "session servers, inference controller and routers). Deploying a subset takes one launch per "
                "subset, and the launch that carries the orchestration script reaches the trainer through the "
                "addresses it is given."
            ),
        ),
    ] = DeployComponent.ALL.value
    deploy_instance_id: A[
        str | None,
        Arg(
            help=(
                "Id of this deployment, telling it apart from the other deployments of the same component "
                "in the same run: a trainer id such as `trainer-a` under `--deploy-component trainer`, or an "
                "engine group id such as `inf-east` under `--deploy-component inference`. A deployment's "
                "arguments describe only what it carries, so this id selects nothing; it is required under "
                "`--deploy-component inference`, which names its engine pools by it, optional under "
                "`--deploy-component trainer`, whose config already declares the one trainer id it carries, and "
                "refused for `all` and `primary`, which a run has exactly one of."
            )
        ),
    ] = None
    init_expected_num_cells: A[
        int | dict[str, int] | None,
        Arg(
            type_parser=int,
            help=(
                "How many engine cells per model this run waits for before it starts, when the engines are "
                "deployed elsewhere and register themselves into it. The run cannot derive the number, because "
                "the engine deployments are launched separately and may arrive late; declare here how many "
                "cells the first rollout needs. It gates startup only, and the run keeps serving whatever "
                "registers or leaves afterwards."
            ),
        ),
    ] = None
    trainer_controller_addrs: A[
        list[str] | None,
        Arg(
            type_parser=str,
            nargs="+",
            help=(
                "Address of every independently deployed trainer controller, one "
                "<trainer_id>=<host:port> entry per trainer the run drives. Required when this launch "
                "carries the orchestration script but not the trainer."
            ),
        ),
    ] = None
    inference_controller_addr: A[
        str | None,
        Arg(
            help=(
                "Address of the one inference controller of the run, as host:port. Given "
                "to a `--deploy-component inference` launch, whose reporter registers the engines it deploys "
                "into that controller."
            )
        ),
    ] = None
    actor_num_nodes: A[int, Arg(help="Number of nodes for training actor")] = 1
    actor_num_gpus_per_node: A[int, Arg(help="Number of gpus per node for training actor")] = 8
    critic_num_nodes: A[int | None, Arg(help="Number of nodes for training actor")] = None
    critic_num_gpus_per_node: A[int | None, Arg(help="Number of gpus per node for training actor")] = None
    rollout_num_gpus: A[
        int | None,
        Arg(
            help=(
                "Number of GPUs for inference. Note that when using --colocate, "
                "i.e. the training and the inference engines are on the same gpus, this param will be ignored and will be set as "
                "actor_num_gpus_per_node * actor_num_nodes."
            )
        ),
    ] = None
    rollout_num_gpus_per_engine: A[
        int, Arg(help="Number of GPUs per inference engine, just like the tp_size in sglang.")
    ] = 1
    num_gpus_per_node: A[
        int,
        Arg(
            help=(
                "Number of gpus per node for rollout."
                "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
            )
        ),
    ] = 8
    colocate: A[
        bool,
        Arg(
            help=(
                "Whether to colocate the inference engines and the actor. "
                "Turning this on will also set --offload to true."
            )
        ),
    ] = False
    offload: A[bool, Arg(help="Equivalent to --offload-train + --offload-rollout. ")] = False
    offload_train: A[
        bool | None,
        Arg(
            action=argparse.BooleanOptionalAction,
            help=(
                "Whether to offload the training actor to CPU while the rollout engines generate. "
                "Defaults to true when --colocate is set; an explicit --no-offload-train is respected."
            ),
        ),
    ] = None
    clear_quantized_weight_workspaces_on_offload: A[
        bool,
        Arg(
            action=argparse.BooleanOptionalAction,
            help=(
                "Drop TransformerEngine's cached quantized weights before offloading the "
                "training actor. They are rebuilt on the next forward, so backing them up "
                "to pinned host memory is pure overhead. Ignored when TransformerEngine "
                "is not in use or CUDA graphs are enabled."
            ),
        ),
    ] = True
    offload_rollout: A[
        bool | None,
        Arg(
            action=argparse.BooleanOptionalAction,
            help=(
                "Whether to offload the rollout generator to CPU during training. "
                "Defaults to true when --colocate is set; an explicit --no-offload-rollout is respected."
            ),
        ),
    ] = None
    offload_rollout_level: A[
        list[str],
        Arg(
            type_parser=str,
            nargs="+",
            help=(
                "Specifies what to offload during rollout when offload-rollout is set. "
                "Possible values: 'kv_cache', 'weight'. Default: both 'kv_cache' and 'weight'. "
                "Example: --offload-rollout-level kv_cache weight"
            ),
        ),
    ] = ["kv_cache", "weight"]
    offload_train_target: A[
        str,
        Arg(
            choices=["cpu", "disk"],
            help=(
                "Where the training actor is backed up while offloaded during rollout "
                "(only used with --offload-train on the megatron backend). "
                "'cpu' (default) keeps a pinned host copy; 'disk' streams it to node-local "
                "NVMe (--offload-train-disk-dir) for the case where even CPU RAM cannot hold it."
            ),
        ),
    ] = "cpu"
    stream_optimizer_state_to_disk: A[
        bool,
        Arg(
            help=(
                "Hold optimizer state in files on node-local NVMe, for when it does not fit the "
                "GPU *while the step runs*; --offload-train-target=disk cannot help there.\n"
                "adam: streams fp32 main params and moments through per-bucket files, one bucket "
                "resident at a time. Requires the distributed optimizer, excludes "
                "--offload-optimizer-states and --optimizer-cpu-offload.\n"
                "dist_muon: the disk backend for --chunked-optimizer-state-offload, so pass that "
                "plus a non-zero --optimizer-state-offload-fraction. --optimizer-cpu-offload is "
                "Adam-only. This bounds host residency, not the GPU restore window -- for that "
                "set --optimizer-state-offload-chunk-size-mb, which Megatron warns about at 0."
            )
        ),
    ] = False
    stream_optimizer_state_moment_dtype: A[
        str,
        Arg(
            choices=["fp32", "bf16", "fp16", "fp8e4m3", "fp8e5m2"],
            help=(
                "On-disk dtype for the streamed Adam moments; the fp32 master copy is always "
                "fp32. This is a serialization format, not a compute precision: the step still "
                "hands fp32 tensors to FusedAdam, and the cast happens on the way to and from "
                "disk. That makes it distinct from --exp-avg-dtype / --exp-avg-sq-dtype, which "
                "change what the optimizer holds and require --use-precision-aware-optimizer, "
                "so they cannot be combined with streaming at all. "
                "The step is I/O bound and the moments tolerate less precision than the master "
                "copy, so bf16 cuts streaming volume by a third (12 bytes per param to 8). "
                "fp32 is bit-identical to keeping the moments on GPU. The fp8 options need "
                "per-block scaling for exp_avg_sq to be sound, which this does not implement, "
                "and are not recommended."
            ),
        ),
    ] = "fp32"
    offload_train_disk_dir: A[
        str | None,
        Arg(
            help=(
                "Node-local directory for the disk-offload files, used by both "
                "--offload-train-target=disk and --stream-optimizer-state-to-disk (each under "
                "its own subdirectory). Should be fast local NVMe (e.g. /scratch); a tmpfs "
                "mount, which /tmp is on many systems, keeps the data in RAM and defeats both. "
                "Files are per-process and overwritten in place every step (bounded size); "
                "defaults to $SCRATCH/miles_train_offload_<uid>. Muon's optimizer-state buffers "
                "are unlinked once mapped, so their footprint shows in df but not du."
            )
        ),
    ] = None
    offload_train_disk_chunk_mb: A[
        int,
        Arg(
            help=(
                "Chunk size (MiB) for the GPU<->disk transfers, i.e. the pinned host staging "
                "buffer, which bounds host memory regardless of how much is moved. Used by both "
                "--offload-train-target=disk and --stream-optimizer-state-to-disk, and each "
                "allocates its own, so enabling both costs 2x this per rank."
            )
        ),
    ] = 256
    colocate_memory_peak_device: A[
        str,
        Arg(
            choices=["cpu", "gpu"],
            help=(
                "Which device absorbs the trainer<->rollout handoff overlap. 'cpu' "
                "(default): each side offloads before the other onloads, so the "
                "engine's weight mirror and the trainer's backup briefly coexist in "
                "host memory. 'gpu': onload the other side first, so both sides "
                "briefly coexist in GPU memory instead and the two host copies never "
                "overlap. Use 'gpu' when host RAM is the tighter budget than the "
                "handoff headroom on the GPU."
            ),
        ),
    ] = "cpu"
