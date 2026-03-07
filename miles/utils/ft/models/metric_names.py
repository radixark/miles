# GPU metrics (DCGM Exporter convention, lowercased)
GPU_AVAILABLE = "miles_ft_gpu_available"
DCGM_FI_DEV_GPU_TEMP = "miles_ft_dcgm_fi_dev_gpu_temp"
DCGM_FI_DEV_ROW_REMAP_PENDING = "miles_ft_dcgm_fi_dev_row_remap_pending"
DCGM_FI_DEV_ROW_REMAP_FAILURE = "miles_ft_dcgm_fi_dev_row_remap_failure"
DCGM_FI_DEV_PCIE_TX_THROUGHPUT = "miles_ft_dcgm_fi_dev_pcie_tx_throughput"
DCGM_FI_DEV_GPU_UTIL = "miles_ft_dcgm_fi_dev_gpu_util"

# Network metrics (Node Exporter convention)
NODE_NETWORK_UP = "miles_ft_node_network_up"
NODE_NETWORK_RECEIVE_ERRS_TOTAL = "miles_ft_node_network_receive_errs_total"
NODE_NETWORK_TRANSMIT_ERRS_TOTAL = "miles_ft_node_network_transmit_errs_total"
NODE_NETWORK_RECEIVE_DROP_TOTAL = "miles_ft_node_network_receive_drop_total"
NODE_NETWORK_TRANSMIT_DROP_TOTAL = "miles_ft_node_network_transmit_drop_total"

# Filesystem / Disk metrics (Node Exporter convention)
NODE_FILESYSTEM_AVAIL_BYTES = "miles_ft_node_filesystem_avail_bytes"
NODE_DISK_IO_TIME_SECONDS_TOTAL = "miles_ft_node_disk_io_time_seconds_total"

# Host custom metrics
XID_CODE_RECENT = "miles_ft_xid_code_recent"
XID_COUNT_TOTAL = "miles_ft_xid_count_total"
XID_NON_AUTO_RECOVERABLE_COUNT_TOTAL = "miles_ft_xid_non_auto_recoverable_count_total"
KERNEL_EVENT_COUNT = "miles_ft_kernel_event_count"

# Training heartbeat (FtTrainingRankAgent)
TRAINING_ITERATION = "miles_ft_training_iteration"
AGENT_HEARTBEAT = "miles_ft_agent_heartbeat"
TRAINING_PHASE = "miles_ft_training_phase"

# Controller synthetic
TRAINING_JOB_STATUS = "miles_ft_training_job_status"
TRAINING_LOSS_LATEST = "miles_ft_training_loss_latest"
TRAINING_MFU_LATEST = "miles_ft_training_mfu_latest"

# Training phase numeric encoding (shared between FtTrainingRankAgent and detectors)
PHASE_IDLE: float = 0.0
PHASE_TRAINING: float = 1.0
PHASE_CHECKPOINT_SAVING: float = 2.0

PHASE_TO_NUMERIC: dict[str, float] = {
    "idle": PHASE_IDLE,
    "training": PHASE_TRAINING,
    "checkpoint_saving": PHASE_CHECKPOINT_SAVING,
}

# Controller operational
CONTROLLER_MODE = "miles_ft_controller_mode"
CONTROLLER_TICK_COUNT = "miles_ft_controller_tick_count"
CONTROLLER_RECOVERY_PHASE = "miles_ft_controller_recovery_phase"
CONTROLLER_TICK_DURATION_SECONDS = "miles_ft_controller_tick_duration_seconds"
CONTROLLER_DECISION_TOTAL = "miles_ft_controller_decision_total"
CONTROLLER_RECOVERY_DURATION_SECONDS = "miles_ft_controller_recovery_duration_seconds"
CONTROLLER_LAST_TICK_TIMESTAMP = "miles_ft_controller_last_tick_timestamp"
