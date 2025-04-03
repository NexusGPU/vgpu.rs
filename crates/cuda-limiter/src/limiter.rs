use cudarc::driver::{sys::CUdevice_attribute, CudaDevice, DriverError};
use nvml_wrapper::{enums::device::UsedGpuMemory, error::NvmlError, Nvml};
use std::{
    sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering},
    thread::sleep,
    time::{Duration, SystemTime},
};
use thiserror::Error;

use crate::detour::NvmlDeviceT;

const FACTOR: u32 = 32;

#[derive(Error, Debug)]
pub(crate) enum Error {
    #[error("nvmlError: `{0}`")]
    NvmlError(NvmlError),

    #[error("cuDriverError: `{0}`")]
    CuDriverError(DriverError),
}

impl From<NvmlError> for Error {
    fn from(err: NvmlError) -> Self {
        Self::NvmlError(err)
    }
}

impl From<DriverError> for Error {
    fn from(err: DriverError) -> Self {
        Self::CuDriverError(err)
    }
}

#[derive(Debug)]
pub(crate) struct Limiter {
    nvml: Nvml,
    device_idx: u32,
    pid: u32,
    sm_count: u32,
    max_thread_per_sm: u32,
    total_cuda_cores: u32,
    available_cuda_cores: AtomicI32,

    // percentage
    up_limit: AtomicU32,
    // bytes
    mem_limit: AtomicU64,
    // set by cuFuncSetBlockShape
    pub block_x: AtomicU32,
    pub block_y: AtomicU32,
    pub block_z: AtomicU32,
}

#[derive(Debug, Default)]
struct Utilization {
    user_current: u32,
    sys_current: u32,
    sys_process_num: u32,
}

impl Limiter {
    pub(crate) fn init(
        pid: u32,
        device_idx: u32,
        up_limit: u32,
        mem_limit: u64,
    ) -> Result<Self, Error> {
        let nvml = match Nvml::init() {
            Ok(nvml) => Ok(nvml),
            Err(_) => Nvml::builder()
                .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
                .init(),
        }?;
        let dev = CudaDevice::new(device_idx as usize)?;

        let sm_count =
            dev.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        let max_thread_per_sm = dev
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?
            as u32;

        let total_cuda_cores = sm_count * max_thread_per_sm * FACTOR;

        tracing::trace!("sm_count: {sm_count}, max_thread_per_sm: {max_thread_per_sm}, mem_limit: {mem_limit} bytes");

        Ok(Self {
            nvml,
            device_idx,
            pid,
            sm_count,
            max_thread_per_sm,
            total_cuda_cores,
            available_cuda_cores: AtomicI32::new(0),
            up_limit: AtomicU32::new(up_limit),
            mem_limit: AtomicU64::new(mem_limit),
            block_x: AtomicU32::new(0),
            block_y: AtomicU32::new(0),
            block_z: AtomicU32::new(0),
        })
    }

    pub(crate) fn set_uplimit(&self, up_limit: u32) {
        self.up_limit.store(up_limit, Ordering::Release);
    }

    pub(crate) fn set_mem_limit(&self, mem_limit: u64) {
        self.mem_limit.store(mem_limit, Ordering::Release);
    }

    pub(crate) fn rate_limiter(&self, grids: u32, _blocks: u32) {
        let kernel_size = grids;
        loop {
            if self.available_cuda_cores.load(Ordering::Acquire) > 0 {
                break;
            }
            sleep(Duration::from_millis(10));
        }
        self.available_cuda_cores
            .fetch_sub(kernel_size as i32, Ordering::Release);
    }

    pub(crate) fn get_used_gpu_memory(&self) -> Result<u64, NvmlError> {
        match self
            .nvml
            .device_by_index(self.device_idx)
            .and_then(|dev| dev.running_compute_processes())
        {
            Ok(process_info) => {
                for pi in process_info {
                    if pi.pid == self.pid {
                        match pi.used_gpu_memory {
                            UsedGpuMemory::Unavailable => {
                                tracing::warn!(
                                    "failed to get used memory, err: NVML_VALUE_NOT_AVAILABLE"
                                );
                                break;
                            }
                            UsedGpuMemory::Used(bytes) => return Ok(bytes),
                        }
                    }
                }
                Ok(0)
            }
            Err(err) => {
                tracing::warn!("Failed to get running compute processes, err: {:?}", err);
                Err(NvmlError::Unknown)
            }
        }
    }

    pub(crate) fn run_watcher(&self, watch_duration: Duration) {
        let mut share: i32 = 0;
        loop {
            sleep(watch_duration);
            let util = loop {
                match self.get_used_gpu_utilization() {
                    Ok(Some(util)) => break util,
                    Ok(None) => {
                        continue;
                    }
                    Err(err) => {
                        tracing::trace!("failed to get_used_gpu_utilization, err: {err}");
                        sleep(watch_duration * 2);
                    }
                }
            };

            let available_cuda_cores = self.available_cuda_cores.load(Ordering::Acquire);

            share = self.delta(util.user_current, share);

            if available_cuda_cores + share >= self.total_cuda_cores as i32 {
                self.available_cuda_cores
                    .store(self.total_cuda_cores as i32, Ordering::Release);
            } else if available_cuda_cores + share < 0 {
                self.available_cuda_cores.store(0, Ordering::Release);
                share = 0;
            } else {
                self.available_cuda_cores
                    .fetch_add(share, Ordering::Release);
            }
        }
    }

    pub(crate) fn get_mem_limit(&self) -> u64 {
        let mem_limit = self.mem_limit.load(Ordering::Acquire);
        if mem_limit == 0 {
            u64::MAX
        } else {
            mem_limit
        }
    }

    pub(crate) fn device_handle(&self) -> Result<NvmlDeviceT, NvmlError> {
        self.nvml
            .device_by_index(self.device_idx)
            .map(|dev| unsafe { dev.handle() as NvmlDeviceT })
    }

    fn delta(&self, user_current: u32, share: i32) -> i32 {
        let up_limit = self.up_limit.load(Ordering::Acquire) as i32;

        let utilization_diff = if (up_limit - user_current as i32).abs() < 5 {
            5
        } else {
            (up_limit - user_current as i32).abs()
        };

        let mut increment =
            self.sm_count as i32 * self.sm_count as i32 * self.max_thread_per_sm as i32 / 256
                * utilization_diff
                / 10;

        if utilization_diff > up_limit / 2 {
            increment = increment * utilization_diff * 2 / (up_limit + 1);
        }

        if user_current <= up_limit as u32 {
            let total_cuda_cores = self.total_cuda_cores as i32;
            if share + increment > total_cuda_cores {
                total_cuda_cores
            } else {
                share + increment
            }
        } else {
            share - increment
            // if share - increment < 0 {
            //     0
            // } else {
            //     share - increment
            // }
        }
    }

    fn get_used_gpu_utilization(&self) -> Result<Option<Utilization>, NvmlError> {
        let dev = self.nvml.device_by_index(self.device_idx)?;
        let last_seen_timestamp = unix_as_millis()
            .saturating_mul(1000)
            .saturating_sub(1_000_000);

        let process_utilization_samples = dev.process_utilization_stats(last_seen_timestamp)?;

        let mut current = Utilization::default();
        let mut vaild = false;
        current.sys_process_num = dev.running_compute_processes_count()?;
        for process_utilization_sample in process_utilization_samples {
            if process_utilization_sample.timestamp < last_seen_timestamp {
                continue;
            }
            vaild = true;
            current.sys_current += process_utilization_sample.sm_util;
            let codec_util = codec_normalize(
                process_utilization_sample.enc_util + process_utilization_sample.dec_util,
            );
            current.sys_current += codec_util;

            if process_utilization_sample.pid == self.pid {
                current.user_current += process_utilization_sample.sm_util;
                current.user_current += codec_util;
            }
        }

        if !vaild {
            Ok(None)
        } else {
            Ok(Some(current))
        }
    }
}

const fn codec_normalize(x: u32) -> u32 {
    x * 85 / 100
}

pub(crate) fn unix_as_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
