use candle_core::Device;

use crate::error::{Result, VibeVoiceCoreError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeviceSpec {
    Cpu,
    Cuda(usize),
}

impl DeviceSpec {
    pub fn resolve(&self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Cuda(index) => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(*index).map_err(Into::into)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(VibeVoiceCoreError::UnsupportedConfig(format!(
                        "cuda device requested (index {index}) but vibevoice-core was built without the `cuda` feature"
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_resolves() {
        let device = DeviceSpec::Cpu.resolve().unwrap();
        assert!(matches!(device, Device::Cpu));
    }
}
