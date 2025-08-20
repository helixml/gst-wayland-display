use crate::utils::device::PCIVendor;
use smithay::backend::drm::DrmNode;
use smithay::backend::egl::EGLDevice;
use std::error::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct GPUDevice {
    drm_node: DrmNode,
    pci_vendor: PCIVendor,
    device_name: String,
}
impl GPUDevice {
    pub fn drm_node(&self) -> &DrmNode {
        &self.drm_node
    }

    pub fn pci_vendor(&self) -> &PCIVendor {
        &self.pci_vendor
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}
impl TryFrom<DrmNode> for GPUDevice {
    type Error = Box<dyn std::error::Error>;
    fn try_from(drm_node: DrmNode) -> Result<Self, Self::Error> {
        let devices = enumerate_gpu_devices()?;
        if let Some(device) = devices.iter().find(|d| d.drm_node == drm_node) {
            Ok(device.clone())
        } else {
            Err("No GPU device for given DRM node".into())
        }
    }
}
impl std::fmt::Display for GPUDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPUDevice {{ drm_node: {}, pci_vendor: {}, device_name: {} }}",
            self.drm_node, self.pci_vendor, self.device_name
        )
    }
}

pub fn enumerate_gpu_devices() -> Result<Vec<GPUDevice>, Box<dyn Error>> {
    EGLDevice::enumerate()?
        .filter(|dev| !dev.is_software())
        .map(|dev| -> Result<GPUDevice, Box<dyn Error>> {
            let drm_path = dev.drm_device_path()?;
            let drm_node = DrmNode::from_path(drm_path)?;
            let minor = drm_node.minor();
            let vendor_str =
                std::fs::read_to_string(format!("/sys/class/drm/card{}/device/vendor", minor))?;
            let vendor_str = vendor_str.trim_start_matches("0x").trim_end_matches('\n');
            let vendor = u32::from_str_radix(&vendor_str, 16)?;

            let device_id =
                std::fs::read_to_string(format!("/sys/class/drm/card{}/device/device", minor))?;
            let device_id = device_id.trim_start_matches("0x").trim_end_matches('\n');

            // Look up in hwdata PCI database
            let device_name = match std::fs::read_to_string("/usr/share/hwdata/pci.ids") {
                Ok(pci_ids) => parse_pci_ids(&pci_ids, device_id).unwrap_or("".to_owned()),
                Err(e) => {
                    tracing::warn!("Failed to read /usr/share/hwdata/pci.ids: {}", e);
                    "".to_owned()
                }
            };

            Ok(GPUDevice {
                drm_node,
                pci_vendor: PCIVendor::try_from(vendor)?,
                device_name,
            })
        })
        .collect()
}

fn parse_pci_ids(pci_data: &str, device_id: &str) -> Option<String> {
    for line in pci_data.lines() {
        if let Some(stripped) = line.strip_prefix(&format!("\t{}", device_id)) {
            if stripped.starts_with("  ") {
                let device_name = stripped.trim();
                return Some(device_name.to_owned());
            }
        }
    }
    None
}
