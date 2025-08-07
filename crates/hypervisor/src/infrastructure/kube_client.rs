use std::path::PathBuf;

use error_stack::Report;
use error_stack::ResultExt;
use kube::config::KubeConfigOptions;
use kube::config::Kubeconfig;
use kube::Client;
use kube::Config;

use crate::infrastructure::k8s::KubernetesError;

pub async fn init_kube_client(
    kubeconfig: Option<PathBuf>,
) -> Result<Client, Report<KubernetesError>> {
    let client = match kubeconfig {
        Some(kubeconfig_path) => {
            // Load kubeconfig from the specified file
            let kubeconfig = Kubeconfig::read_from(&kubeconfig_path).change_context(
                KubernetesError::ConnectionFailed {
                    message: format!(
                        "Failed to read kubeconfig file: {}",
                        kubeconfig_path.display()
                    ),
                },
            )?;

            let config = Config::from_custom_kubeconfig(kubeconfig, &KubeConfigOptions::default())
                .await
                .change_context(KubernetesError::ConnectionFailed {
                    message: format!(
                        "Failed to create config from kubeconfig: {}",
                        kubeconfig_path.display()
                    ),
                })?;

            Client::try_from(config).change_context(KubernetesError::ConnectionFailed {
                message: "Failed to create Kubernetes client from custom kubeconfig".to_string(),
            })?
        }
        None => {
            // Use default configuration (in-cluster or ~/.kube/config)
            Client::try_default()
                .await
                .change_context(KubernetesError::ConnectionFailed {
                    message: "Failed to create Kubernetes client".to_string(),
                })?
        }
    };
    Ok(client)
}
