use std::path::PathBuf;

use hf_hub::{Repo, RepoType, api::sync::Api};

use crate::error::Result;

pub fn model_repo(repo_id: impl Into<String>) -> Repo {
    Repo::new(repo_id.into(), RepoType::Model)
}

pub fn download_files(repo_id: impl Into<String>, files: &[&str]) -> Result<Vec<PathBuf>> {
    let api = Api::new()?;
    let repo = api.repo(model_repo(repo_id));
    files.iter().map(|file| repo.get(file).map_err(Into::into)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_repo_uses_model_repo_type() {
        let repo = model_repo("microsoft/VibeVoice-ASR-HF");
        let debug = format!("{repo:?}");
        assert!(debug.contains("VibeVoice-ASR-HF"));
        assert!(debug.contains("Model"));
    }

    #[test]
    fn download_files_with_empty_list_returns_empty() {
        let files = download_files("microsoft/VibeVoice-ASR-HF", &[]).unwrap();
        assert!(files.is_empty());
    }
}
