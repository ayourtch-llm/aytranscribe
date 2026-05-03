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

