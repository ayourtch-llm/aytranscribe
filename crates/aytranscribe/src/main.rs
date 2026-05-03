use std::{fs, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use vibevoice_asr::{DEFAULT_MODEL_REPO, VibeVoiceAsrModel, VibeVoiceAsrProcessor};
use vibevoice_core::DeviceSpec;

#[derive(Debug, Parser)]
struct Cli {
    audio: PathBuf,

    output: PathBuf,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long)]
    context: Option<String>,

    #[arg(long, default_value = "auto")]
    device: String,

    #[arg(long)]
    model_dir: Option<PathBuf>,

    #[arg(long)]
    tokenizer: Option<PathBuf>,

    #[arg(long, default_value = DEFAULT_MODEL_REPO)]
    model_repo: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(cli)
}

fn run(cli: Cli) -> Result<()> {
    let device = parse_device(&cli.device)?.resolve()?;
    let (processor, mut model) = match (&cli.model_dir, &cli.tokenizer) {
        (Some(model_dir), Some(tokenizer)) => {
            let processor = VibeVoiceAsrProcessor::from_file(tokenizer)?;
            let model = VibeVoiceAsrModel::from_local_dir(model_dir, tokenizer, device.clone())?;
            (processor, model)
        }
        _ => {
            let model = VibeVoiceAsrModel::from_hf_hub(Some(&cli.model_repo), device.clone())?;
            let processor = model.processor_from_tokenizer()?;
            (processor, model)
        }
    };
    let inputs = processor.prepare_audio_file(&cli.audio, &device, cli.context.as_deref())?;

    let output = model.transcribe_inputs(&inputs, cli.max_new_tokens)?;
    fs::write(&cli.output, &output)?;
    println!("{output}");
    Ok(())
}

fn parse_device(spec: &str) -> Result<DeviceSpec> {
    if spec.eq_ignore_ascii_case("auto") {
        return Ok(DeviceSpec::Auto);
    }
    if spec.eq_ignore_ascii_case("cpu") {
        return Ok(DeviceSpec::Cpu);
    }
    if let Some(rest) = spec.strip_prefix("cuda:") {
        return Ok(DeviceSpec::Cuda(rest.parse()?));
    }
    anyhow::bail!("unsupported device spec `{spec}`, use `auto`, `cpu` or `cuda:N`")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_parses_required_args() {
        let cli = Cli::try_parse_from([
            "aytranscribe",
            "sample.wav",
            "out.txt",
        ])
        .unwrap();
        assert_eq!(cli.max_new_tokens, 128);
        assert_eq!(cli.device, "auto");
    }

    #[test]
    fn parse_device_rejects_invalid_value() {
        assert!(parse_device("gpu").is_err());
    }
}
