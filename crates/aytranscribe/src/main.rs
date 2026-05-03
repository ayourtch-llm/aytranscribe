use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use vibevoice_asr::VibeVoiceAsrProcessor;
use vibevoice_core::DeviceSpec;

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long)]
    model_dir: PathBuf,

    #[arg(long)]
    tokenizer: PathBuf,

    #[arg(long)]
    audio: PathBuf,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long)]
    context: Option<String>,

    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(cli)
}

fn run(cli: Cli) -> Result<()> {
    let device = parse_device(&cli.device)?.resolve()?;
    let processor = VibeVoiceAsrProcessor::from_file(&cli.tokenizer)?;
    let inputs = processor.prepare_audio_file(&cli.audio, &device, cli.context.as_deref())?;
    let mut model =
        vibevoice_asr::VibeVoiceAsrModel::from_local_dir(&cli.model_dir, &cli.tokenizer, device)?;

    let output = model.transcribe_inputs(&inputs, cli.max_new_tokens)?;
    println!("{output}");
    Ok(())
}

fn parse_device(spec: &str) -> Result<DeviceSpec> {
    if spec.eq_ignore_ascii_case("cpu") {
        return Ok(DeviceSpec::Cpu);
    }
    if let Some(rest) = spec.strip_prefix("cuda:") {
        return Ok(DeviceSpec::Cuda(rest.parse()?));
    }
    anyhow::bail!("unsupported device spec `{spec}`, use `cpu` or `cuda:N`")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_parses_required_args() {
        let cli = Cli::try_parse_from([
            "aytranscribe",
            "--model-dir",
            "model",
            "--tokenizer",
            "tok.json",
            "--audio",
            "sample.wav",
        ])
        .unwrap();
        assert_eq!(cli.max_new_tokens, 128);
        assert_eq!(cli.device, "cpu");
    }

    #[test]
    fn parse_device_rejects_invalid_value() {
        assert!(parse_device("gpu").is_err());
    }
}
