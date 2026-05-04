use std::{fs, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use vibevoice_asr::TranscriptionSegment;

#[derive(Debug, Parser)]
struct Cli {
    input: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(long, default_value_t = 80)]
    width: usize,
}

fn format_timestamp(seconds: f64) -> String {
    let total = seconds.floor() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let secs = total % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes:02}:{secs:02}")
    }
}

fn wrap_text(text: &str, width: usize, indent: &str) -> String {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if line.is_empty() {
            line.push_str(word);
        } else if line.len() + 1 + word.len() > width {
            lines.push(line);
            line = format!("{indent}{word}");
        } else {
            line.push(' ');
            line.push_str(word);
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines.join("\n")
}

fn format_segments(segments: &[TranscriptionSegment], width: usize) -> String {
    let mut out = String::new();
    let mut last_speaker: Option<i64> = None;

    for seg in segments {
        let ts = format!("[{}->{}]", format_timestamp(seg.start), format_timestamp(seg.end));
        let speaker_changed = last_speaker != Some(seg.speaker);
        last_speaker = Some(seg.speaker);

        if speaker_changed {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&format!("Speaker {}:\n", seg.speaker));
        }

        let prefix = format!("  {ts} ");
        let indent = " ".repeat(prefix.len());
        let content_width = width.saturating_sub(prefix.len());
        let wrapped = wrap_text(seg.content.trim(), content_width, &indent);

        out.push_str(&prefix);
        out.push_str(wrapped.trim_start());
        out.push('\n');
    }
    out
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let json = fs::read_to_string(&cli.input)?;
    let segments: Vec<TranscriptionSegment> = serde_json::from_str(&json)?;
    let text = format_segments(&segments, cli.width);

    if let Some(output) = &cli.output {
        fs::write(output, &text)?;
        println!("Written to {}", output.display());
    } else {
        print!("{text}");
    }
    Ok(())
}
