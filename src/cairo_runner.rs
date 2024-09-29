use std::{
    io::{self, Write},
    path::PathBuf,
};

use bincode::enc::write::Writer;

struct FileWriter {
    buf_writer: io::BufWriter<std::fs::File>,
    bytes_written: usize,
}

impl Writer for FileWriter {
    fn write(&mut self, bytes: &[u8]) -> Result<(), bincode::error::EncodeError> {
        self.buf_writer
            .write_all(bytes)
            .map_err(|e| bincode::error::EncodeError::Io {
                inner: e,
                index: self.bytes_written,
            })?;

        self.bytes_written += bytes.len();

        Ok(())
    }
}

impl FileWriter {
    fn new(buf_writer: io::BufWriter<std::fs::File>) -> Self {
        Self {
            buf_writer,
            bytes_written: 0,
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        self.buf_writer.flush()
    }
}

#[derive(Debug, Clone)]
pub struct CairoRunnerConfig {
    pub trace_file: Option<PathBuf>,
    pub memory_file: Option<PathBuf>,
    pub proof_mode: bool,
    pub air_public_input: Option<PathBuf>,
    pub air_private_input: Option<PathBuf>,
    pub cairo_pie_output: Option<PathBuf>,
    pub append_return_values: bool,
}

impl Default for CairoRunnerConfig {
    fn default() -> Self {
        Self {
            trace_file: None,
            memory_file: None,
            proof_mode: false,
            air_public_input: None,
            air_private_input: None,
            cairo_pie_output: None,
            append_return_values: false,
        }
    }
}

pub(crate) struct CairoRunner {
    config: CairoRunnerConfig,
}

impl Default for CairoRunner {
    fn default() -> Self {
        Self {
            config: CairoRunnerConfig::default(),
        }
    }
}

