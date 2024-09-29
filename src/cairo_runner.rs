use std::path::PathBuf;

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
