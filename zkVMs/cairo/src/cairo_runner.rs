use num_bigint::BigInt;
use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    path::PathBuf,
};

use bincode::enc::write::Writer;
use cairo1_run::{
    cairo_run_program, error::Error, Cairo1RunConfig, FuncArg, MaybeRelocatable, VirtualMachine,
};
use cairo_lang_sierra::program::{Program, VersionedProgram};
use cairo_vm::{
    air_public_input::PublicInputError, types::layout_name::LayoutName,
    vm::errors::trace_errors::TraceError, vm::runners::cairo_runner::CairoRunner as CairoVMRunner,
};
use luminal::prelude::*;
use tracing::info;

use crate::{fixed_point::felt_fp_to_float, CairoCompilerError};

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

impl CairoRunner {
    pub(crate) fn new(config: CairoRunnerConfig) -> Self {
        Self { config }
    }

    pub(crate) fn run(
        &self,
        sierra_file: PathBuf,
        inputs: Vec<FuncArg>,
        returns_dict: bool,
    ) -> Result<Tensor, CairoCompilerError> {
        // Load program
        let program = self.load_sierra_file(sierra_file)?;

        // Set up cairo runner config.
        let config = &self.config;
        let cairo_run_config = Cairo1RunConfig {
            args: &inputs,
            serialize_output: true,
            trace_enabled: config.trace_file.is_some() || config.air_public_input.is_some(),
            relocate_mem: config.memory_file.is_some() || config.air_public_input.is_some(),
            layout: LayoutName::all_cairo,
            proof_mode: config.proof_mode,
            finalize_builtins: config.air_private_input.is_some()
                || config.cairo_pie_output.is_some(),
            append_return_values: config.append_return_values,
            dynamic_layout_params: None,
        };

        // Run the program
        let (runner, return_values, _) = cairo_run_program(&program, cairo_run_config)?;

        // Generate output files (trace, memory, cairopie files)
        self.generate_output_files(&runner)?;

        // Fetch the actual data from memory
        let data = if returns_dict {
            self.fetch_data_from_memory_dict(&runner.vm, &return_values)?
        } else {
            self.fetch_data_from_memory_arr(&runner.vm, &return_values)?
        };

        Ok(Tensor::new(data))
    }

    fn fetch_data_from_memory_dict(
        &self,
        vm: &VirtualMachine,
        return_values: &[MaybeRelocatable],
    ) -> Result<Vec<f32>, CairoCompilerError> {
        let dict_ptr = return_values[0].get_relocatable().ok_or_else(|| {
            CairoCompilerError::RuntimeError("Invalid dictionary pointer".to_string())
        })?;

        let dict_mem = vm
            .get_continuous_range((dict_ptr.segment_index, 0).into(), dict_ptr.offset)
            .map_err(|e| {
                CairoCompilerError::RuntimeError(format!(
                    "Failed to fetch dictionary memory: {}",
                    e
                ))
            })?;

        let mut data: Vec<f32> = Vec::new();
        let mut last_values: HashMap<BigInt, f32> = HashMap::new();

        for chunk in dict_mem.chunks(3) {
            if let [key, _, value] = chunk {
                let key = match key {
                    MaybeRelocatable::Int(felt) => felt.to_bigint(),
                    _ => continue, // Skip non-integer keys
                };

                let float_value = match value {
                    MaybeRelocatable::Int(felt) => felt_fp_to_float(&felt.to_bigint()),
                    MaybeRelocatable::RelocatableValue(ptr) => {
                        if let Some(MaybeRelocatable::Int(value_felt)) = vm.get_maybe(ptr) {
                            felt_fp_to_float(&value_felt.to_bigint())
                        } else {
                            None
                        }
                    }
                };

                if let Some(float_value) = float_value {
                    last_values.insert(key, float_value);
                }
            }
        }

        // Sort the keys to ensure consistent ordering
        let mut sorted_keys: Vec<_> = last_values.keys().collect();
        sorted_keys.sort();

        for key in sorted_keys {
            data.push(last_values[key]);
        }

        Ok(data)
    }

    fn fetch_data_from_memory_arr(
        &self,
        vm: &VirtualMachine,
        return_values: &[MaybeRelocatable],
    ) -> Result<Vec<f32>, CairoCompilerError> {
        if return_values.len() != 2 {
            return Err(CairoCompilerError::RuntimeError(
                "Expected 2 return values (start and end pointers)".to_string(),
            ));
        }

        let start = return_values[0]
            .get_relocatable()
            .ok_or_else(|| CairoCompilerError::RuntimeError("Invalid start pointer".to_string()))?;
        let end = return_values[1]
            .get_relocatable()
            .ok_or_else(|| CairoCompilerError::RuntimeError("Invalid end pointer".to_string()))?;

        let size = (end - start).map_err(|e| CairoCompilerError::RuntimeError(e.to_string()))?;

        let memory_data = vm
            .get_continuous_range(start, size)
            .map_err(|e| CairoCompilerError::RuntimeError(e.to_string()))?;

        // Convert the memory data to f32
        let data: Result<Vec<f32>, CairoCompilerError> = memory_data
            .into_iter()
            .map(|v| match v {
                MaybeRelocatable::Int(felt) => {
                    let x = felt_fp_to_float(&felt.to_bigint()).ok_or_else(|| {
                        CairoCompilerError::RuntimeError(format!("Failed to parse bigint as float"))
                    });

                    x
                }
                _ => Err(CairoCompilerError::RuntimeError(
                    "Unexpected relocatable value in output".to_string(),
                )),
            })
            .collect();

        data
    }

    fn load_sierra_file(&self, file_path: PathBuf) -> Result<Program, CairoCompilerError> {
        let content = fs::read(&file_path).map_err(|e| {
            CairoCompilerError::SierraLoadError(format!("Failed to read file: {:?}", e))
        })?;

        let versioned_program =
            serde_json::from_slice::<VersionedProgram>(&content).map_err(|e| {
                CairoCompilerError::SierraLoadError(format!("Failed to deserialize file: {:?}", e))
            })?;

        let program = versioned_program
            .into_v1()
            .map_err(|_| CairoCompilerError::SierraLoadError("Version conversion failed".into()))?
            .program;

        let file_name = file_path
            .file_name()
            .ok_or_else(|| CairoCompilerError::SierraLoadError("Failed to get file name".into()))?
            .to_str()
            .ok_or_else(|| {
                CairoCompilerError::SierraLoadError("Failed to convert file name to string".into())
            })?
            .to_string();

        info!("📄 Loaded program: {}", file_name);

        Ok(program)
    }

    fn generate_output_files(&self, runner: &CairoVMRunner) -> Result<(), CairoCompilerError> {
        let config = &self.config;

        if let (Some(file_path), Some(trace_file), Some(memory_file)) = (
            config.air_private_input.clone(),
            config.trace_file.clone(),
            config.memory_file.clone(),
        ) {
            // Get absolute paths of trace_file & memory_file
            let trace_path = trace_file
                .as_path()
                .canonicalize()
                .unwrap_or(trace_file.clone())
                .to_string_lossy()
                .to_string();
            let memory_path = memory_file
                .as_path()
                .canonicalize()
                .unwrap_or(memory_file.clone())
                .to_string_lossy()
                .to_string();

            let json = runner
                .get_air_private_input()
                .to_serializable(trace_path, memory_path)
                .serialize_json()
                .map_err(PublicInputError::Serde)?;
            std::fs::write(file_path, json)?;
        }

        if let Some(ref file_path) = config.cairo_pie_output {
            runner.get_cairo_pie()?.write_zip_file(file_path)?
        }

        if let Some(trace_path) = &config.trace_file {
            let relocated_trace = runner
                .relocated_trace
                .clone()
                .ok_or(Error::Trace(TraceError::TraceNotRelocated))?;
            let trace_file = std::fs::File::create(trace_path)?;
            let mut trace_writer =
                FileWriter::new(io::BufWriter::with_capacity(3 * 1024 * 1024, trace_file));

            cairo_vm::cairo_run::write_encoded_trace(&relocated_trace, &mut trace_writer)?;
            trace_writer.flush()?;
        }
        if let Some(memory_path) = &config.memory_file {
            let memory_file = std::fs::File::create(memory_path)?;
            let mut memory_writer =
                FileWriter::new(io::BufWriter::with_capacity(5 * 1024 * 1024, memory_file));

            cairo_vm::cairo_run::write_encoded_memory(
                &runner.relocated_memory,
                &mut memory_writer,
            )?;
            memory_writer.flush()?;
        }

        Ok(())
    }
}
