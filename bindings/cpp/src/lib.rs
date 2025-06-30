use crate::ffi::{PairStrStr, PairStrTensorView, TensorView};
use safetensors::Dtype as RDtype;
use safetensors::{SafeTensorError, SafeTensors, View};
use std::borrow::Cow;
use std::collections::HashMap;
mod conversion;

#[cxx::bridge(namespace = "safetensors")]
mod ffi {
    /// The various available dtypes. They MUST be in increasing alignment order
    #[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
    enum Dtype {
        /// Boolan type
        BOOL,
        /// MXF4 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
        F4,
        /// MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
        #[allow(non_camel_case_types)]
        F6_E2M3,
        /// MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
        #[allow(non_camel_case_types)]
        F6_E3M2,
        /// Unsigned byte
        U8,
        /// Signed byte
        I8,
        /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
        #[allow(non_camel_case_types)]
        F8_E5M2,
        /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
        #[allow(non_camel_case_types)]
        F8_E4M3,
        /// F8_E8M0 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
        #[allow(non_camel_case_types)]
        F8_E8M0,
        /// Signed integer (16-bit)
        I16,
        /// Unsigned integer (16-bit)
        U16,
        /// Half-precision floating point
        F16,
        /// Brain floating point
        BF16,
        /// Signed integer (32-bit)
        I32,
        /// Unsigned integer (32-bit)
        U32,
        /// Floating point (32-bit)
        F32,
        /// Floating point (64-bit)
        F64,
        /// Signed integer (64-bit)
        I64,
        /// Unsigned integer (64-bit)
        U64,
    }

    #[derive(Debug, PartialEq, Eq, Clone)]
    struct TensorView<'a> {
        shape: Vec<usize>,
        dtype: Dtype,
        data: &'a [u8],
        data_len: usize,
    }

    #[derive(Debug, Clone)]
    struct PairStrStr {
        key: String,
        value: String,
    }

    struct PairStrTensorView<'a> {
        key: String,
        value: TensorView<'a>,
    }

    // Rust types and signatures exposed to C++.
    extern "Rust" {
        // TODO(dp): implement with HashMap
        fn serialize(data: Vec<PairStrTensorView>, data_info: Vec<PairStrStr>) -> Result<Vec<u8>>;

        fn serialize_to_file(
            data: Vec<PairStrTensorView>,
            data_info: Vec<PairStrStr>,
            path: &str,
        ) -> Result<()>;

        fn deserialize(bytes: &[u8]) -> Result<Vec<PairStrTensorView>>;

        fn metadata(bytes: &[u8]) -> Result<Vec<PairStrStr>>;
    }
}

fn serialize(
    data: Vec<PairStrTensorView>,
    data_info: Vec<PairStrStr>,
) -> Result<Vec<u8>, SafeTensorError> {
    let tensors = prepare(data)?;
    let out = safetensors::tensor::serialize(tensors, convert_to_hashmap_string(data_info))?;
    Ok(out)
}

fn serialize_to_file(
    data: Vec<PairStrTensorView>,
    data_info: Vec<PairStrStr>,
    path: &str,
) -> Result<(), SafeTensorError> {
    let tensors = prepare(data)?;
    safetensors::tensor::serialize_to_file(
        tensors,
        convert_to_hashmap_string(data_info),
        path.as_ref(),
    )?;
    Ok(())
}

fn deserialize(bytes: &[u8]) -> Result<Vec<PairStrTensorView>, SafeTensorError> {
    let safetensor = SafeTensors::deserialize(bytes)?;
    let tensors = safetensor.tensors();

    let mut items = Vec::with_capacity(tensors.len());
    for (tensor_name, tensor) in tensors {
        let mut shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();
        if dtype == RDtype::F4 {
            let n = shape.len();
            shape[n - 1] /= 2; // F4 is stored as F8
        }
        let data = tensor.data();
        let data_len = tensor.data_len();
        items.push(PairStrTensorView {
            key: tensor_name,
            value: TensorView {
                shape,
                dtype: dtype.into(),
                data,
                data_len,
            },
        });
    }
    Ok(items)
}

fn metadata(bytes: &[u8]) -> Result<Vec<PairStrStr>, SafeTensorError> {
    let (_n, metadata) = SafeTensors::read_metadata(bytes)?;
    let Some(metadata) = &metadata.metadata() else {
        return Ok(Vec::new());
    };
    let mut items = Vec::with_capacity(metadata.len());
    for (key, value) in metadata {
        items.push(PairStrStr {
            key: key.to_string(),
            value: value.to_string(),
        });
    }
    Ok(items)
}

// private
impl View for TensorView<'_> {
    fn data(&self) -> Cow<[u8]> {
        self.data.into()
    }

    fn data_len(&self) -> usize {
        self.data_len
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> RDtype {
        self.dtype.into()
    }
}

fn prepare(
    tensor_dict: Vec<PairStrTensorView>,
) -> Result<HashMap<String, TensorView>, SafeTensorError> {
    let mut tensors = HashMap::with_capacity(tensor_dict.len());
    for tensor in tensor_dict {
        let mut shape: Vec<usize> = tensor.value.shape().to_vec();
        let dtype: RDtype = tensor.value.dtype();

        if dtype == RDtype::F4 {
            let n = shape.len();
            shape[n - 1] *= 2;
        };

        tensors.insert(tensor.key, tensor.value);
    }
    Ok(tensors)
}

fn convert_to_hashmap_string(dict: Vec<PairStrStr>) -> Option<HashMap<String, String>> {
    if dict.is_empty() {
        None
    } else {
        Some(
            dict.into_iter()
                .map(|item| (item.key, item.value))
                .collect(),
        )
    }
}
