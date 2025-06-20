use safetensors::Dtype as RDtype;
use safetensors::SafeTensorError as RSafeTensorError;
use safetensors::tensor::TensorView as RTensorView;
use safetensors::View;
use std::borrow::Cow;
use std::collections::HashMap;
use crate::ffi::{ Dtype, TensorView, PairStrStr, PairStrUsize, PairStrTensorView};
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

    #[derive(Debug)]
    enum SafeTensorError {
        /// The header is an invalid UTF-8 string and cannot be read.
        InvalidHeader,
        /// The header's first byte is not the expected `{`.
        InvalidHeaderStart,
        /// The header does contain a valid string, but it is not valid JSON.
        InvalidHeaderDeserialization,
        /// The header is large than 100Mo which is considered too large (Might evolve in the future).
        HeaderTooLarge,
        /// The header is smaller than 8 bytes
        HeaderTooSmall,
        /// The header length is invalid
        InvalidHeaderLength,
        /// The tensor name was not found in the archive
        TensorNotFound,
        /// Invalid information between shape, dtype and the proposed offsets in the file
        TensorInvalidInfo,
        /// The offsets declared for tensor with name `String` in the header are invalid
        InvalidOffset,
        /// IoError
        IoError,
        /// JSON error
        JsonError,
        /// The follow tensor cannot be created because the buffer size doesn't match shape + dtype
        InvalidTensorView,
        /// The metadata is invalid because the data offsets of the tensor does not
        /// fully cover the buffer part of the file. The last offset **must** be
        /// the end of the file.
        MetadataIncompleteBuffer,
        /// The metadata contains information (shape or shape * dtype size) which lead to an
        /// arithmetic overflow. This is most likely an error in the file.
        ValidationOverflow,
        /// For smaller than 1 byte dtypes, some slices will happen outside of the byte boundary, some special care has to be taken
        /// and standard functions will fail
        MisalignedSlice,
    }

    
    #[derive(Debug, PartialEq, Eq, Clone)]
    struct TensorView <'a> {
        shape: Vec<usize>,
        dtype: Dtype,
        data: &'a [u8],
    }

    struct PairStrStr {
        key: String,
        value: String,
    }

    struct PairStrUsize {
        key: String,
        value: usize,
    }

    struct PairStrTensorView<'a> {
        key: String,
        value: TensorView<'a>,
    }

    // Rust types and signatures exposed to C++.
    extern "Rust" {
        fn make_tensor_view(dtype: Dtype, shape: Vec<usize>, data: &'static [u8]) -> TensorView<'static>;

        fn serialize(data: Vec<PairStrTensorView>, data_info: Vec<PairStrStr>) -> Vec<u8>;
    }
}

fn make_tensor_view(dtype: Dtype, shape: Vec<usize>, data: &'static [u8]) -> TensorView<'static> {
    RTensorView::new(dtype.into(), shape, data)
        .map(|tv| tv.into()).expect("Failed to create TensorView")
}


fn prepare(
    tensor_dict: Vec<PairStrTensorView>
) -> Result<HashMap<String, TensorView>, RSafeTensorError> {
    let mut tensors = HashMap::with_capacity(tensor_dict.len());
    for tensor in tensor_dict {
        let shape = tensor.value.shape();
        let dtype: RDtype = tensor.value.dtype();
        let data = tensor.value.data();
        if data.len() != shape.iter().product::<usize>() * dtype.size() {
            return Err(RSafeTensorError::InvalidTensorView(
                dtype,
                shape.to_vec(),
                data.len()
            ));
        }
        tensors.insert(tensor.key, tensor.value);
    }
    Ok(tensors)
}

fn convert_to_hashmap_string(dict: Vec<PairStrStr>) -> Result<Option<HashMap<String, String>>, RSafeTensorError> {
    let mut hashmap = HashMap::with_capacity(dict.len());
    for item in dict {
        hashmap.insert(item.key, item.value);
    }
    Ok(Some(hashmap))
}

fn convert_to_hashmap_usize(dict: Vec<PairStrUsize>) -> Result<HashMap<String, usize>, RSafeTensorError> {
    let mut hashmap = HashMap::with_capacity(dict.len());
    for item in dict {
        hashmap.insert(item.key, item.value);
    }
    Ok(hashmap)
}

fn serialize(
    data: Vec<PairStrTensorView>,
    data_info: Vec<PairStrStr>,
) -> Vec<u8> {
    let tensors = prepare(data).expect("Failed to prepare tensors for serialization");
    safetensors::tensor::serialize(
        tensors,
        convert_to_hashmap_string(data_info).expect("Failed to convert data_info to hashmap"),
    ).expect("Failed to serialize tensors")
}

impl View for TensorView<'_> {
    fn data(&self) -> Cow<[u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data().len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> RDtype {
        self.dtype.into()
    }
}

