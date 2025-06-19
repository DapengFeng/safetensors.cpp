use safetensors::{Dtype, View, SafeTensorError, tensor::TensorView};
use std::borrow::Cow;
use std::collections::HashMap;
use crate::ffi::{ CxxDtype, CxxSafeTensorError, CxxTensorView, CxxStrStr, CxxStrUsize, CxxStrTensorView};

#[cxx::bridge(namespace = "safetensors")]
mod ffi {
    /// The various available dtypes. They MUST be in increasing alignment order
    #[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
    enum CxxDtype {
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
    enum CxxSafeTensorError {
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
    struct CxxTensorView {
        shape_: Vec<usize>,
        dtype_: CxxDtype,
        data_: Vec<u8>,
    }

    struct CxxStrStr {
        key: String,
        value: String,
    }

    struct CxxStrUsize {
        key: String,
        value: usize,
    }

    struct CxxStrTensorView {
        key: String,
        value: CxxTensorView,
    }

    // Rust types and signatures exposed to C++.
    extern "Rust" {
        /// The shape of the tensor
        fn shape(self: &CxxTensorView) -> &[usize];
        /// The `Dtype` of the tensor
        fn dtype(self: &CxxTensorView) -> CxxDtype;
        /// The data of the tensor
        fn data(self: &CxxTensorView) -> &[u8];
        /// The length of the data, in bytes.
        /// This is necessary as this might be faster to get than `data().len()`
        /// for instance for tensors residing in GPU.
        fn data_len(self: &CxxTensorView) -> usize;

        fn make_tensor_view(dtype: CxxDtype, shape: Vec<usize>, data: Vec<u8>) -> CxxTensorView;

        fn serialize(data: Vec<CxxStrTensorView>, data_info: Vec<CxxStrStr>) -> Vec<u8>;
    }
}

impl CxxTensorView {
    fn shape(&self) -> &[usize] {
        &self.shape_
    }

    fn dtype(&self) -> CxxDtype {
        return self.dtype_
    }

    fn data(&self) -> &[u8] {
        &self.data_
    }

    fn data_len(&self) -> usize {
        self.data_.len()
    }
}

fn make_tensor_view(dtype: CxxDtype, shape: Vec<usize>, data: Vec<u8>) -> CxxTensorView {
    TensorView::new(dtype.into(), shape, &data)
        .map(|tv| tv.into()).expect("Failed to create CxxTensorView")
}


fn prepare(
    tensor_dict: Vec<CxxStrTensorView>
) -> Result<HashMap<String, CxxTensorView>, SafeTensorError> {
    let mut tensors = HashMap::with_capacity(tensor_dict.len());
    for tensor in tensor_dict {
        let shape = tensor.value.shape();
        let dtype: Dtype = tensor.value.dtype().into();
        let data = tensor.value.data();
        if data.len() != shape.iter().product::<usize>() * dtype.size() {
            return Err(SafeTensorError::InvalidTensorView(
                dtype,
                shape.to_vec(),
                data.len()
            ));
        }
        tensors.insert(tensor.key, tensor.value);
    }
    Ok(tensors)
}

fn convert_to_hashmap_string(dict: Vec<CxxStrStr>) -> Result<Option<HashMap<String, String>>, SafeTensorError> {
    let mut hashmap = HashMap::with_capacity(dict.len());
    for item in dict {
        hashmap.insert(item.key, item.value);
    }
    Ok(Some(hashmap))
}

fn convert_to_hashmap_usize(dict: Vec<CxxStrUsize>) -> Result<HashMap<String, usize>, SafeTensorError> {
    let mut hashmap = HashMap::with_capacity(dict.len());
    for item in dict {
        hashmap.insert(item.key, item.value);
    }
    Ok(hashmap)
}

fn serialize(
    data: Vec<CxxStrTensorView>,
    data_info: Vec<CxxStrStr>,
) -> Vec<u8> {
    let tensors = prepare(data).expect("Failed to prepare tensors for serialization");
    safetensors::tensor::serialize(
        tensors,
        convert_to_hashmap_string(data_info).expect("Failed to convert data_info to hashmap"),
    ).expect("Failed to serialize tensors")
}

impl<'a> Into<CxxTensorView> for TensorView<'a> {
    fn into(self) -> CxxTensorView {
        CxxTensorView {
            shape_: self.shape().to_vec(),
            dtype_: self.dtype().into(),
            data_: self.data().to_vec(),
        }
    }
}

impl View for CxxTensorView {
    fn data(&self) -> Cow<[u8]> {
        self.data().into()
    }

    fn data_len(&self) -> usize {
        self.data_len()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn dtype(&self) -> Dtype {
        self.dtype().into()
    }
}

impl Into<CxxSafeTensorError> for SafeTensorError {
    fn into(self) -> CxxSafeTensorError {
        match self {
            SafeTensorError::InvalidHeader(_) => CxxSafeTensorError::InvalidHeader,
            SafeTensorError::InvalidHeaderStart => CxxSafeTensorError::InvalidHeaderStart,
            SafeTensorError::InvalidHeaderDeserialization(_) => {
                CxxSafeTensorError::InvalidHeaderDeserialization
            }
            SafeTensorError::HeaderTooLarge => CxxSafeTensorError::HeaderTooLarge,
            SafeTensorError::HeaderTooSmall => CxxSafeTensorError::HeaderTooSmall,
            SafeTensorError::InvalidHeaderLength => CxxSafeTensorError::InvalidHeaderLength,
            SafeTensorError::TensorNotFound(_) => CxxSafeTensorError::TensorNotFound,
            SafeTensorError::TensorInvalidInfo => CxxSafeTensorError::TensorInvalidInfo,
            SafeTensorError::InvalidOffset(_) => CxxSafeTensorError::InvalidOffset,
            SafeTensorError::IoError(_) => CxxSafeTensorError::IoError,
            SafeTensorError::JsonError(_) => CxxSafeTensorError::JsonError,
            SafeTensorError::InvalidTensorView(_, _, _) => CxxSafeTensorError::InvalidTensorView,
            SafeTensorError::MetadataIncompleteBuffer => CxxSafeTensorError::MetadataIncompleteBuffer,
            SafeTensorError::ValidationOverflow => CxxSafeTensorError::ValidationOverflow,
            SafeTensorError::MisalignedSlice => CxxSafeTensorError::MisalignedSlice,
        }
    }
}

impl From<CxxDtype> for Dtype {
    fn from(dtype: CxxDtype) -> Self {
        match dtype {
            CxxDtype::BOOL => Dtype::BOOL,
            CxxDtype::F4 => Dtype::F4,
            CxxDtype::F6_E2M3 => Dtype::F6_E2M3,
            CxxDtype::F6_E3M2 => Dtype::F6_E3M2,
            CxxDtype::U8 => Dtype::U8,
            CxxDtype::I8 => Dtype::I8,
            CxxDtype::F8_E5M2 => Dtype::F8_E5M2,
            CxxDtype::F8_E4M3 => Dtype::F8_E4M3,
            CxxDtype::F8_E8M0 => Dtype::F8_E8M0,
            CxxDtype::I16 => Dtype::I16,
            CxxDtype::U16 => Dtype::U16,
            CxxDtype::F16 => Dtype::F16,
            CxxDtype::BF16 => Dtype::BF16,
            CxxDtype::I32 => Dtype::I32,
            CxxDtype::U32 => Dtype::U32,
            CxxDtype::F32 => Dtype::F32,
            CxxDtype::F64 => Dtype::F64,
            CxxDtype::I64 => Dtype::I64,
            CxxDtype::U64 => Dtype::U64,
            CxxDtype { repr: 19_u8..=u8::MAX } => todo!(),
        }
    }
}


impl From<Dtype> for CxxDtype {
    fn from(dtype: Dtype) -> Self {
        match dtype {
            Dtype::BOOL => CxxDtype::BOOL,
            Dtype::F4 => CxxDtype::F4,
            Dtype::F6_E2M3 => CxxDtype::F6_E2M3,
            Dtype::F6_E3M2 => CxxDtype::F6_E3M2,
            Dtype::U8 => CxxDtype::U8,
            Dtype::I8 => CxxDtype::I8,
            Dtype::F8_E5M2 => CxxDtype::F8_E5M2,
            Dtype::F8_E4M3 => CxxDtype::F8_E4M3,
            Dtype::F8_E8M0 => CxxDtype::F8_E8M0,
            Dtype::I16 => CxxDtype::I16,
            Dtype::U16 => CxxDtype::U16,
            Dtype::F16 => CxxDtype::F16,
            Dtype::BF16 => CxxDtype::BF16,
            Dtype::I32 => CxxDtype::I32,
            Dtype::U32 => CxxDtype::U32,
            Dtype::F32 => CxxDtype::F32,
            Dtype::F64 => CxxDtype::F64,
            Dtype::I64 => CxxDtype::I64,
            Dtype::U64 => CxxDtype::U64,
            _ => todo!(),
        }
    }
}
