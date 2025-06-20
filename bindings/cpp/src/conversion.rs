use safetensors::Dtype as RDtype;
use safetensors::tensor::TensorView as RTensorView;

use crate::ffi::{ Dtype, TensorView};

// Upload: Rust -> Cxx
impl Into<Dtype> for RDtype {
    fn into(self) -> Dtype {
        match self {
            RDtype::BOOL => Dtype::BOOL,
            RDtype::F4 => Dtype::F4,
            RDtype::F6_E2M3 => Dtype::F6_E2M3,
            RDtype::F6_E3M2 => Dtype::F6_E3M2,
            RDtype::U8 => Dtype::U8,
            RDtype::I8 => Dtype::I8,
            RDtype::F8_E5M2 => Dtype::F8_E5M2,
            RDtype::F8_E4M3 => Dtype::F8_E4M3,
            RDtype::F8_E8M0 => Dtype::F8_E8M0,
            RDtype::I16 => Dtype::I16,
            RDtype::U16 => Dtype::U16,
            RDtype::F16 => Dtype::F16,
            RDtype::BF16 => Dtype::BF16,
            RDtype::I32 => Dtype::I32,
            RDtype::U32 => Dtype::U32,
            RDtype::F32 => Dtype::F32,
            RDtype::F64 => Dtype::F64,
            RDtype::I64 => Dtype::I64,
            RDtype::U64 => Dtype::U64,
            _ => todo!(),
        }
    }
}

impl<'a> Into<TensorView<'a>> for RTensorView<'a> {
    fn into(self) -> TensorView<'a> {
        TensorView {
            shape: self.shape().to_vec(),
            dtype: self.dtype().into(),
            data: self.data(),
        }
    }
}

// Download: Cxx -> Rust
impl Into<RDtype> for Dtype {
    fn into(self) -> RDtype {
        match self {
            Dtype::BOOL => RDtype::BOOL,
            Dtype::F4 => RDtype::F4,
            Dtype::F6_E2M3 => RDtype::F6_E2M3,
            Dtype::F6_E3M2 => RDtype::F6_E3M2,
            Dtype::U8 => RDtype::U8,
            Dtype::I8 => RDtype::I8,
            Dtype::F8_E5M2 => RDtype::F8_E5M2,
            Dtype::F8_E4M3 => RDtype::F8_E4M3,
            Dtype::F8_E8M0 => RDtype::F8_E8M0,
            Dtype::I16 => RDtype::I16,
            Dtype::U16 => RDtype::U16,
            Dtype::F16 => RDtype::F16,
            Dtype::BF16 => RDtype::BF16,
            Dtype::I32 => RDtype::I32,
            Dtype::U32 => RDtype::U32,
            Dtype::F32 => RDtype::F32,
            Dtype::F64 => RDtype::F64,
            Dtype::I64 => RDtype::I64,
            Dtype::U64 => RDtype::U64,
            _ => todo!(),
        }
    }
}
