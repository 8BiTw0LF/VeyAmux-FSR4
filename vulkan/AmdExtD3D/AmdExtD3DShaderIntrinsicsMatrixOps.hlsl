// VeyAmux-FSR4: AMD's WaveMatrix (WMMA) shader API for Vulkan, on VK_KHR_cooperative_matrix.
// Copyright (c) 2026 VeyAmux contributors. MIT licence (see NOTICE.md).
//
// AMD's FP8 model code (ml2code_runtime/operators/float8_NHWC, pre_wmma.hlsl, post_wmma.hlsl) is written against the
// AmdWaveMatrix* classes of AMD's D3D12 driver extension. This file gives the same classes and macros, built on
// standard SPIR-V instead (DXC inline SPIR-V):
//   SPV_KHR_cooperative_matrix                load, store, multiply-add, length, element access
//   SPV_EXT_float8                            E4M3 components (Float8EXT, Float8CooperativeMatrixEXT)
//   SPV_EXT_cooperative_matrix_maintenance1   accumulator -> A/B use conversion (OpCooperativeMatrixConvertUseEXT)
// Only the parts of AMD's API the FSR 4 model uses are here: 16x16 matrices, FP8 (E4M3) / F16 / F32 components.
//
// Two differences from AMD's driver that the code below handles:
// * Vulkan has no FP8 accumulator type. An "FP8 accumulator" (AMD loads residuals and stores outputs through it) is
//   kept as an FP8 B matrix: the same 16x16 values, and the same memory layout when loaded or stored. Turning it into
//   an F32 accumulator is an exact multiply with the identity matrix.
// * The model's buffers are ByteAddressBuffers, which have no element a SPIR-V pointer can point at. Matrix loads and
//   stores go through StructuredBuffer<uint> aliases of the same descriptors instead: WMMA_SRV for read-only buffers
//   (the model's weights, InitializerBuffer) and WMMA_UAV for read-write ones (ScratchBuffer). build.sh sets the
//   bindings per shader and checks that these are the only buffers the matrix code touches.
#ifndef VEYAMUX_WMMA_HLSL
#define VEYAMUX_WMMA_HLSL

#include <vk/spirv.h>

#define AmdExtD3DShaderIntrinsicsWaveMatrixType_A             0x0
#define AmdExtD3DShaderIntrinsicsWaveMatrixType_B             0x1
#define AmdExtD3DShaderIntrinsicsWaveMatrixType_Accumulator   0x2

#define AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I8    0x2
#define AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_U8    0x3
#define AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16   0x4
#define AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F32   0x6
#define AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_I32   0x7
#define AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_FP8   0xa

#ifndef WMMA_SRV_BINDING
#error WMMA_SRV_BINDING (the weights buffer's binding) must be set
#endif
#ifndef WMMA_UAV_BINDING
#error WMMA_UAV_BINDING (the scratch buffer's binding) must be set
#endif
[[vk::binding(WMMA_SRV_BINDING, 0)]] StructuredBuffer<uint> WMMA_SRV;
[[vk::binding(WMMA_UAV_BINDING, 0)]] RWStructuredBuffer<uint> WMMA_UAV;

namespace wmma {

// ---- SPIR-V types
// DXC's SPIR-V optimizer does not know SPV_EXT_float8 yet and skips legalization when it sees it, so the shaders are
// compiled with a stand-in component type, a signed 8-bit integer (which nothing else in the model uses), and
// tools/spv_float8.py then turns it into Float8E4M3EXT: the type, the conversions to it (OpConvertFToS here becomes
// OpFConvert) and the capabilities. It refuses a module where the stand-in is used for anything else.
using f8 = vk::SpirvType</*OpTypeInt*/ 21, 1, 1, vk::Literal<vk::integral_constant<uint, 8> >,
                         vk::Literal<vk::integral_constant<uint, 1> > >;

template <typename T, uint Use>
using Mat = vk::SpirvOpaqueType</*OpTypeCooperativeMatrixKHR*/ 4456, T, vk::integral_constant<uint, /*Subgroup*/ 3>,
                                vk::integral_constant<uint, 16>, vk::integral_constant<uint, 16>, vk::integral_constant<uint, Use> >;

static const uint UseA = 0, UseB = 1, UseAcc = 2;

typedef Mat<f8, UseA> A8;
typedef Mat<f8, UseB> B8;
typedef Mat<half, UseA> A16;
typedef Mat<half, UseB> B16;
typedef Mat<half, UseAcc> C16;
typedef Mat<float, UseAcc> C32;

// AMD (format, use) -> the Vulkan component type and use. FP8 accumulators are B matrices (see the top).
template <uint fmt> struct Comp;
template <> struct Comp<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_FP8> { using T = f8; using E = uint; };
template <> struct Comp<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16> { using T = half; using E = half; };
template <> struct Comp<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F32> { using T = float; using E = float; };
template <uint fmt, uint type> struct Use { static const uint v = type; };
template <> struct Use<AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_FP8, AmdExtD3DShaderIntrinsicsWaveMatrixType_Accumulator> { static const uint v = 1; };

// ---- instructions
template <typename R, typename P>
[[vk::ext_capability(/*CooperativeMatrixKHR*/ 6022)]] [[vk::ext_capability(/*VulkanMemoryModel*/ 5345)]]
[[vk::ext_capability(/*Int8*/ 39)]] [[vk::ext_capability(/*Float16*/ 9)]]
[[vk::ext_extension("SPV_KHR_cooperative_matrix")]]
[[vk::ext_instruction(/*OpCooperativeMatrixLoadKHR*/ 4457)]]
R load([[vk::ext_reference]] P pointer, uint layout, uint stride);

template <typename R, typename P>
[[vk::ext_instruction(/*OpCooperativeMatrixLoadKHR*/ 4457)]]
R loadShared(vk::WorkgroupSpirvPointer<P> pointer, uint layout, uint stride);

template <typename P, typename M>
[[vk::ext_instruction(/*OpCooperativeMatrixStoreKHR*/ 4458)]]
void store([[vk::ext_reference]] P pointer, M m, uint layout, uint stride);

template <typename P, typename M>
[[vk::ext_instruction(/*OpCooperativeMatrixStoreKHR*/ 4458)]]
void storeShared(vk::WorkgroupSpirvPointer<P> pointer, M m, uint layout, uint stride);

template <typename C, typename A, typename B>
[[vk::ext_instruction(/*OpCooperativeMatrixMulAddKHR*/ 4459)]]
C mulAdd(A a, B b, C c);

template <typename M, typename T>
[[vk::ext_instruction(/*OpCompositeConstruct*/ 80)]]
M splat(T v);

template <typename R, typename M>
[[vk::ext_capability(/*Float16*/ 9)]]
[[vk::ext_instruction(/*OpFConvert*/ 115)]]
R fconvert(M m);

// F16 -> the 8-bit stand-in; spv_float8.py makes it OpFConvert (to E4M3, round to nearest even)
template <typename R, typename M>
[[vk::ext_instruction(/*OpConvertFToS*/ 110)]]
R toF8(M m);

// Accumulator -> A/B. Written as OpFConvert between the two uses (DXC cannot take the real instruction yet);
// spv_float8.py makes it OpCooperativeMatrixConvertUseEXT.
template <typename R, typename M>
[[vk::ext_capability(/*Float16*/ 9)]]
[[vk::ext_instruction(/*OpFConvert*/ 115)]]
R convertUse(M m);

template <class ResultPointerType, class BaseType>
[[vk::ext_instruction(/*OpAccessChain*/ 65)]]
ResultPointerType chain([[vk::ext_reference]] BaseType base, uint index);

template <class ObjectType, class PointerType>
[[vk::ext_instruction(/*OpLoad*/ 61)]]
ObjectType deref(PointerType p);

template <class PointerType, class ObjectType>
[[vk::ext_instruction(/*OpStore*/ 62)]]
void assign(PointerType p, ObjectType v);

// DXC expands this builtin (OpCooperativeMatrixLengthKHR takes a type operand)
template <class MatrixType> uint __builtin_spv_CooperativeMatrixLengthKHR();

template <typename T>
using FunctionPtr = vk::SpirvOpaqueType</*OpTypePointer*/ 32, vk::Literal<vk::integral_constant<uint, /*Function*/ 7> >, T>;

// e4m3's largest finite value: AMD's "Sat" conversions clamp to it
static const float kF8Max = 448.0f;

// The identity as an FP8 A matrix (exact products and sums), from shared memory. All 32 lanes of the (one-wave)
// group take part, so callers must be in uniform control flow, which AMD's model code is.
groupshared uint identityLds[64];
A8 identity8() {
    const uint lane = WaveGetLaneIndex();
    [unroll] for (uint j = 0; j < 2; ++j) {
        const uint i = lane * 2 + j;            // row-major, 4 bytes per uint
        const uint row = i / 4, col0 = (i % 4) * 4;
        identityLds[i] = (row >= col0 && row < col0 + 4) ? (0x38u << (8 * (row - col0))) : 0u;   // e4m3 1.0 = 0x38
    }
    GroupMemoryBarrierWithGroupSync();
    A8 r = loadShared<A8>(vk::GetGroupSharedAddress(identityLds[0]), /*RowMajor*/ 0, 4);
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// F32 accumulator -> FP8 B, clamped to e4m3's range (AMD's CopySat). Through F16: F32 -> F16 (exact after the clamp
// up to F16's precision), accumulator -> B, F16 -> E4M3 (round to nearest even).
B8 toB8(C32 c) {
    const uint n = __builtin_spv_CooperativeMatrixLengthKHR<C32>();
    for (uint i = 0; i < n; ++i) {
        FunctionPtr<float> p = chain<FunctionPtr<float> >(c, i);
        assign(p, clamp(deref<float>(p), -kF8Max, kF8Max));
    }
    return toF8<B8>(convertUse<B16>(fconvert<C16>(c)));
}

C32 fromB8(B8 b) {
    return mulAdd(identity8(), b, splat<C32>(0.0f));
}

// Copy / CopySat by (destination, source) type
void copy(out C32 d, C32 s) { d = s; }
void copy(out C32 d, B8 s) { d = fromB8(s); }
void copy(out B8 d, C32 s) { d = toB8(s); }
void copy(out B8 d, B8 s) { d = s; }
void copy(out A8 d, A8 s) { d = s; }
void copy(out C16 d, C16 s) { d = s; }
void copy(out C16 d, C32 s) { d = fconvert<C16>(s); }
void copy(out C32 d, C16 s) { d = fconvert<C32>(s); }

// Fill. AMD's model only ever fills FP8 matrices with 0 (packed: four zeros), which is the only value used here.
void fill(out C32 m, float v) { m = splat<C32>(v); }
void fill(out C16 m, half v) { m = splat<C16>(v); }
void fill(out A16 m, half v) { m = splat<A16>(v); }
void fill(out B16 m, half v) { m = splat<B16>(v); }
void fill(out A8 m, uint v) { m = toF8<A8>(splat<A16>(half(0))); }
void fill(out B8 m, uint v) { m = toF8<B8>(splat<B16>(half(0))); }

} // namespace wmma

template <uint matrixDataFmt, uint DIMM, uint DIMN, uint matrixType>
struct AmdWaveMatrix {
    using T = typename wmma::Comp<matrixDataFmt>::T;
    using elementType = typename wmma::Comp<matrixDataFmt>::E;
    using M = wmma::Mat<T, wmma::Use<matrixDataFmt, matrixType>::v>;
    M m;

    // AMD: offsets and strides in bytes, bColMajor = column-major. SPIR-V: a uint element and a stride in uints.
    void Load(ByteAddressBuffer srv, uint startOffsetInBytes, uint strideInBytes, bool bColMajor = false) {
        m = wmma::load<M>(WMMA_SRV[startOffsetInBytes >> 2], bColMajor ? 1 : 0, strideInBytes >> 2);
    }
    void Load(RWByteAddressBuffer uav, uint startOffsetInBytes, uint strideInBytes, bool bColMajor = false) {
        m = wmma::load<M>(WMMA_UAV[startOffsetInBytes >> 2], bColMajor ? 1 : 0, strideInBytes >> 2);
    }
    void Store(RWByteAddressBuffer uav, uint startOffsetInBytes, uint strideInBytes, bool bColMajor = false) {
        wmma::store(WMMA_UAV[startOffsetInBytes >> 2], m, bColMajor ? 1 : 0, strideInBytes >> 2);
    }
    void LoadShared(vk::WorkgroupSpirvPointer<uint> p, uint strideInElements, bool bColMajor) {
        m = wmma::loadShared<M>(p, bColMajor ? 1 : 0, strideInElements);
    }
    void StoreShared(vk::WorkgroupSpirvPointer<uint> p, uint strideInElements, bool bColMajor) {
        wmma::storeShared(p, m, bColMajor ? 1 : 0, strideInElements);
    }

    uint Length() {
        return wmma::__builtin_spv_CooperativeMatrixLengthKHR<M>();
    }
    void Fill(elementType data) { wmma::fill(m, data); }
    elementType Element(uint i) {
        wmma::FunctionPtr<T> p = wmma::chain<wmma::FunctionPtr<T> >(m, i);
        return wmma::deref<T>(p);
    }
    void SetElement(uint i, elementType data) {
        wmma::FunctionPtr<T> p = wmma::chain<wmma::FunctionPtr<T> >(m, i);
        wmma::assign(p, T(data));
    }
    template <uint srcMatrixDataFmt, uint srcMatrixType>
    void Copy(AmdWaveMatrix<srcMatrixDataFmt, DIMM, DIMN, srcMatrixType> input) { wmma::copy(m, input.m); }
    template <uint srcMatrixDataFmt, uint srcMatrixType>
    void CopySat(AmdWaveMatrix<srcMatrixDataFmt, DIMM, DIMN, srcMatrixType> input) { wmma::copy(m, input.m); }
};

template <uint matrixDataFmt, uint DIMM, uint DIMN>
using AmdWaveMatrixA = AmdWaveMatrix<matrixDataFmt, DIMM, DIMN, AmdExtD3DShaderIntrinsicsWaveMatrixType_A>;
template <uint matrixDataFmt, uint DIMM, uint DIMN>
using AmdWaveMatrixB = AmdWaveMatrix<matrixDataFmt, DIMM, DIMN, AmdExtD3DShaderIntrinsicsWaveMatrixType_B>;
template <uint matrixDataFmt, uint DIMM, uint DIMN>
using AmdWaveMatrixAccumulator = AmdWaveMatrix<matrixDataFmt, DIMM, DIMN, AmdExtD3DShaderIntrinsicsWaveMatrixType_Accumulator>;

template <uint matrixDataFmtA, uint matrixDataFmtB, uint matrixDataFmtAccumulator, uint DIMM, uint DIMN, uint DIMK>
AmdWaveMatrixAccumulator<matrixDataFmtAccumulator, DIMM, DIMN> AmdWaveMatrixMultiply(
    AmdWaveMatrixA<matrixDataFmtA, DIMM, DIMK> matA, AmdWaveMatrixB<matrixDataFmtB, DIMK, DIMN> matB,
    AmdWaveMatrixAccumulator<matrixDataFmtAccumulator, DIMM, DIMN> matAccumulator) {
    AmdWaveMatrixAccumulator<matrixDataFmtAccumulator, DIMM, DIMN> r;
    r.m = wmma::mulAdd(matA.m, matB.m, matAccumulator.m);
    return r;
}

// Shared memory: offset and stride in elements of the groupshared uint array. AMD's code writes shared memory with
// plain stores right before a matrix load from it, and reads a matrix store back right after, with no barrier: on
// AMD's driver one group is one wave in lockstep. Vulkan's memory model needs a barrier for that, so both macros
// have them, before and after (the group is one wave and the code is in uniform control flow, so it costs little).
#define AMD_GROUPSHARED_LOAD(matrix_object, gsArray, offsetInArrayElements, strideInElements, bColMajor) \
    do { GroupMemoryBarrierWithGroupSync(); \
         (matrix_object).LoadShared(vk::GetGroupSharedAddress((gsArray)[(offsetInArrayElements)]), (strideInElements), (bColMajor)); } while (0)
#define AMD_GROUPSHARED_STORE(matrix_object, gsArray, offsetInArrayElements, strideInElements, bColMajor) \
    do { GroupMemoryBarrierWithGroupSync(); (matrix_object).StoreShared(vk::GetGroupSharedAddress((gsArray)[(offsetInArrayElements)]), (strideInElements), (bColMajor)); \
         GroupMemoryBarrierWithGroupSync(); } while (0)

#endif // VEYAMUX_WMMA_HLSL
