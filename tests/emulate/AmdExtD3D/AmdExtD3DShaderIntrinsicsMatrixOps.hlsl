// VeyAmux-FSR4 test build: AMD's WaveMatrix API emulated in plain HLSL (no cooperative matrices), so the FP8 model,
// its dispatches, bindings and padding passes can be checked on any Vulkan device (lavapipe in CI).
// Copyright (c) 2026 VeyAmux contributors. MIT licence (see NOTICE.md).
//
// Every 16x16 matrix is spread over the 32 invocations of the group: invocation l holds column l % 16, rows
// 8 * (l / 16) + i for i = 0..7, for every use (so CopySat between uses needs no data movement). Values are kept as
// float; FP8 and F16 values are rounded to their format whenever a matrix of that format is written, so the numbers
// match what the hardware computes up to the order of the additions. Multiply-add goes through shared memory.
// Slow, and only meant for tests (tests/run_emulated.sh).
#ifndef VEYAMUX_WMMA_EMULATED_HLSL
#define VEYAMUX_WMMA_EMULATED_HLSL

#define AmdExtD3DShaderIntrinsicsWaveMatrixType_A             0x0
#define AmdExtD3DShaderIntrinsicsWaveMatrixType_B             0x1
#define AmdExtD3DShaderIntrinsicsWaveMatrixType_Accumulator   0x2
#define AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16   0x4
#define AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F32   0x6
#define AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_FP8   0xa

#ifndef WMMA_SRV_BINDING
#error WMMA_SRV_BINDING must be set
#endif
[[vk::binding(WMMA_SRV_BINDING, 0)]] StructuredBuffer<uint> WMMA_SRV;
[[vk::binding(WMMA_UAV_BINDING, 0)]] RWStructuredBuffer<uint> WMMA_UAV;

namespace emu {

// e4m3 ("fn": no infinities, 0x7f/0xff NaN, largest 448)
float decode8(uint b) {
    const float s = (b & 0x80u) ? -1.0f : 1.0f;
    const uint e = (b >> 3) & 15u, m = b & 7u;
    if (e == 0) return s * float(m) * (1.0f / 512.0f);      // subnormal: m * 2^-9
    if (e == 15 && m == 7) return asfloat(0x7fc00000u);
    return s * (1.0f + float(m) / 8.0f) * exp2(float(e) - 7.0f);
}
// round to nearest even, clamped to +-448 (AMD's saturating conversion)
uint encode8(float x) {
    if (isnan(x)) return 0x7fu;
    const uint sign = x < 0 ? 0x80u : 0u;
    float a = min(abs(x), 448.0f);
    if (a < exp2(-6.0f)) {
        const uint q = uint(round(a * 512.0f));               // 0..8; 8 is the smallest normal (0x08)
        return sign | q;
    }
    int e = int(floor(log2(a)));
    float f = a / exp2(float(e));
    if (f >= 2.0f) { f *= 0.5f; e += 1; }                   // log2 rounding at powers of two
    if (f < 1.0f) { f *= 2.0f; e -= 1; }
    uint m = uint(round((f - 1.0f) * 8.0f));
    if (m == 8) { m = 0; e += 1; }
    const uint eb = uint(e + 7);
    uint v = (eb << 3) | m;
    if (v > 0x7eu) v = 0x7eu;                                // 448
    return sign | v;
}
float q8(float x) { return decode8(encode8(x)); }
float q16(float x) { return f16tof32(f32tof16(x)); }

[[vk::ext_builtin_input(/*LocalInvocationIndex*/ 29)]] static const uint localIndex;
uint lane() { return localIndex; }

groupshared float mA[256];
groupshared float mB[256];

uint rowOf(uint i) { return 8 * (lane() / 16) + i; }
uint colOf() { return lane() % 16; }

uint byteAddr(uint off, uint strideBytes, bool colMajor, uint r, uint c, uint esize) {
    return colMajor ? off + c * strideBytes + r * esize : off + r * strideBytes + c * esize;
}
float readElem(uint word, uint byteInWord, uint fmt) {
    if (fmt == AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_FP8) return decode8((word >> (8 * byteInWord)) & 0xffu);
    if (fmt == AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16) return f16tof32((word >> (8 * byteInWord)) & 0xffffu);
    return asfloat(word);
}
uint esizeOf(uint fmt) {
    return fmt == AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_FP8 ? 1 : fmt == AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16 ? 2 : 4;
}
uint bitsOf(float v, uint fmt) {
    if (fmt == AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_FP8) return encode8(v);
    if (fmt == AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16) return f32tof16(v);
    return asuint(v);
}
float quant(float v, uint fmt) {
    if (fmt == AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_FP8) return q8(v);
    if (fmt == AmdExtD3DShaderIntrinsicsWaveMatrixDataFormat_F16) return q16(v);
    return v;
}

} // namespace emu

template <uint matrixDataFmt, uint DIMM, uint DIMN, uint matrixType>
struct AmdWaveMatrix {
    float v[8];
    uint Fmt() { return matrixDataFmt; }

    void Load(ByteAddressBuffer srv, uint off, uint stride, bool colMajor = false) {
        const uint es = emu::esizeOf(matrixDataFmt);
        [unroll] for (uint i = 0; i < 8; ++i) {
            const uint a = emu::byteAddr(off, stride, colMajor, emu::rowOf(i), emu::colOf(), es);
            v[i] = emu::readElem(WMMA_SRV[a >> 2], a & 3, matrixDataFmt);
        }
    }
    void Load(RWByteAddressBuffer uav, uint off, uint stride, bool colMajor = false) {
        const uint es = emu::esizeOf(matrixDataFmt);
        [unroll] for (uint i = 0; i < 8; ++i) {
            const uint a = emu::byteAddr(off, stride, colMajor, emu::rowOf(i), emu::colOf(), es);
            v[i] = emu::readElem(WMMA_UAV[a >> 2], a & 3, matrixDataFmt);
        }
    }
    void Store(RWByteAddressBuffer uav, uint off, uint stride, bool colMajor = false) {
        const uint es = emu::esizeOf(matrixDataFmt);
        [unroll] for (uint i = 0; i < 8; ++i) {
            const uint a = emu::byteAddr(off, stride, colMajor, emu::rowOf(i), emu::colOf(), es);
            if (es == 4) { WMMA_UAV[a >> 2] = asuint(v[i]); continue; }
            const uint shift = 8 * (a & 3), mask = (es == 1 ? 0xffu : 0xffffu) << shift;
            uint old;
            InterlockedAnd(WMMA_UAV[a >> 2], ~mask, old);
            InterlockedOr(WMMA_UAV[a >> 2], (emu::bitsOf(v[i], matrixDataFmt) << shift) & mask, old);
        }
    }
    uint Length() { return 8; }
    void Fill(float data) { [unroll] for (uint i = 0; i < 8; ++i) v[i] = emu::quant(data, matrixDataFmt); }
    float Element(uint i) { return v[i]; }
    void SetElement(uint i, float data) { v[i] = emu::quant(data, matrixDataFmt); }
    template <uint srcFmt, uint srcType>
    void Copy(AmdWaveMatrix<srcFmt, DIMM, DIMN, srcType> input) {
        [unroll] for (uint i = 0; i < 8; ++i) v[i] = emu::quant(input.v[i], matrixDataFmt);
    }
    template <uint srcFmt, uint srcType>
    void CopySat(AmdWaveMatrix<srcFmt, DIMM, DIMN, srcType> input) {
        [unroll] for (uint i = 0; i < 8; ++i) v[i] = emu::quant(input.v[i], matrixDataFmt);   // encode8 saturates
    }
};

template <uint fmt, uint M, uint N> using AmdWaveMatrixA = AmdWaveMatrix<fmt, M, N, AmdExtD3DShaderIntrinsicsWaveMatrixType_A>;
template <uint fmt, uint M, uint N> using AmdWaveMatrixB = AmdWaveMatrix<fmt, M, N, AmdExtD3DShaderIntrinsicsWaveMatrixType_B>;
template <uint fmt, uint M, uint N> using AmdWaveMatrixAccumulator = AmdWaveMatrix<fmt, M, N, AmdExtD3DShaderIntrinsicsWaveMatrixType_Accumulator>;

template <uint fa, uint fb, uint fc, uint M, uint N, uint K>
AmdWaveMatrixAccumulator<fc, M, N> AmdWaveMatrixMultiply(AmdWaveMatrixA<fa, M, K> a, AmdWaveMatrixB<fb, K, N> b,
                                                         AmdWaveMatrixAccumulator<fc, M, N> c) {
    [unroll] for (uint i = 0; i < 8; ++i) {
        emu::mA[emu::rowOf(i) * 16 + emu::colOf()] = a.v[i];   // A[r][k], k = this column
        emu::mB[emu::rowOf(i) * 16 + emu::colOf()] = b.v[i];   // B[k][n], k = this row
    }
    GroupMemoryBarrierWithGroupSync();
    AmdWaveMatrixAccumulator<fc, M, N> r;
    [unroll] for (uint i = 0; i < 8; ++i) {
        float s = c.v[i];
        for (uint k = 0; k < 16; ++k) s += emu::mA[emu::rowOf(i) * 16 + k] * emu::mB[k * 16 + emu::colOf()];
        r.v[i] = emu::quant(s, fc);
    }
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Shared memory (uint arrays): offset and stride in uints
#define AMD_GROUPSHARED_LOAD(mo, gs, offEl, strideEl, colMajor) \
    GroupMemoryBarrierWithGroupSync(); \
    [unroll] for (uint _i = 0; _i < 8; ++_i) { \
        const uint _a = emu::byteAddr((offEl) * 4, (strideEl) * 4, (colMajor), emu::rowOf(_i), emu::colOf(), emu::esizeOf(mo.Fmt())); \
        mo.v[_i] = emu::readElem((gs)[_a >> 2], _a & 3, mo.Fmt()); \
    }
#define AMD_GROUPSHARED_STORE(mo, gs, offEl, strideEl, colMajor) \
    GroupMemoryBarrierWithGroupSync(); \
    [unroll] for (uint _i = 0; _i < 8; ++_i) { \
        const uint _a = emu::byteAddr((offEl) * 4, (strideEl) * 4, (colMajor), emu::rowOf(_i), emu::colOf(), emu::esizeOf(mo.Fmt())); \
        const uint _es = emu::esizeOf(mo.Fmt()); \
        if (_es == 4) (gs)[_a >> 2] = asuint(mo.v[_i]); \
        else { const uint _sh = 8 * (_a & 3), _mk = (_es == 1 ? 0xffu : 0xffffu) << _sh; uint _o; \
               InterlockedAnd((gs)[_a >> 2], ~_mk, _o); InterlockedOr((gs)[_a >> 2], (emu::bitsOf(mo.v[_i], mo.Fmt()) << _sh) & _mk, _o); } \
    } \
    GroupMemoryBarrierWithGroupSync();

#endif
