#!/usr/bin/env python3
# VeyAmux-FSR4: finish the FP8 (WMMA) shaders after DXC.
#
# DXC's SPIR-V optimizer does not know SPV_EXT_float8 or SPV_EXT_cooperative_matrix_maintenance1 yet: with either in
# the module it skips legalization, and its validator rejects a cooperative matrix use conversion. So
# vulkan/AmdExtD3D/AmdExtD3DShaderIntrinsicsMatrixOps.hlsl writes two stand-ins, compiled with -Vd, which this script
# replaces:
#   * the component type "signed 8-bit integer" (OpTypeInt 8 1)  ->  OpTypeFloat 8 Float8E4M3EXT
#   * OpConvertFToS to a matrix of that type                        ->  OpFConvert (to E4M3)
#   * OpFConvert between matrices of different uses                 ->  OpCooperativeMatrixConvertUseEXT
# and declares Float8EXT, Float8CooperativeMatrixEXT and CooperativeMatrixConversionsEXT with their extensions.
# It stops with an error when the stand-in type is used for anything else, so a wrong module is never written.
#   spv_float8.py in.spv out.spv
import struct
import sys

OpExtension, OpCapability, OpTypeInt, OpTypeFloat = 10, 17, 21, 22
OpConstant, OpConstantComposite, OpConstantNull = 43, 44, 46
OpConvertFToS, OpFConvert = 110, 115
OpTypeCooperativeMatrixKHR, OpCooperativeMatrixConvertUseEXT = 4456, 5293
Float8E4M3EXT = 4214
CAPS = [4212, 4213, 5429]   # Float8EXT, Float8CooperativeMatrixEXT, CooperativeMatrixConversionsEXT
EXTS = ["SPV_EXT_float8", "SPV_EXT_cooperative_matrix_maintenance1"]


def fail(msg):
    sys.exit("spv_float8: " + msg)


def string_words(s):
    b = s.encode() + b"\0"
    b += b"\0" * (-len(b) % 4)
    return list(struct.unpack("<%dI" % (len(b) // 4), b))


def main(src, dst):
    data = open(src, "rb").read()
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    if words[0] != 0x07230203:
        fail("not SPIR-V")
    header, insts = words[:5], []
    i = 5
    while i < len(words):
        n = words[i] >> 16
        if n == 0:
            fail("bad instruction length")
        insts.append(words[i:i + n])
        i += n

    def op(w):
        return w[0] & 0xffff

    # the stand-in type, and every cooperative matrix type: id -> (component, use id)
    i8 = [w[1] for w in insts if op(w) == OpTypeInt and w[2] == 8 and w[3] == 1]
    if not i8:   # no FP8 matrices here (the padding passes): nothing to do
        mat_ids = {x[1] for x in insts if op(x) == OpTypeCooperativeMatrixKHR}
        if any(op(w) == OpFConvert and w[1] in mat_ids for w in insts):
            fail("matrix conversions without the FP8 stand-in")
        open(dst, "wb").write(data)
        return
    if len(i8) != 1:
        fail("expected one 8-bit signed integer type (the FP8 stand-in), found %d" % len(i8))
    i8 = i8[0]
    consts = {w[2]: w[3] for w in insts if op(w) == OpConstant and len(w) == 4}
    mats = {w[1]: (w[2], consts.get(w[6])) for w in insts if op(w) == OpTypeCooperativeMatrixKHR}
    value_types = {}   # result id -> type id
    # where may the stand-in appear? As a matrix component, and as a zero constant (an FP8 matrix filled with 0)
    zero_i8 = set()
    out = []
    convert_use = to_f8 = 0
    for w in insts:
        o = op(w)
        if o == OpTypeInt and w[1] == i8:
            w = [(4 << 16) | OpTypeFloat, i8, 8, Float8E4M3EXT]
        elif o == OpConstant and w[1] == i8:
            if w[3] != 0:
                fail("the FP8 stand-in has a non-zero constant")
            zero_i8.add(w[2])     # integer 0 and E4M3 +0 have the same bits
        elif o == OpTypeCooperativeMatrixKHR:
            pass
        elif o == OpConvertFToS and w[1] in mats and mats[w[1]][0] == i8:
            w = [(w[0] & 0xffff0000) | OpFConvert] + w[1:]
            to_f8 += 1
        elif o == OpFConvert and w[1] in mats:
            src_type = value_types.get(w[3])
            if src_type in mats and mats[src_type][1] != mats[w[1]][1]:
                if mats[src_type][0] != mats[w[1]][0]:
                    fail("a use conversion that also changes the component type")
                w = [(w[0] & 0xffff0000) | OpCooperativeMatrixConvertUseEXT] + w[1:]
                convert_use += 1
        elif o == 32 and w[3] == i8:   # OpTypePointer to a scalar of the stand-in
            fail("a pointer to the FP8 stand-in")
        elif o >= 43 and len(w) >= 3 and w[1] == i8:
            fail("the FP8 stand-in type is the result of opcode %d" % o)
        # record result types (instructions with <result type> <result id>)
        if len(w) >= 3 and o not in (OpTypeInt, OpTypeFloat, OpTypeCooperativeMatrixKHR, OpCapability, OpExtension) and o >= 43:
            value_types[w[2]] = w[1]
        out.append(w)

    # Int8 was only there for the stand-in when no other 8-bit integer type is left
    if not any(op(w) == OpTypeInt and w[2] == 8 for w in out):
        out = [w for w in out if not (op(w) == OpCapability and w[1] == 39)]
    # capabilities and extensions after the existing ones
    caps = {w[1] for w in out if op(w) == OpCapability}
    exts = set()
    for w in out:
        if op(w) == OpExtension:
            b = struct.pack("<%dI" % (len(w) - 1), *w[1:])
            exts.add(b.split(b"\0")[0].decode())
    first_ext = next(k for k, w in enumerate(out) if op(w) not in (OpCapability,))
    add = [[(2 << 16) | OpCapability, c] for c in CAPS if c not in caps]
    for e in EXTS:
        if e not in exts:
            sw = string_words(e)
            add.append([((1 + len(sw)) << 16) | OpExtension] + sw)
    out = out[:first_ext] + add + out[first_ext:]

    flat = header[:]
    for w in out:
        flat += w
    open(dst, "wb").write(struct.pack("<%dI" % len(flat), *flat))
    print("%s: %d FP8 conversions, %d use conversions" % (dst, to_f8, convert_use))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: spv_float8.py in.spv out.spv")
    main(sys.argv[1], sys.argv[2])
