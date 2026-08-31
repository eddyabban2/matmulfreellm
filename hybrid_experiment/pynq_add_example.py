#!/usr/bin/env python3
"""Minimal PYNQ example: one vector add on the ternip accelerator.

Self-contained -- only pynq and numpy are needed. The kernel configuration and
the instruction encoding are inlined, so nothing from the ternary_matmul repo
has to be copied alongside this file.

Exercises the whole host<->card path:

  PCIe : pynq.Overlay programs the card, and buffers sync over XDMA
  DDR  : the input/output vectors and the instruction stream live in a DDR bank
  DMA  : the AXI DMA streams instruction words into s_axis_instruction, and the
         kernel's loadstore master reads/writes the vectors in DDR

The program is five instructions:

    ldv v0, A        ; DDR -> vector register 0
    ldv v1, B        ; DDR -> vector register 1
    add v2, v0, v1   ; v2 = v0 + v1
    sv  v2, C        ; vector register 2 -> DDR
    stall            ; hand control back to the host

Usage:
    python3 pynq_add_example.py [kernel.xclbin]
"""
import sys

import numpy as np
import pynq

XCLBIN = sys.argv[1] if len(sys.argv) > 1 else 'kernel.xclbin'

# Must match the xclbin: the ternip_big kernel of the BS=9 heterogeneous AU250
# build.
VECTOR_LENGTH = 1024              # elements per lane
BATCH_SIZE = 9                    # independent lanes per vector
FIXED_POINT_PRECISION = 16        # bits per element
FIXED_POINT_EXPONENT = -5         # value = raw * 2**exponent
NUM_VECTOR_REGISTERS = 4
INSTRUCTION_WIDTH = 128
DDR_ADDRESS_WIDTH = 64
DDR_BANK = 0                      # compute unit 0 is pinned to SLR/bank 0

ELEMENT_BYTES = FIXED_POINT_PRECISION // 8
VECTOR_BYTES = BATCH_SIZE * VECTOR_LENGTH * ELEMENT_BYTES
VECTOR_SHAPE = (BATCH_SIZE, VECTOR_LENGTH)
LEAST_SIGNIFICANT_BIT = 2.0 ** FIXED_POINT_EXPONENT

# Control registers, at raw offsets within one compute unit's mmio.
DMA_CONTROL = 0x0000
DMA_SOURCE_ADDRESS = 0x0018
DMA_TRANSFER_LENGTH = 0x0028
STALL = 0x1000
RESET = 0x2000

# An instruction is this bit string, left-zero-padded to INSTRUCTION_WIDTH and
# emitted least-significant byte first:
#
#   functional_unit(3) rowwise_operation(4) register_a register_b
#   register_destination loadstore_operation(2) tmatmul(2) rms(3)
#   address(DDR_ADDRESS_WIDTH)
#
# Each register field is ceil(log2(NUM_VECTOR_REGISTERS)) bits wide.
REGISTER_WIDTH = max(1, (NUM_VECTOR_REGISTERS - 1).bit_length())
FUNCTIONAL_UNIT = {'loadstore': '001', 'rowwise': '010', 'stall': '101'}
ROWWISE_OPERATION = {'none': '0000', 'add': '0001'}
LOADSTORE_OPERATION = {'none': '00', 'ldv': '01', 'sv': '10'}
NO_TMATMUL_OPERATION = '00'
NO_RMS_OPERATION = '000'


def encode_instruction(functional_unit, rowwise_operation, register_a,
                       register_b, register_destination, loadstore_operation,
                       address):
    bits = (FUNCTIONAL_UNIT[functional_unit]
            + ROWWISE_OPERATION[rowwise_operation]
            + format(register_a, f'0{REGISTER_WIDTH}b')
            + format(register_b, f'0{REGISTER_WIDTH}b')
            + format(register_destination, f'0{REGISTER_WIDTH}b')
            + LOADSTORE_OPERATION[loadstore_operation]
            + NO_TMATMUL_OPERATION
            + NO_RMS_OPERATION
            + format(address, f'0{DDR_ADDRESS_WIDTH}b')).zfill(INSTRUCTION_WIDTH)
    return bytes(int(bits[i:i + 8], 2)
                 for i in range(0, INSTRUCTION_WIDTH, 8))[::-1]


def load_vector(register, address):
    return encode_instruction('loadstore', 'none', register, register, register,
                              'ldv', address)


def store_vector(register, address):
    return encode_instruction('loadstore', 'none', register, register, register,
                              'sv', address)


def add_vectors(destination, source_a, source_b):
    return encode_instruction('rowwise', 'add', source_a, source_b, destination,
                              'none', 0)


def stall():
    return encode_instruction('stall', 'none', 0, 0, 0, 'none', 0)


def encode_fixed_point(values):
    return np.round(values / LEAST_SIGNIFICANT_BIT).astype(f'<i{ELEMENT_BYTES}')


def decode_fixed_point(raw_bytes):
    return (np.frombuffer(raw_bytes, dtype=f'<i{ELEMENT_BYTES}')
            .reshape(VECTOR_SHAPE).astype(np.float64) * LEAST_SIGNIFICANT_BIT)


def main():
    print(f'D={VECTOR_LENGTH} BatchSize={BATCH_SIZE} '
          f'fixed-point={FIXED_POINT_PRECISION}b exp={FIXED_POINT_EXPONENT} '
          f'({VECTOR_BYTES} bytes/vector)')

    print(f'Loading {XCLBIN} ...', flush=True)
    overlay = pynq.Overlay(XCLBIN)
    compute_unit_names = [name for name in overlay.ip_dict if 'ternip' in name]
    if not compute_unit_names:
        sys.exit(f'no ternip compute unit in {XCLBIN}: {list(overlay.ip_dict)}')
    registers = getattr(overlay, compute_unit_names[0]).mmio
    print(f'Using compute unit {compute_unit_names[0]} '
          f'(of {len(compute_unit_names)})')

    ddr = overlay.device.get_memory_by_name(f'bank{DDR_BANK}')
    data = pynq.allocate(3 * VECTOR_BYTES, dtype=np.uint8, target=ddr)
    address_a, address_b, address_c = 0, VECTOR_BYTES, 2 * VECTOR_BYTES

    vector_a = np.arange(BATCH_SIZE * VECTOR_LENGTH,
                         dtype=np.float32).reshape(VECTOR_SHAPE) % 8
    vector_b = np.full(VECTOR_SHAPE, 1.5, dtype=np.float32)
    data[address_a:address_a + VECTOR_BYTES] = np.frombuffer(
        encode_fixed_point(vector_a).tobytes(), dtype=np.uint8)
    data[address_b:address_b + VECTOR_BYTES] = np.frombuffer(
        encode_fixed_point(vector_b).tobytes(), dtype=np.uint8)
    data[address_c:address_c + VECTOR_BYTES] = 0
    data.sync_to_device()

    # ldv/sv take absolute physical DDR addresses.
    base = data.physical_address
    program = (load_vector(0, base + address_a)
               + load_vector(1, base + address_b)
               + add_vectors(2, 0, 1)
               + store_vector(2, base + address_c)
               + stall())
    instructions = pynq.allocate(len(program), dtype=np.uint8, target=ddr)
    instructions[:] = np.frombuffer(program, dtype=np.uint8)
    instructions.sync_to_device()
    print(f'{len(program) // (INSTRUCTION_WIDTH // 8)} instructions '
          f'({len(program)} bytes) staged on bank{DDR_BANK}')

    registers.write(RESET, 0)
    registers.write(DMA_CONTROL, 1)
    registers.write(DMA_SOURCE_ADDRESS, instructions.physical_address & 0xFFFFFFFF)
    registers.write(DMA_SOURCE_ADDRESS + 4, instructions.physical_address >> 32)
    registers.write(DMA_TRANSFER_LENGTH, len(program))

    while registers.read(STALL) == 0:
        pass
    registers.write(STALL, 1)
    data.sync_from_device()

    result = decode_fixed_point(bytes(data[address_c:address_c + VECTOR_BYTES]))
    expected = vector_a + vector_b
    largest_error = np.abs(result - expected).max()

    print('\nlane 0, first 8 elements:')
    print(f'  a        {vector_a[0][:8]}')
    print(f'  b        {vector_b[0][:8]}')
    print(f'  a+b got  {result[0][:8]}')
    print(f'  expected {expected[0][:8]}')
    print(f'\nlargest error = {largest_error:g} '
          f'(1 LSB = {LEAST_SIGNIFICANT_BIT:g})')
    if largest_error > LEAST_SIGNIFICANT_BIT:
        print('FAIL')
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
