import cv2
import numpy as np
import struct
import math
from reedsolo import RSCodec

# --- Configuration parameters for multi-level modulation ---
# Video parameters:
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CELL_SIZE = 80  # Each cell is 80x80 pixels.
GRID_COLS = FRAME_WIDTH // CELL_SIZE   # 1920 / 80 = 24
GRID_ROWS = FRAME_HEIGHT // CELL_SIZE    # 1080 / 80 = 13
CELLS = GRID_COLS * GRID_ROWS            # 24 * 13 = 312

# Each cell will encode 6 bits (2 bits per color channel: Blue, Green, Red)
BITS_PER_CELL = 6
BITS_PER_FRAME = CELLS * BITS_PER_CELL    # 312 * 6 = 1872 bits
BYTES_PER_FRAME = BITS_PER_FRAME // 8       # 1872 / 8 = 234 bytes

# Reed–Solomon parameters:
ECC_SYMBOLS = 78                           # 33% overhead approximately
DATA_BLOCK_SIZE = BYTES_PER_FRAME - ECC_SYMBOLS  # 234 - 78 = 156 bytes

# Mapping from a 2-bit value (0-3) to an intensity (0, 85, 170, 255)
level_mapping = {0: 0, 1: 85, 2: 170, 3: 255}

def bits_from_bytes(data):
    """Convert a bytes object into a list of bits (0 or 1)."""
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bytes_from_bits(bits):
    """Convert a list of bits into a bytes object (8 bits per byte)."""
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i+j]
        result.append(byte)
    return bytes(result)

# --- Robust Encoding Function with Multi-Level Modulation ---
def robust_encode_file_to_video(input_file, output_video, fps=30):
    # Read and (optionally) precompress your file.
    with open(input_file, 'rb') as f:
        file_data = f.read()
    file_size = len(file_data)
    print("Original file size:", file_size, "bytes")
    
    # Create a simple header (4 bytes) to store the file size.
    header = struct.pack('>I', file_size)
    full_data = header + file_data
    
    # Split data into chunks of DATA_BLOCK_SIZE bytes; pad the last chunk if necessary.
    chunks = [full_data[i:i+DATA_BLOCK_SIZE] for i in range(0, len(full_data), DATA_BLOCK_SIZE)]
    if len(chunks[-1]) < DATA_BLOCK_SIZE:
        chunks[-1] += bytes(DATA_BLOCK_SIZE - len(chunks[-1]))
    
    # Initialize Reed–Solomon encoder.
    rsc = RSCodec(ECC_SYMBOLS)
    
    # Encode each chunk into a block of BYTES_PER_FRAME bytes.
    encoded_blocks = []
    for chunk in chunks:
        encoded = rsc.encode(chunk)  # Produces a block of length DATA_BLOCK_SIZE + ECC_SYMBOLS = 234 bytes.
        encoded_blocks.append(encoded)
    
    total_frames = len(encoded_blocks)
    print("Total frames to encode:", total_frames)
    
    # Set up the video writer.
    # (Using FFV1 as a lossless codec for local testing; note that YouTube will re-encode.)
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(output_video, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))
    if not out.isOpened():
        print("Error: Could not open video writer.")
        return
    
    # Process each encoded block.
    for block in encoded_blocks:
        # Convert the RS block (234 bytes) into a bit list.
        block_bits = bits_from_bytes(block)  # Should yield 234 * 8 = 1872 bits.
        if len(block_bits) != BITS_PER_FRAME:
            print("Error: Bit length mismatch.")
            return
        
        # Create an empty frame (color image).
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        bit_index = 0
        
        # For each cell in the grid, extract 6 bits (2 bits per channel).
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                # Extract 2 bits for each channel (order: Blue, Green, Red).
                blue_val = (block_bits[bit_index] << 1) | block_bits[bit_index + 1]
                green_val = (block_bits[bit_index + 2] << 1) | block_bits[bit_index + 3]
                red_val = (block_bits[bit_index + 4] << 1) | block_bits[bit_index + 5]
                bit_index += 6
                
                top = row * CELL_SIZE
                left = col * CELL_SIZE
                # Map the 2-bit value to an intensity.
                b_intensity = level_mapping[blue_val]
                g_intensity = level_mapping[green_val]
                r_intensity = level_mapping[red_val]
                
                # Fill the cell in the frame.
                frame[top:top+CELL_SIZE, left:left+CELL_SIZE, 0] = b_intensity  # Blue channel
                frame[top:top+CELL_SIZE, left:left+CELL_SIZE, 1] = g_intensity  # Green channel
                frame[top:top+CELL_SIZE, left:left+CELL_SIZE, 2] = r_intensity  # Red channel
        
        out.write(frame)
    
    out.release()
    print(f"Robust encoding complete. Video saved as '{output_video}'.")

# --- Robust Decoding Function with Multi-Level Modulation ---
def robust_decode_video_to_file(input_video, output_file):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video file for decoding.")
        return
    
    rsc = RSCodec(ECC_SYMBOLS)
    decoded_blocks = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # Recover bits from the frame by processing each cell.
        bits = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                top = row * CELL_SIZE
                left = col * CELL_SIZE
                cell = frame[top:top+CELL_SIZE, left:left+CELL_SIZE]
                
                # Compute the average intensity for each channel.
                b_avg = np.mean(cell[:, :, 0])
                g_avg = np.mean(cell[:, :, 1])
                r_avg = np.mean(cell[:, :, 2])
                
                # Map the average to the nearest level (0-3). Since our levels are multiples of 85:
                b_level = int(round(b_avg / 85))
                g_level = int(round(g_avg / 85))
                r_level = int(round(r_avg / 85))
                b_level = max(0, min(3, b_level))
                g_level = max(0, min(3, g_level))
                r_level = max(0, min(3, r_level))
                
                # Convert the level (0-3) back into 2 bits.
                bits.extend([(b_level >> 1) & 1, b_level & 1])
                bits.extend([(g_level >> 1) & 1, g_level & 1])
                bits.extend([(r_level >> 1) & 1, r_level & 1])
        
        # Verify that we have the expected number of bits.
        if len(bits) != BITS_PER_FRAME:
            print("Error: Bit length mismatch in frame", frame_count)
            continue
        
        # Convert the recovered bits to a byte block.
        block_bytes = bytes_from_bits(bits)
        
        # Use Reed–Solomon to correct any errors and decode the block.
        try:
            decoded = rsc.decode(block_bytes)
            if isinstance(decoded, tuple):
                decoded = decoded[0]
        except Exception as e:
            print(f"Error decoding RS block in frame {frame_count}: {e}")
            continue
        
        decoded_blocks.append(decoded)
    
    cap.release()
    print("Total frames processed:", frame_count)
    
    if not decoded_blocks:
        print("No valid blocks were decoded.")
        return
    
    # Reassemble the full data stream.
    full_data = b''.join(decoded_blocks)
    if len(full_data) < 4:
        print("Error: Decoded data is too short to contain a valid header.")
        return
    
    # The first 4 bytes contain the original file size.
    file_size = struct.unpack('>I', full_data[:4])[0]
    print("Recovered file size (from header):", file_size, "bytes")
    
    file_data = full_data[4:4+file_size]
    with open(output_file, 'wb') as f:
        f.write(file_data)
    
    print(f"Robust decoding complete. Recovered file saved as '{output_file}'.")

# --- Command-Line Interface ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Robustly encode a file into a video using multi-level (quaternary) modulation and decode it back."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands: encode, decode")
    
    parser_encode = subparsers.add_parser("encode", help="Encode a file into a robust multi-level video.")
    parser_encode.add_argument("input_file", help="Path to the input file.")
    parser_encode.add_argument("output_video", help="Path to the output video file.")
    parser_encode.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    
    parser_decode = subparsers.add_parser("decode", help="Decode a robust multi-level video back into a file.")
    parser_decode.add_argument("input_video", help="Path to the input video file.")
    parser_decode.add_argument("output_file", help="Path to save the recovered file.")
    
    args = parser.parse_args()
    
    if args.command == "encode":
        robust_encode_file_to_video(args.input_file, args.output_video, fps=args.fps)
    elif args.command == "decode":
        robust_decode_video_to_file(args.input_video, args.output_file)
    else:
        parser.print_help()
