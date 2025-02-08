import cv2
import numpy as np
import struct
import math
from reedsolo import RSCodec

# --- Configuration parameters ---
FRAME_WIDTH = 512
FRAME_HEIGHT = 512
CELL_SIZE = 16                     # Each cell is 16x16 pixels.
GRID_COLS = FRAME_WIDTH // CELL_SIZE  # 32 cells per row.
GRID_ROWS = FRAME_HEIGHT // CELL_SIZE   # 32 cells per column.
BITS_PER_FRAME = GRID_COLS * GRID_ROWS   # 1024 bits per frame.
BYTES_PER_FRAME = BITS_PER_FRAME // 8      # 128 bytes per frame.

# Reed–Solomon parameters:
ECC_SYMBOLS = 16  # Number of error correction bytes per block.
DATA_BLOCK_SIZE = BYTES_PER_FRAME - ECC_SYMBOLS  # 112 bytes of actual data per block.

# --- Robust Encoding Function ---
def robust_encode_file_to_video(input_file, output_video, fps=30):
    # Read the input file.
    with open(input_file, 'rb') as f:
        file_data = f.read()
    file_size = len(file_data)
    print("Original file size:", file_size, "bytes")
    
    # Create a simple header (4 bytes, big-endian) to store file size.
    header = struct.pack('>I', file_size)
    full_data = header + file_data
    
    # Split the data into chunks of DATA_BLOCK_SIZE bytes; pad the final chunk if needed.
    chunks = [full_data[i:i+DATA_BLOCK_SIZE] for i in range(0, len(full_data), DATA_BLOCK_SIZE)]
    if len(chunks[-1]) < DATA_BLOCK_SIZE:
        chunks[-1] += bytes(DATA_BLOCK_SIZE - len(chunks[-1]))
    
    # Initialize the Reed–Solomon encoder.
    rsc = RSCodec(ECC_SYMBOLS)
    
    # Encode each chunk to a 128-byte block.
    encoded_blocks = []
    for chunk in chunks:
        encoded = rsc.encode(chunk)  # Each block is DATA_BLOCK_SIZE + ECC_SYMBOLS = 128 bytes.
        encoded_blocks.append(encoded)
    
    total_frames = len(encoded_blocks)
    print("Total frames to encode:", total_frames)
    
    # Set up the video writer.
    # (Ideally you’d use a lossless codec when testing; note that YouTube will re‑encode this video.)
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(output_video, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))
    if not out.isOpened():
        print("Error: Could not open video writer.")
        return
    
    # Process each encoded block.
    for block in encoded_blocks:
        # Convert the 128-byte block into a list of 1024 bits.
        bits = []
        for byte in block:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        if len(bits) != BITS_PER_FRAME:
            print("Error: Bit length mismatch.")
            return
        
        # Create an empty grayscale frame.
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        bit_index = 0
        # Fill the frame with cells. Each cell is filled white for a 1 and black for a 0.
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                bit = bits[bit_index]
                bit_index += 1
                top = row * CELL_SIZE
                left = col * CELL_SIZE
                frame[top:top+CELL_SIZE, left:left+CELL_SIZE] = 255 if bit == 1 else 0
        
        # Convert the grayscale frame to BGR (as required by many video codecs).
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Robust encoding complete. Video saved as '{output_video}'.")

# --- Robust Decoding Function ---
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
        # Convert the frame to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract the bit values from each cell by thresholding.
        bits = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                top = row * CELL_SIZE
                left = col * CELL_SIZE
                cell = gray[top:top+CELL_SIZE, left:left+CELL_SIZE]
                avg_intensity = np.mean(cell)
                bit = 1 if avg_intensity > 128 else 0
                bits.append(bit)
        
        # Convert the bits back into a 128-byte block.
        block_bytes = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i+j]
            block_bytes.append(byte)
        block_bytes = bytes(block_bytes)
        
        # Use Reed–Solomon to decode the block (correcting errors if possible).
        try:
            decoded = rsc.decode(block_bytes)
            # Depending on the library version, decoded might be a tuple; take the message portion.
            if isinstance(decoded, tuple):
                decoded = decoded[0]
        except Exception as e:
            print(f"Error decoding RS block in frame {frame_count}: {e}")
            continue  # Skip blocks that couldn’t be corrected.
        
        decoded_blocks.append(decoded)
    
    cap.release()
    print("Total frames processed:", frame_count)
    
    if not decoded_blocks:
        print("No valid blocks were decoded.")
        return
    
    # Combine the decoded blocks into one data stream.
    full_data = b''.join(decoded_blocks)
    if len(full_data) < 4:
        print("Error: Decoded data is too short to contain a valid header.")
        return
    
    # The first 4 bytes hold the original file size.
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
        description="Robustly encode a file into a video (aimed at surviving YouTube compression) and decode it back."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands: encode, decode")
    
    parser_encode = subparsers.add_parser("encode", help="Encode a file into a robust video.")
    parser_encode.add_argument("input_file", help="Path to the input file.")
    parser_encode.add_argument("output_video", help="Path to the output video file.")
    parser_encode.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    
    parser_decode = subparsers.add_parser("decode", help="Decode a robust video back into a file.")
    parser_decode.add_argument("input_video", help="Path to the input video file.")
    parser_decode.add_argument("output_file", help="Path to save the recovered file.")
    
    args = parser.parse_args()
    
    if args.command == "encode":
        robust_encode_file_to_video(args.input_file, args.output_video, fps=args.fps)
    elif args.command == "decode":
        robust_decode_video_to_file(args.input_video, args.output_file)
    else:
        parser.print_help()
