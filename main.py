import cv2
import numpy as np
import struct
import math

def encode_file_to_video(input_file, output_video, resolution=(512, 512), fps=60):
    """
    Encodes an arbitrary binary file into a video file.
    
    The file’s size (in bytes) is stored in an 8-byte header at the beginning.
    Then the header and file bytes are split into chunks that exactly fill video frames.
    Any leftover space in the final frame is zero-padded.
    
    Args:
        input_file (str): Path to the file to encode.
        output_video (str): Path to the output video file.
        resolution (tuple): (width, height) in pixels for each video frame.
        fps (int): Frames per second to use in the video.
    """
    width, height = resolution
    capacity = width * height * 3  # Each frame holds this many bytes (3 channels per pixel)
    
    # Read the input file as binary
    with open(input_file, 'rb') as f:
        file_bytes = f.read()
    file_size = len(file_bytes)
    
    # Create a header of 8 bytes that stores the original file size (big-endian)
    header = struct.pack('>Q', file_size)
    data = header + file_bytes  # Prepend the header to the file data

    total_bytes = len(data)
    total_frames = math.ceil(total_bytes / capacity)
    print("Total file size (with header):", total_bytes, "bytes")
    print("Capacity per frame:", capacity, "bytes")
    print("Total frames required:", total_frames)

    # Set up the video writer.
    # NOTE: We use the FFV1 codec for lossless encoding. Ensure your OpenCV build supports it.
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        print("Error: Could not open video writer. Check codec support and output path.")
        return

    # Write the file data into successive frames
    for i in range(total_frames):
        start = i * capacity
        end = start + capacity
        frame_data = data[start:end]
        # Pad the last frame with zeros if necessary
        if len(frame_data) < capacity:
            frame_data += bytes(capacity - len(frame_data))
        # Convert the bytes into a NumPy array and reshape it to a frame (height, width, 3)
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
        video_writer.write(frame)

    video_writer.release()
    print(f"Encoding complete. The file '{input_file}' was encoded into '{output_video}'.")

def decode_video_to_file(input_video, output_file):
    """
    Decodes a video file (that was created by encode_file_to_video) back into the original file.
    
    It reads every frame, concatenates all the bytes, then uses the first 8 bytes (header)
    to determine the size of the original file data.
    
    Args:
        input_video (str): Path to the encoded video file.
        output_file (str): Path where the decoded file will be saved.
    """
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video file for decoding.")
        return

    all_bytes = bytearray()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to bytes and add to our data buffer
        all_bytes.extend(frame.tobytes())
        frame_count += 1

    cap.release()
    print("Total frames read:", frame_count)

    # Check that we have at least 8 bytes for the header
    if len(all_bytes) < 8:
        print("Error: Encoded data is too short to contain a valid header.")
        return

    # The first 8 bytes store the original file size
    header = all_bytes[:8]
    file_size = struct.unpack('>Q', header)[0]
    print("Original file size (from header):", file_size, "bytes")

    # Extract exactly the file_size bytes (after the header)
    file_data = all_bytes[8:8 + file_size]
    with open(output_file, 'wb') as f:
        f.write(file_data)
    
    print(f"Decoding complete. The video '{input_video}' was decoded into '{output_file}'.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Encode any file into a video and decode a video back into a file."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands: encode, decode")
    
    # Subparser for encoding
    parser_encode = subparsers.add_parser("encode", help="Encode a file into a video.")
    parser_encode.add_argument("input_file", help="Path to the input file to encode.")
    parser_encode.add_argument("output_video", help="Path to the output video file.")
    parser_encode.add_argument("--width", type=int, default=512, help="Frame width in pixels (default: 256).")
    parser_encode.add_argument("--height", type=int, default=512, help="Frame height in pixels (default: 256).")
    parser_encode.add_argument("--fps", type=int, default=60, help="Frames per second for the video (default: 30).")
    
    # Subparser for decoding
    parser_decode = subparsers.add_parser("decode", help="Decode a video back into a file.")
    parser_decode.add_argument("input_video", help="Path to the input video file.")
    parser_decode.add_argument("output_file", help="Path to save the decoded file.")
    
    args = parser.parse_args()
    
    if args.command == "encode":
        encode_file_to_video(
            args.input_file,
            args.output_video,
            resolution=(args.width, args.height),
            fps=args.fps
        )
    elif args.command == "decode":
        decode_video_to_file(args.input_video, args.output_file)
    else:
        parser.print_help()
