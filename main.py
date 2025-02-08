import cv2
import numpy as np
import struct
import math

def encode_file_to_video(input_file, output_video, resolution=(512, 512), fps=60):
    width, height = resolution
    capacity = width * height * 3
    
    with open(input_file, 'rb') as f:
        file_bytes = f.read()
    file_size = len(file_bytes)
    
    header = struct.pack('>Q', file_size)
    data = header + file_bytes
    total_bytes = len(data)
    total_frames = math.ceil(total_bytes / capacity)
    
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        print("Error: Could not open video writer. Check codec support and output path.")
        return
    
    for i in range(total_frames):
        start = i * capacity
        end = start + capacity
        frame_data = data[start:end]
        if len(frame_data) < capacity:
            frame_data += bytes(capacity - len(frame_data))
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Encoding complete. The file '{input_file}' was encoded into '{output_video}'.")

def decode_video_to_file(input_video, output_file):
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
        all_bytes.extend(frame.tobytes())
        frame_count += 1
    
    cap.release()
    
    if len(all_bytes) < 8:
        print("Error: Encoded data is too short to contain a valid header.")
        return
    
    header = all_bytes[:8]
    file_size = struct.unpack('>Q', header)[0]
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
    
    parser_encode = subparsers.add_parser("encode", help="Encode a file into a video.")
    parser_encode.add_argument("input_file", help="Path to the input file to encode.")
    parser_encode.add_argument("output_video", help="Path to the output video file.")
    parser_encode.add_argument("--width", type=int, default=512, help="Frame width in pixels.")
    parser_encode.add_argument("--height", type=int, default=512, help="Frame height in pixels.")
    parser_encode.add_argument("--fps", type=int, default=60, help="Frames per second for the video.")
    
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
