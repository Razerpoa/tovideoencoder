# tovideoencoder

## Overview
`tovideoencoder` is a tool that encodes any file into an `.mkv` video format using the FFV1 codec. But this one is made for youtube lossy compression so you can store unlimited amount of files inside youtube

## Installation
Clone the repository:
```sh
git clone https://github.com/Razerpoa/tovideoencoder
```
Navigate to the directory and install dependencies:
```sh
cd tovideoencoder
python -m pip install -r requirements.txt
```
Run the program:
```sh
python main.py
```

## Usage

### Encoding a file
To encode a file into an `.mkv` video:
```sh
python main.py encode my_groceries.txt groceries.mkv
```

### Decoding a file
To decode an `.mkv` video back into its original file:
```sh
python main.py decode groceries.mkv my_groceries.txt
```

## License
This project is licensed under the MIT License.
