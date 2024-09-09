# CS455+555 Code Examples
This repository contains code snippets and examples used for CS455+555, Machine Vision. The original author of these examples is Mary Everett, and contains code compiled from learning resources online as well. 

##### Required
1. Python 3
2. OpenCV 4.10 recommended

>[!NOTE]
>Check these requirements by running `/coding resources/examples/base_import.py`

## Index

| File | From Lecture | Description |
| :--- | :---: | :--- | 
| `examples/basic_image_stuff.py`         | 4 | Reading + displaying image, webcam capture, image pixel access, image indexing, writing out image, drawing shapes |
| `examples/basic_image_stuff.py`         | 5 | Pixel indexing, image datatype, saving images to files, drawing shapes over image (annotation) |
| `examples/geometric_transformations.py` | 5 | Basic image transformation examples such as rotation, translation, resizing, etc. |
| `examples/colormap.py`                  | 7 | Changing color space, RGB to grayscale, and colormapping (chaning color space) |
| `examples/smoothing.py`                 | 7 | Applying average blur, Gaussian, median, and bilateral filters. |
| `examples/arrays_json.py`               | ?? | Basic array/matrix manipulations and operations using `numpy` library and exporting to JSON file |
| `examples/numpy_json.py`                | ?? | More array manipulations and JSON output | 
| `examples/gradient.py`                  | 8 | Gradient calculation using Gaussian, Sobel kernels. |





### Usage

1. Clone
    
    ```console
   git clone https://github.com/grwells/CS455-555_Machine_Vision.git
    ```

2. Move to examples.
    
    ```console
    cd CS455-555_Machine_Vision/examples
    ```

3. Create virtual environment using desired python version:

    ```console
    python3.xx -m venv venv
    ```

>[!NOTE]
>Replace `python3.xx` with something like `python3.11`, i.e. replace `xx` with the two digits of the python minor version number.

4. Activate virtual environment.

    ```console
    source venv/bin/activate
    ```

5. Install requirements(OpenCV):

    ```console 
    pip3 install -r requirements.txt
    ```

6. Verify install with base import:

    ```console 
    python3 -V && pip3 list | grep opencv
    ```
