# StyleCraft

This repository contains the project submission by **Euisuh Jeong** for the CUDA Advanced Libraries course. 

This Python project leverages the power of deep learning and GPUs to perform image style transfer, a process where the artistic style of one image is applied to the content of another. Utilizing PyTorch and a pre-trained VGG-19 model, the program transforms images by merging the style of one image with the content of another. The script is designed to work efficiently on GPUs, making it suitable for handling high-resolution images. It features functions for image loading, preprocessing, and displaying results, providing a foundation for further customization and exploration in the field of artistic image manipulation. This project is ideal for those interested in computer vision, artificial intelligence, and creative applications of deep learning.

## Running the Code

To run the code, follow these steps:

1. **Install Requirements**: First, make sure you have all the required dependencies installed. You can install them using pip by running the following command in your terminal:

    ```
    pip install -r requirements.txt
    ```

2. **Run the Script**: Once you have installed the requirements, you can run the main script. Make sure to replace `<content_image_path>`, `<style_image_path>`, and `<output_path>` with the actual paths of your input image, style image, and desired output image respectively. Run the following command:

    ```
    python main.py <content_image_path> <style_image_path> <output_path>
    ```

    For example:

    ```
    python main.py assets/newjeans-omg.jpeg assets/les-demoiselles-davignon.jpeg assets/output.jpeg
    ```

3. **Check the Output**: Once the script finishes running, you should find the styled image saved at the specified output path.

Feel free to reach out if you encounter any issues or have any questions!
