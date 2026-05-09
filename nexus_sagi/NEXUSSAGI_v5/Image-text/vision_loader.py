import numpy as np
from PIL import Image
import os
import imagecodecs # Import imagecodecs for JPEG-LS support

class VisionLoader:
    """
    The Intelligent Eye (Module A).
    
    Responsibilities:
    1. Ingest raw visual data from user formats (.jpg, .png).
    2. Convert 'Visual Data' (Integers) into 'Mathematical Data' (High-precision Floats).
    3. Handle the reconstruction of math back into visual formats.
    """

    def __init__(self):
        self.supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".jls") # Add .jls to supported formats

    def load_to_math(self, file_path):
        """
        Loads an image and converts it into a Normalized Mathematical Matrix.
        
        Smart Format Handling:
        - PNG: Direct lossless processing
        - JPG: Auto-converts to JLS intermediate for lossless reconstruction
        - JLS: Direct lossless processing
        
        Args:
            file_path (str): Path to the input image.
            
        Returns:
            dict: A 'Smart Context' containing:
                - 'matrix': The 3D numpy array of floats (0.0 to 1.0).
                - 'shape': Dimensions (height, width, channels).
                - 'original_format': The source file type.
                - 'jls_path': Path to JLS intermediate (if JPG input).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"System cannot find visual input at: {file_path}")

        try:
            # Handle JLS format
            if file_path.lower().endswith('.jls'):
                print(f"[VisionLoader] Loading JPEG-LS: {file_path}")
                with open(file_path, "rb") as f:
                    data = f.read()
                raw_data = imagecodecs.jpegls_decode(data)
                if raw_data.ndim == 2:
                    raw_data = np.stack([raw_data]*3, axis=-1)
                elif raw_data.shape[-1] == 4:
                    raw_data = raw_data[..., :3]
                
                math_matrix = raw_data.astype(np.float64) / 255.0
                print(f"[VisionLoader] Successfully atomized {file_path}")
                print(f"               Dimensions: {math_matrix.shape}")
                print(f"               Precision: 64-bit Floating Point")
                return {
                    "matrix": math_matrix,
                    "shape": math_matrix.shape,
                    "original_format": "JLS",
                    "jls_path": None
                }
            
            # Handle JPG format - convert to JLS for lossless processing
            elif file_path.lower().endswith(('.jpg', '.jpeg')):
                print(f"[VisionLoader] JPG detected: {file_path}")
                print(f"[VisionLoader] Converting to JLS for lossless processing...")
                
                # Load JPG
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    raw_data = np.asarray(img)
                
                # Save as JLS intermediate
                jls_path = file_path.rsplit('.', 1)[0] + '_intermediate.jls'
                encoded = imagecodecs.jpegls_encode(raw_data, out=raw_data.size * 2)
                with open(jls_path, "wb") as f:
                    f.write(encoded)
                
                print(f"[VisionLoader] JLS intermediate created: {jls_path}")
                
                # Load from JLS
                math_matrix = raw_data.astype(np.float64) / 255.0
                print(f"[VisionLoader] Successfully atomized {file_path}")
                print(f"               Dimensions: {math_matrix.shape}")
                print(f"               Precision: 64-bit Floating Point")
                
                return {
                    "matrix": math_matrix,
                    "shape": math_matrix.shape,
                    "original_format": "JPG",
                    "jls_path": jls_path
                }
            
            # Handle PNG and other formats
            else:
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    raw_data = np.asarray(img)
                    math_matrix = raw_data.astype(np.float64) / 255.0

                    print(f"[VisionLoader] Successfully atomized {file_path}")
                    print(f"               Dimensions: {math_matrix.shape}")
                    print(f"               Precision: 64-bit Floating Point")

                    return {
                        "matrix": math_matrix,
                        "shape": math_matrix.shape,
                        "original_format": img.format,
                        "jls_path": None
                    }

        except Exception as e:
            raise RuntimeError(f"VisionLoader failed to process image: {e}")

    def save_from_math(self, matrix, output_path, jls_intermediate=None):
        """
        Converts Mathematical Logic back into a Real Image.
        
        Smart Format Handling:
        - If JPG output and JLS intermediate exists: JLS -> JPG conversion
        - Otherwise: Direct save in requested format
        
        Args:
            matrix (numpy array): The modified/generated float matrix.
            output_path (str): Where to save the result.
            jls_intermediate (str): Path to JLS intermediate (for JPG reconstruction).
        """
        try:
            matrix = np.clip(matrix, 0.0, 1.0)
            visual_data = (matrix * 255.0).astype(np.uint8)

            # Handle JPG output with JLS intermediate
            if output_path.lower().endswith(('.jpg', '.jpeg')) and jls_intermediate:
                print(f"[VisionLoader] Reconstructing JPG via JLS intermediate...")
                
                # Save as JLS first
                temp_jls = output_path.rsplit('.', 1)[0] + '_reconstructed.jls'
                encoded = imagecodecs.jpegls_encode(visual_data, out=visual_data.size * 2)
                with open(temp_jls, "wb") as f:
                    f.write(encoded)
                
                # Convert JLS to JPG
                with open(temp_jls, "rb") as f:
                    jls_data = imagecodecs.jpegls_decode(f.read())
                img = Image.fromarray(jls_data)
                img.save(output_path, quality=100, subsampling=0)
                
                print(f"[VisionLoader] Lossless reconstruction: {temp_jls}")
                print(f"[VisionLoader] Final JPG output: {output_path}")
            
            # Handle JLS format
            elif output_path.lower().endswith('.jls'):
                print(f"[VisionLoader] Saving JPEG-LS: {output_path}")
                encoded = imagecodecs.jpegls_encode(visual_data, out=visual_data.size * 2)
                with open(output_path, "wb") as f:
                    f.write(encoded)
                print(f"[VisionLoader] Reconstructed reality saved to: {output_path}")
            
            # Handle PNG and other formats
            else:
                img = Image.fromarray(visual_data)
                img.save(output_path)
                print(f"[VisionLoader] Reconstructed reality saved to: {output_path}")

        except Exception as e:
            raise RuntimeError(f"VisionLoader failed to render image: {e}")

# --- Self-Test Block (Runs if you execute this file directly) ---
if __name__ == "__main__":
    # Create a dummy image to test the logic
    loader = VisionLoader()
    
    # Generate a pure math gradient (0.0 to 1.0)
    print("Running Logic Test...")
    width, height = 256, 256
    test_matrix = np.zeros((height, width, 3), dtype=np.float64)
    
    for y in range(height):
        for x in range(width):
            # Create a logical Red/Blue gradient
            test_matrix[y, x] = [x/255.0, y/255.0, 0.0]

    # Save it to prove the system works
    loader.save_from_math(test_matrix, "system_check_gradient.png")
    print("Test Complete. Check 'system_check_gradient.png'.")
