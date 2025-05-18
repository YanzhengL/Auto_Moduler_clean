import matlab.engine

# Start MATLAB Engine
eng = matlab.engine.start_matlab()

# Add the directory where the Simulink model is located
model_path = r"C:\Users\ezzyl4\Documents\MATLAB"
eng.addpath(model_path, nargout=0)

# Load the Simulink Model
model_name = "boost_auto"
eng.load_system(model_name, nargout=0)

# Open the model to make it visible for printing
eng.open_system(model_name, nargout=0)

# Save the model as an image
model_handle = eng.get_param(model_name, 'Handle')
image_path = rf"{model_path}\{model_name}.png"  # Full path for the image
eng.print(model_handle, "-dpng", "-r300", image_path, nargout=0)  # High-res PNG

# Close Simulink without saving changes
eng.close_system(model_name, nargout=0)

print(f"Model image saved at {image_path}")

# Stop MATLAB Engine
eng.quit()
