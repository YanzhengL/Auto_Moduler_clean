import matlab.engine
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sympy.physics.units import length


def create_simulation_video(model_name, model_path, fps=30, frame_size=(800, 600)):
    """
    Runs a Simulink model simulation and generates a video from the simulation results.

    :param model_name: Name of the Simulink model (without .slx extension)
    :param model_path: Path to the directory containing the model
    :param output_video: Path to save the output video
    :param fps: Frames per second for the video
    :param frame_size: Tuple (width, height) for video resolution
    """
    # Start MATLAB Engine
    eng = matlab.engine.start_matlab()
    eng.addpath(model_path, nargout=0)

    # Load and configure the Simulink model
    eng.load_system(model_name, nargout=0)
    eng.set_param(model_name, 'SaveTime', 'on',
                  'SaveOutput', 'on',
                  'SignalLogging', 'on',
                  'SaveFormat', 'StructureWithTime', nargout=0)

    # Run simulation
    sim_out = eng.sim(model_name, 'SimulationMode', 'normal', nargout=1)

    # Extract simulation data
    time = eng.getfield(sim_out, 'tout')
    s_vin = eng.getfield(sim_out, 'vin')
    s_vo = eng.getfield(sim_out, 'vo')
    s_iL = eng.getfield(sim_out, 'iL')

    time_list = [item for sublist in list(time) for item in sublist]
    s_vin_list = [item for sublist in list(s_vin) for item in sublist]
    s_vo_list = [item for sublist in list(s_vo) for item in sublist]
    s_iL_list = [item for sublist in list(s_iL) for item in sublist]

    # Ensure all lists have the same length
    min_len = min(len(time_list), len(s_vin_list), len(s_vo_list), len(s_iL_list))

    # Trim lists to the minimum length
    time_list = time_list[:min_len]
    s_vin_list = s_vin_list[:min_len]
    s_vo_list = s_vo_list[:min_len]
    s_iL_list = s_iL_list[:min_len]

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(time_list, s_vin_list, 'b', label="Vin")
    plt.plot(time_list, s_vo_list, 'r', label="Vo")
    plt.plot(time_list, s_iL_list, 'g', label="iL")

    # Formatting
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.title("Time vs. Vin, Vo, and iL")
    plt.legend()
    plt.grid()

    # Save the plot as an image
    plt.savefig("plot.png", dpi=300, bbox_inches="tight")

    # Show the plot (optional)
    plt.show()

    eng.close_system(model_name, 0, nargout=0)
    eng.quit()




# Example usage
if __name__ == "__main__":
    model_name = "boost"
    model_path = r"E:\\job\\AutoModler\\src"
    output_video = os.path.join(model_path, f"{model_name}.mp4")
    fps = 30  # Ensure it's an integer
    frame_size = (800, 600)
    create_simulation_video(model_name, model_path,  fps, frame_size)
