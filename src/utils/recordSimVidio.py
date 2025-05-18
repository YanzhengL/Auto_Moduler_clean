import matlab.engine
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Start MATLAB Engine
eng = matlab.engine.start_matlab()

# Add the directory where the Simulink model is located
model_path = r"C:\Users\ezzyl4\Documents\MATLAB"
eng.addpath(model_path, nargout=0)

# Load the Simulink Model
model_name = "boost_auto"
eng.load_system(model_name, nargout=0)
eng.set_param(model_name, 'SaveTime', 'on',
              'SaveOutput', 'on',
              'SignalLogging', 'on',
              'SaveFormat', 'StructureWithTime', nargout=0)


# Open the model to make it visible for printing
eng.open_system(model_name, nargout=0)
#eng.doc("sim", nargout=0)
#https://uk.mathworks.com/help/releases/R2024b/simulink/slref/sim.html?overload=sim+false
# Save the model as an image
model_handle = eng.get_param(model_name, 'Handle')
sim_out = eng.sim(model_handle, 'SimulationMode', 'normal', nargout=1)
#sim_out = eng.eval("struct(sim('" + str(model_handle) + "', 'SimulationMode', 'normal'))", nargout=1)

#sim_out_dict = eng.struct(sim_out)
#print(sim_out_dict)

#sim_out_dict = sim_out.__dict__  #'matlab.object' object has no attribute '__dict__'
#print("Keys of sim_out.__dict__:")
#print(sim_out_dict.keys())


fields = eng.fieldnames(sim_out)
print("Fieldnames of sim_out:")
print(fields)

all_fields = dir(sim_out)
print("All fields of sim_out:")
print(all_fields)

meta = sim_out[0]
print("Meta type of sim_out:")
print(meta.size)
print(sim_out)

# Retrieve the properties of the simulation output object
#properties = sim_out.getattribute  #'matlab.object' object has no attribute 'getattribute'

time = eng.getfield(sim_out, 'tout')
#print(time)
s_vin = eng.getfield(sim_out, 'vin')
s_vo = eng.getfield(sim_out, 'vo')
s_iL = eng.getfield(sim_out, 'iL')

time_list = list(time)
s_vin_list = list(eng.getfield(sim_out, 'vin'))
s_vo_list = list(eng.getfield(sim_out, 'vo'))
s_iL_list = list(eng.getfield(sim_out, 'iL'))
# Flatten nested lists in case they are 2D
s_vin_list = [item for sublist in s_vin_list for item in sublist]
s_vo_list = [item for sublist in s_vo_list for item in sublist]
s_iL_list = [item for sublist in s_iL_list for item in sublist]
time_list = [item for sublist in time_list for item in sublist]
time_max = float(max(time_list))
print(time_max)
ylim_min = float(min(s_vin_list + s_vo_list + s_iL_list))
ylim_max = float(max(s_vin_list + s_vo_list + s_iL_list))
# Setup video writer
video_path = rf"{model_path}\{model_name}.mp4"
fps = 30
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 600))

# Define your time array and signals
time = np.linspace(0, 10, 100)  # Example time array
s_vin = np.sin(time)  # Example signal for vin
s_vo = np.cos(time)  # Example signal for vo
s_iL = np.tan(time)  # Example signal for iL

# Set limits for the plot
ylim_min, ylim_max = -2, 2  # Example y-axis limits
time_max = time[-1]  # Last value of time for x-axis limit

# Loop through simulation results and save frames
#for i in range(len(time)):
#    # Create a new figure
#    plt.figure(figsize=(8, 6))
#
#    # Plot vin, vo, iL over time
#    plt.plot(time[:i + 1], s_vin[:i + 1], 'b', label="Vin")
#    plt.plot(time[:i + 1], s_vo[:i + 1], 'r', label="Vo")
#    plt.plot(time[:i + 1], s_iL[:i + 1], 'g', label="iL")
#
#    # Set axis limits dynamically
#    plt.xlim(0.0, time_max)
#    plt.ylim(ylim_min, ylim_max)
#
#    # Add legend
#    plt.legend()
#
#    # Save the figure as an image
#    frame_filename = f"frame_{i}.png"
#    plt.savefig(frame_filename, bbox_inches='tight')
#    plt.close()  # Close the figure to avoid memory issues
#
#    # Read and process frame
#    frame = cv2.imread(frame_filename)
#    frame = cv2.resize(frame, (800, 600))
#    video_writer.write(frame)


# Loop through simulation results and save frames
for i in range(len(time)):
    fig = eng.figure(nargout=1)
    eng.hold('on', nargout=0)

    # Plot vin, vo, iL over time
    eng.plot(time[:i + 1], s_vin[:i + 1], 'b',  nargout=0)
    eng.plot(time[:i + 1], s_vo[:i + 1], 'r',  nargout=0)
    eng.plot(time[:i + 1], s_iL[:i + 1], 'g',  nargout=0)
    #Warning: MATLAB has disabled some advanced graphics rendering features by switching to software OpenGL. For more information, click <a href="matlab:opengl('problems')">here</a>.
    # Set axis limits dynamically
    eng.xlim(matlab.double([0.0, time_max]), nargout=0)
    eng.ylim(matlab.double([ylim_min, ylim_max]), nargout=0)

    # Add legend
    eng.legend(["Vin", "Vo", "iL"], nargout=0)

    # Save the figure as an image (PNG)
    frame_filename = f"frame_{i}.png"
    eng.saveas(fig, frame_filename, nargout=0)

    # Read and process frame
    frame = cv2.imread(frame_filename)
    frame = cv2.resize(frame, (800, 600))
    video_writer.write(frame)

    # Close figure to prevent memory leak
    eng.close(fig, nargout=0)


# Cleanup
video_writer.release()
eng.close_system(model_name,nargout=0)
eng.quit()

print(f"Simulation video saved at {video_path}")
