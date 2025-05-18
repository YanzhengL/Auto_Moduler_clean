import win32com.client

com_objects = win32com.client.gencache.EnsureDispatch("Excel.Application")
print(com_objects)

# Initialize PSIM COM Object
psim = win32com.client.Dispatch("PSIM.AutoSim")

# Create a new schematic
psim.NewSch()

# Add a DC voltage source
voltage_source = psim.AddElement("Vdc", 0, 0)
psim.SetParam(voltage_source, "Vdc", "10")  # Set voltage to 10V

# Add a resistor
resistor = psim.AddElement("R", 100, 0)
psim.SetParam(resistor, "R", "100")  # Set resistance to 100 ohms

# Add a ground
ground = psim.AddElement("GND", 50, -50)

# Connect components
psim.AddWire(0, 0, 100, 0)  # Connect voltage source to resistor
psim.AddWire(100, 0, 100, -50)  # Connect resistor to ground
psim.AddWire(0, 0, 0, -50)  # Connect voltage source to ground

# Save schematic
psim.SaveSch("C:\\path\\to\\your\\circuit.psimsch")

# Run simulation
psim.RunSimulation()

print("PSIM circuit created and simulated successfully.")
