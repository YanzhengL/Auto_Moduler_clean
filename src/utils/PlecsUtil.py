import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Open a new PLECS model
eng.eval("model = plecs('new', 'dc_dc_converter.psc');")

# Add components to the model, for example, a Voltage Source, Capacitor, and Load
eng.eval("""
    % Create components in the model
    voltageSource = plecs('addBlock', 'VoltageSource', model);
    capacitor = plecs('addBlock', 'Capacitor', model);
    resistor = plecs('addBlock', 'Resistor', model);

    % Set parameters for the blocks
    plecs('setParameter', voltageSource, 'V', '10');    % 10V source
    plecs('setParameter', capacitor, 'C', '1e-6');      % 1 microfarad
    plecs('setParameter', resistor, 'R', '100');        % 100 Ohms

    % Connect the components
    plecs('connect', voltageSource, 1, capacitor, 1);
    plecs('connect', capacitor, 2, resistor, 1);
""", nargout=0)

# Save the model
eng.eval("plecs('save', model);", nargout=0)

# Close the model
eng.eval("plecs('close', model);", nargout=0)

# End the MATLAB engine
eng.quit()

print("PLECS model created and saved!")
