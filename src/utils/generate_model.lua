model = plecs.new('dc_dc_converter.psc')

vs = plecs.addBlock('VoltageSource', model)
cap = plecs.addBlock('Capacitor', model)
res = plecs.addBlock('Resistor', model)

plecs.setParameter(vs, 'V', '10')
plecs.setParameter(cap, 'C', '1e-6')
plecs.setParameter(res, 'R', '100')

plecs.connect(vs, 1, cap, 1)
plecs.connect(cap, 2, res, 1)

plecs.save(model)
plecs.close(model)
