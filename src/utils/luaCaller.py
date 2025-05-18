import subprocess

# Run the Lua script inside PLECS
subprocess.run(["plecscli", "-exec", "@generate_model.lua"], check=True)
print("PLECS model created using Lua scripting!")
