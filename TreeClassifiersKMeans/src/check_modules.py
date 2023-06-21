import sys

dependencies = ["numpy", "pandas", "sklearn", "sklearnex"]

for module in dependencies:
    print("\nChecking for " + module + "...", end='')
    try:
        # Import module from string variable:
        # https://stackoverflow.com/questions/8718885/import-module-from-string-variable
        # To import using a variable, call __import__(name)
        module_obj = __import__(module)
        # To contain the module, create a global object using globals()
        globals()[module] = module_obj
    except ImportError:
        print("Install " + module + " before continuing")
        print("In a terminal type the following commands:")
        print("python get-pip.py")
        print("pip install " + module + "\n")
        sys.exit(1)

print("\nSystem is ready!")
