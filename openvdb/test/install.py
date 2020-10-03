try:
    import pyopenvdb as vdb
    print("OpenVDB already installed")
except:
    import os
    print("Installing pyopenvdb")
    os.system("pip install pyopenvdb")