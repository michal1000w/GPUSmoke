import struct
import os
def dump_one_file(fname):
    f = open(fname, "rb")

    magic = f.read(8)
    if magic != b'BPHYSICS':
        raise Exception("not a blender physics cache")

    flavor = f.read(12)
    (flavor,count,something) = struct.unpack("iii", flavor)

    print( "%d\t%d\t%d"%(flavor,count,something))


    if flavor==1: # point cache
        lines = []

        rec_len = 28
        while True:
            chunk = f.read(rec_len)

            if chunk is None or len(chunk)==0:
                break
            if len(chunk) != rec_len:
                raise Exception("short read (%d<%d)"%(len(chunk), rec_len))

            all = struct.unpack("i fff fff ", chunk)
            line = "%d;%f,%f,%f;%f,%f,%f"%all
            lines.append(line+"\n")
            print(line)
        
        f.close()

        try:
            os.mkdir(".\\output")
        except:
            print("Cannot create output folder")

        f = open(".\\output\\" + fname.split('\\')[-1].split('.')[0] + ".particle","w+")
        f.writelines(lines)
        f.close()




folder = ".\\blendcache_blender_particle_system"
filees = []
for root, dirs, files in os.walk(folder):
    for name in files:
        print(name)
        filees.append(os.path.join(root, name))
    break


for i in filees:
    dump_one_file(i)
#dump_one_file("./blendcache_blender_particle_system/explosion_000003_00.bphys")