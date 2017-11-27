import config

def readGroundTruth(filename):
    name=filename
    if '.txt' not in name:
        pos=name.rfind('.')
        name=name[:pos]+'.txt'

    if '\\' not in name:
        name=config.GroundTruthPath+name
    try:
        txt_reader=open(name,'r')
        line = txt_reader.readline()
        ret=[float(x) for x in line.split()]
        print('ground truth',ret)
        return ret
    except:
        print('cannot find ground truth file!',name)
        return [-1,-1,-1,-1]


