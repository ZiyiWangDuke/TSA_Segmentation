import pysftp
import sys
import pdb

# path = './THETARGETDIRECTORY/' + sys.argv[1]    #hard-coded
# localpath = sys.argv[1]

host = "152.3.214.126"                    #hard-coded
password = "seg123"                #hard-coded
username = "seg"                #hard-coded

with pysftp.Connection(host, username=username, password=password) as sftp:
    sftp.chdir('..')
    sftp.chdir('/ext2/rec/PSRC/Data/stage1/a3d') # data path
    data = sftp.listdir()
    # filelist = data[1104:1172] # download this files
    filelist = data[1126:1172]
    for filename in filelist:
        sftp.get(filename)
        print filename+' is done.'
    # sftp.put(localpath, path)

print 'All done.'
