import paramiko
from threading import Timer
import os, os.path as op, pathlib, fnmatch
from datetime import datetime

from ssh_credentials import SSH_USER, SSH_PORT, SSH_HOST, SSH_KEY

REMOTE_FOLDER = '/tmp/remdeb/'
LOCAL_FOLDER = 'D:\\edu\\UniBonn\\Study\\thesis\\codes\\NSVF\\remdeb\\'
INTERVAL = 5   # in seconds

FILTER = '*.rmdb'


# Start connection
client = paramiko.SSHClient()
client.load_system_host_keys()
client.load_host_keys(SSH_KEY)
client.connect(SSH_HOST, SSH_PORT, SSH_USER)
sftp = client.open_sftp()

# Change dir to remote_folder or create if not existing
try:
	sftp.chdir(REMOTE_FOLDER)
except IOError:
	sftp.mkdir(REMOTE_FOLDER)
	sftp.chdir(REMOTE_FOLDER)

# Prepare local directory
if not op.exists(LOCAL_FOLDER):
	os.makedirs(LOCAL_FOLDER)

fileList = {f: pathlib.Path(op.join(LOCAL_FOLDER, f)).stat() \
			for f in os.listdir(LOCAL_FOLDER) if op.isfile(op.join(LOCAL_FOLDER, f))}

# Function of watchdog to be executed each INTERVAL seconds
def watchFolder():
	print(datetime.now().strftime("%H:%M:%S"))
	Timer(INTERVAL, watchFolder).start()

	remoteDir = {f: sftp.lstat(f) for f in sftp.listdir()}

	for f in remoteDir:
		lstat = remoteDir[f]
		# Only filter files
		if 'd' in str(lstat).split()[0] or not fnmatch.fnmatch(f, FILTER):
			continue

		if f not in fileList or fileList[f].st_mtime < lstat.st_mtime:
			print('> {}'.format(f))
			sftp.get(op.join(REMOTE_FOLDER, f), op.join(LOCAL_FOLDER, f))
			fileList.update({f: pathlib.Path(op.join(LOCAL_FOLDER, f)).stat()})

watchFolder()