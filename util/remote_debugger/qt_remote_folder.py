from datetime import datetime
from functools import wraps
try:
	from IPython import get_ipython
except:
	pass
import numpy as np
import os
import sys
import traceback
import time
import types
from warnings import warn

import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QCheckBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, \
	QLineEdit, QMainWindow, QPushButton, QShortcut, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget, \
	QFileDialog, QListWidget
from PyQt5.Qt import QImage

import matplotlib
try:
	matplotlib.use('Qt5Agg')
except:
	pass
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import paramiko

try:
	from torch import Tensor
except:
	Tensor = type(None)

class rd(QMainWindow):
	sftp = None
	app = None

	def __init__(self, hostname, username, port=22, **kwargs):
		self.sftp = SFTP(hostname, username, port)
		self.sftp.connect()
		assert self.sftp.connected, 'Could not open SFTP connection'

		self.app = QtCore.QCoreApplication.instance()
		if self.app is None:
			self.app = QApplication([''])
		QMainWindow.__init__(self, parent=None)

		self.timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
		self.setWindowTitle('rd ' + self.timestamp)

		try:
			shell = get_ipython()
			if not shell is None:
				shell.magic('%matplotlib qt')
		except:
			pass

		self.initUI()
		self.show()

	def initUI(self):
		self.widget = QWidget()

		form = QFormLayout()
		self.uiLEHost = QLineEdit('localhost')
		self.uiLEUser = QLineEdit('mazlov')
		self.uiLEPort = QLineEdit('22')
		form.addRow(QLabel('Hostname:'), self.uiLEHost)
		form.addRow(QLabel('Username:'), self.uiLEUser)
		form.addRow(QLabel('Port:'), self.uiLEPort)

		self.uiBtnRefresh = QPushButton("Refresh")
		self.uiBtnRefresh.clicked.connect(self.updateFileList)

		vbox = QVBoxLayout()
		# vbox.addLayout(form)
		vbox.addWidget(self.uiBtnRefresh)
		vbox.addStretch()
		# vbox.addItem(QSpacerItem(1, 1, vPolicy=QSizePolicy.Expanding))
		# vbox.addLayout(form_bottom)
		# vbox.addLayout(form_bottom2)

		self.list = QListWidget()
		# self.list.addItem('hey there!')

		hbox = QHBoxLayout()
		hbox.addWidget(self.list)
		hbox.addLayout(vbox)
		hbox.addStretch()


		self.widget.setLayout(hbox)
		self.setCentralWidget(self.widget)

		# keyboard shortcuts
		closeShortcut = QShortcut(QKeySequence('Escape'), self.widget)
		closeShortcut.activated.connect(self.close)

	def updateFileList(self):
		files = self.sftp.getFileList()
		self.list.clear()
		for file in files:
			self.list.addItem(file)

class SFTP:
	client = None
	sftp = None
	connected = False

	def __init__(self, hostname, username, port, dir='/tmp/remdeb/'):
		self.hostname = hostname
		self.username = username
		self.port = port
		self.dir = dir

		paramiko.util.log_to_file("remote_debugger.sftp.log")

	def connect(self):
		self.client = paramiko.SSHClient()
		self.client.load_system_host_keys()
		self.client.load_host_keys("\\\\wsl$\\Ubuntu-18.04\\home\\mazlov\\.ssh\\id_rsa")
		self.client.connect(self.hostname, self.port, self.username)
		self.sftp = self.client.open_sftp()

		try:
			self.sftp.chdir(self.dir)  # Test if remote_path exists
		except IOError:
			self.sftp.mkdir(self.dir)  # Create remote_path
			self.sftp.chdir(self.dir)

		if (self.sftp):
			self.connected = True


	def getFileList(self):
		return [file for file in self.listdir() if 'd' not in str(self.sftp.lstat(file)).split()[0]]

	def listdir(self):
		return self.sftp.listdir()


if __name__ == '__main__':
	rdeb = rd('localhost', 'mazlov', 2040)

	sys.exit(rdeb.app.exec_())