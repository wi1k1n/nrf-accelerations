import xml.etree.ElementTree as ET
import os, os.path as op
from shutil import copyfile

XML_PATH = '.idea/workspace.xml'

myVal = '"myVAL"!!!'

tree = ET.parse(op.abspath(XML_PATH))
# root = tree.getroot()

all_configs = tree.findall('*/configuration')

for config in all_configs:
	if not ('name' in config.attrib) or config.get('name') != 'train': continue
	for option in config.findall('option'):
		if not ('name' in option.attrib) or option.get('name') != 'PARAMETERS': continue
		assert ('value' in option.attrib), 'Theres no VALUE attribute in this configuration! Please check!'
		option.set('value', myVal)
		break
	break

# Create backup
copyfile(op.abspath(XML_PATH), op.abspath(XML_PATH + '.backup'))
tree.write(op.abspath(XML_PATH))

print('hi')