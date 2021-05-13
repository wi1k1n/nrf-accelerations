import os, os.path as op, sys, time
import xml.etree.ElementTree as ET
from shutil import copyfile


def inject_pycharm_config(config_name, xml_path, parameters, num_backups = 10):
	tree = ET.parse(op.abspath(xml_path))
	all_configs = tree.findall('configuration')
	configFound = False
	for config in all_configs:
		if not ('name' in config.attrib) or config.get('name') != config_name: continue
		configFound = True
		for option in config.findall('option'):
			if not ('name' in option.attrib) or option.get('name') != 'PARAMETERS': continue
			assert ('value' in option.attrib), 'Theres no VALUE attribute in this configuration! Please check!'
			option.set('value', parameters)
			break
		break
	assert configFound, 'No configuration "' + config_name + '" found!'

	# Create backup
	xmlDir = op.dirname(xml_path)
	configName = op.basename(xml_path)
	backups = [int(op.splitext(f)[0].split('.')[-1]) for f in os.listdir(xmlDir) if
			   f.endswith('.backup') and op.isfile(op.join(xmlDir, f)) and configName in f]
	lastBackup, firstBackup = (max(backups), min(backups)) if any(backups) else (0, None)

	copyfile(op.abspath(xml_path), op.abspath(xml_path + '.' + str(lastBackup + 1) + '.backup'))

	for rpt in range(2):
		tree.write(op.abspath(xml_path))
		time.sleep(0.5)

	# Delete the oldest backup
	if len(backups) >= num_backups:
		os.remove(op.abspath(xml_path + '.' + str(firstBackup) + '.backup'))