#open printer.cfg file to read configuration parameters

import configparser


file = open('printer.cfg', 'r')
config_data = file.read()
file.close()

lines = config_data.split('\n')

fileconfig = configparser.RawConfigParser(strict=False, inline_comment_prefixes=(';', '#'))
fileconfig.read_string(config_data)

fileconfig.read_dict