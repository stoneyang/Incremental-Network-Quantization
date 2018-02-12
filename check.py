from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import caffe

from utils import *

def weight_power(input_net, output_dir):
	for layer_name, param in input_net.params.iteritems():
		print type(layer_name)
		layer_power_file = output_dir + "{}_power.txt".format(layer_name)
		print layer_power_file
		f = open(layer_power_file, 'wb')
		filter = param[0].data.flat
		for i in filter:
			cur_pow = abs(i)
			# print cur_pow
			if abs(cur_pow - 0.0) == 0.0:
				f.write("{}\n".format(cur_pow))
			else:
				cur_pow = np.log2(cur_pow)
				if abs(i) + i == 0.0:
					f.write("-2 ^ {}\n".format(cur_pow))
				else:
					f.write("2 ^ {}\n".format(cur_pow))
		f.close()
	print "weight_power"

def main(model, weights, output_dir):
	caffe.set_mode_gpu()
	print "main"
	input_net = load_model(model, weights)
	weight_power(input_net, output_dir)

def parse_args():
	"""Parse input arguments
	"""

	parser = ArgumentParser(description=__doc__,
							formatter_class=ArgumentDefaultsHelpFormatter)

	parser.add_argument('model',
						help='model definition')
	parser.add_argument('input_weights',
						help='input weights')
	parser.add_argument('output_dir',
						help='output dir')

	args = parser.parse_args()
	return args
	
if __name__ == '__main__':
	args = parse_args()

	main(args.model, args.input_weights, args.output_dir)