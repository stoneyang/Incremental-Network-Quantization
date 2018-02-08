import caffe

def parse_model(net):
	print "parse_model"
	for layer_name, param in net.params.iteritems():
		print layer_name + '\t' +  str(param[0].data.shape), str(param[1].data.shape)

def load_model(model, input):
	input_net = caffe.Net(model, input, caffe.TEST)
	print "load_model"
	parse_model(input_net)
	return input_net

def save_model(model, output):
	output_net = caffe.Net(model, caffe.TEST)
	output_net.save(output)
	print "save_model"