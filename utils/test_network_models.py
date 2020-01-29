import torch
import numpy as np
import scipy.stats as stats
from utils.network_models import DeepNetwork, GaussianActorNetwork
import pytest

def test_DeepNetwork_forward_1():
	""" build network and forward
	"""
	deepnet = DeepNetwork(1,1,[8,4],1)
	deepnet.forward(torch.from_numpy(np.random.rand(1)).float())


# TODO: convert this to hypothesis test
@pytest.mark.parametrize(
	"input_size,output_size,hidden_sizes,batch_shape",[
	(1, 1, [1], (10,1)),
	(8, 4, [4,2], (64,8))
	])
def test_DeepNetwork_forward_shape_checking(input_size, output_size, hidden_sizes, batch_shape):
	seed = np.random.randint(1e4)
	deepnet = DeepNetwork(input_size, output_size, hidden_sizes, seed)
	for i in range(100):
		vals = np.random.uniform(-10,10, batch_shape)
		output = deepnet.forward(torch.from_numpy(vals).float())
		assert output.shape == (batch_shape[0],output_size)


def test_GaussianActorNetwork_forward_1():
	""" build network and run forward method
	"""
	act_net = GaussianActorNetwork(1,1,[8,4],1)
	act_net.forward(torch.from_numpy(np.random.rand(1)).float())

def test_GaussianActorNetwork_forward_2():
	""" build network and run forward method to look at distribution
	"""

	# create actor network that outputs guassian distribution
	act_net = GaussianActorNetwork(10,1,[8,4],1)

	# generate a random observation to pass to network
	rand_observation = torch.from_numpy(np.random.rand(10)).float()

	# repeatadly pass observation to network to check that output falls in
	# normal distribution
	action_list = []
	for i in range(1000):
		act, _, _ = act_net.forward(rand_observation)
		action_list.append(act.data.numpy()[0])

	# check that random actions from the same input form normal distribution
	k2, p = stats.normaltest(action_list)
	assert p > 1e-2