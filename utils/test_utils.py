import utils.utils as U
import numpy as np
import math

import pytest

def test_episode_returns_0():
	rew = [0]
	gam = 1
	G = U.episode_returns(rew, gam)
	assert len(G) == 1
	assert math.isclose(G[0], 0.0)

def test_episode_returns_1():
	rew = [1, 1]
	gam = 1
	G = U.episode_returns(rew, gam)
	assert len(G) == 2
	assert math.isclose(G[0], 2.0)
	assert math.isclose(G[1], 1.0)

def test_episode_returns_3():
	rew = np.arange(100)
	gam = 0
	G = U.episode_returns(rew, gam)
	assert len(G) == 100
	for i, g in enumerate(G):
		assert math.isclose(i,g)

def test_episode_returns_4():
	rew = np.arange(1,6)
	gam = 0.9
	G = U.episode_returns(rew, gam)
	assert len(G) == 5
	assert math.isclose(G[4], 5)
	assert math.isclose(G[3], 4 + gam*5)
	assert math.isclose(G[2], 3 + gam*(4 + gam*5))
	assert math.isclose(G[1], 2 + gam*(3 + gam*(4 + gam*5)))
	assert math.isclose(G[0], 1 + gam*(2 + gam*(3 + gam*(4 + gam*5))))

def test_episode_returns_5():
	"""randoom, fixed seqeuence of rewards and gamma"""
	rew =  np.array([ 0.78911898, -0.81024471,  0.17957592, -0.68059733,  0.28141679])
	gam = 0.2741021090555267
	G = U.episode_returns(rew, gam)
	assert len(G) == 5
	assert math.isclose(G[4], 0.28141679)
	assert math.isclose(G[3], -0.68059733 + gam*0.28141679)
	assert math.isclose(G[2], 0.17957592 + gam*(-0.68059733 + gam*0.28141679))
	assert math.isclose(G[1], -0.81024471 + gam*(0.17957592 + gam*(-0.68059733 + gam*0.28141679)))
	assert math.isclose(G[0], 0.78911898 + gam*(-0.81024471 + gam*(0.17957592 + gam*(-0.68059733 + gam*0.28141679))))

@pytest.mark.parametrize(
	"rew,val,gam", 
	[([0], [0], 1),
	 ([0,0], [0,0], 1),
	 ([0,0,0], [0,0,0], 1),
	 (np.zeros(10), np.zeros(10), 1)]
	)
def test_episode_deltas_zeros(rew, val, gam):
	# rew = np.array([0])
	# val = np.array([0])
	# gam = 1
	D = U.episode_deltas(rew, val, gam)
	assert len(D) == len(rew)
	assert all([np.isclose(d, 0.0) for d in D])

@pytest.mark.parametrize(
	"rew,val,gam", 
	[
	([1], [1], 1),
	([1,1], [1,1], 1),
	([1,1,1], [1,1,1], 1),
	(np.ones(10), np.ones(10), 1)
	])
def test_episode_deltas_ones(rew, val, gam):
	D = U.episode_deltas(rew, val, gam)
	assert len(D) == len(rew)
	assert all([np.isclose(d, 1.0) for d in D[:-1]])


@pytest.mark.parametrize(
	"rew,val,gam,expected", 
	[
	([1,1], [0,0], 1, [1,1]),
	([1,1,1,1], [4,3,2,1], 1, np.zeros(4)),
	(np.array([ 8.21372432, -7.27595116,  0.91502172]), np.array([ 2.77273702,  0.6683472 , -7.55482125]), 0.8847159858013562, [6.032284751905578, -14.628169489746783, 8.46984297])
	])
def test_episode_deltas_variety(rew, val, gam, expected):
	D = U.episode_deltas(rew, val, gam)
	assert len(D) == len(expected)
	assert all([np.isclose(d, e) for d, e in zip(D, expected)])

@pytest.mark.parametrize(
	"rew,val,lam,gam,expected",[
	([0], [0], 1, 1, [0]),
	([0,0,0], [0,0,0], 1, 1, [0,0,0]),
	(np.array([4.8599633 , 9.30035417, 7.16784835, 2.3347474 ]), 
		np.array([-7.16107065,  8.14701989, -1.70720725, -6.19892907]),  
		0.892311659764275,  0.7109054318827468, 
		[ 21.750843840644524, 6.208014707695574, 9.881534141965549, 8.533676469364682])
	])
def test_episode_general_advantages(rew, val, lam, gam, expected):
	A = U.episode_general_advantages(rewards=rew, values=val, lam=lam, gamma=gam)
	assert all(np.isclose(a, e) for a, e in zip(A, expected))

@pytest.mark.parametrize(
	"rew,val,dne,lam,gam,nval,ndne,expected",[
	([0], [0], [0], 1, 1, 0, 1, [0]),
	([0,0,0], [0,0,0], [0,0,0], 1, 1, 0, 1, [0,0,0]),
	(	np.array([4.8599633 , 9.30035417, 7.16784835, 2.3347474 ]), 
		np.array([-7.16107065,  8.14701989, -1.70720725, -6.19892907]),
		[False, False, False, False],  
		0.892311659764275,  
		0.7109054318827468,
		-85.32492613257081,
		True, 
		[ 21.750843840644524, 6.208014707695574, 9.881534141965549, 8.533676469364682])
	])
def test_general_advantage_estimation_basics(rew, val, dne, lam, gam, nval, ndne, expected):
	rew = np.asarray(rew)
	val = np.asarray(val)
	dne = np.asarray(dne)
	advs, rtns, dels = U.general_advantage_estimation(rewards=rew, values=val, dones=dne, lam=lam, gamma=gam, next_value=nval, next_done=ndne)
	assert all(np.isclose(a, e) for a, e in zip(advs, expected))

